#!/usr/bin/env python3
"""
ADALM-Pluto OFDM 彩色视频/图像传输 (5.8 GHz)
===============================================
参考 Pluto_picSend_58GHz.m 的 802.11a Non-HT 风格实现:
  - L-STF / L-LTF 前导码 (粗/精时序, 粗/精 CFO, LS 信道估计)
  - 64-FFT OFDM, 48 数据子载波 + 4 导频, CP = 16
  - QPSK 调制 (默认), 可改 16QAM
  - 每包帧头携带序号 + 长度 + CRC32
  - 30 MHz 空口采样 (20 * 1.5 过采样, 与 MATLAB 一致)
  - 视频帧 JPEG 压缩后分包传输, RX 端解码并显示

依赖: pip install pyadi-iio numpy scipy opencv-python
"""

import argparse
import os
import queue
import struct
import sys
import threading
import time
import zlib

import cv2
import numpy as np
from scipy import signal as sps

try:
    import adi
    HAS_PLUTO = True
except ImportError:
    HAS_PLUTO = False
    print("[警告] 未安装 pyadi-iio, 仅支持仿真模式")


# ============================================================================
# 全局射频/OFDM 配置
# ============================================================================

CARRIER_FREQ_DEFAULT = 5.8e9           # 与 MATLAB 一致
SAMPLE_RATE_BB = 20e6                  # 基带采样率
OSF = 1.5                              # 过采样因子
RF_RATE = int(SAMPLE_RATE_BB * OSF)    # Pluto 30 MHz
RF_BANDWIDTH = 18e6
TX_GAIN_DEFAULT = -5
RX_GAIN_DEFAULT = 40

NFFT = 64
NCP = 16
NSYM = NFFT + NCP          # 80

# 子载波分配 (DC=0, 索引 -32..31)
DATA_SC = np.array([
    -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -13, -12,
    -11, -10, -9, -8, -6, -5, -4, -3, -2, -1,
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 23, 24, 25, 26,
])  # -21, -7, 7, 21 用作导频
PILOT_SC = np.array([-21, -7, 7, 21])
USED_SC = np.sort(np.concatenate([DATA_SC, PILOT_SC]))
N_DATA = len(DATA_SC)      # 48
N_PILOT = len(PILOT_SC)    # 4

# 802.11a 标准 L-STF (12 个非零子载波, 间隔 4, 在 ±24 范围内)
LSTF_INDEX = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
LSTF_VAL = np.sqrt(13.0 / 6.0) * np.array([
    1 + 1j, -1 - 1j, 1 + 1j, -1 - 1j, -1 - 1j, 1 + 1j,
    -1 - 1j, -1 - 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j,
])

# 802.11a 标准 L-LTF (52 个非零子载波)
LLTF_VAL_NEG = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1,
                         -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])  # 子载波 -26..-1
LLTF_VAL_POS = np.array([1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1,
                         -1, -1, 1, -1, 1, -1, 1, 1, 1, 1])  # 子载波 1..26

PILOT_PATTERN = np.array([1, 1, 1, -1], dtype=np.complex64)

# 预计算 FFT 网格上的索引, 矢量化 OFDM 调制/解调用
_DATA_IDX = (DATA_SC % NFFT).astype(np.intp)
_PILOT_IDX = (PILOT_SC % NFFT).astype(np.intp)

# 帧头幻数 + 结构 (以字节为单位)
SYNC_MAGIC = 0xA55A3CC3
HEADER_FMT = "<IHHII"   # magic, frame_id, total_packets, packet_id, payload_len
HEADER_LEN = struct.calcsize(HEADER_FMT)   # 16 bytes
CRC_LEN = 4                                # CRC32

# QPSK Gray 映射, 平均功率 = 1
_QPSK_LUT = np.array([
    (1 + 1j), (1 - 1j), (-1 + 1j), (-1 - 1j),
], dtype=np.complex64) / np.sqrt(2)


# ============================================================================
# 比特/字节工具
# ============================================================================

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.int8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)
    idx = (pairs[:, 0] << 1) | pairs[:, 1]
    return _QPSK_LUT[idx]


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 2, dtype=np.int8)
    bits[0::2] = (np.real(symbols) < 0).astype(np.int8)
    bits[1::2] = (np.imag(symbols) < 0).astype(np.int8)
    return bits


# 16QAM Gray, 平均功率归一化为 1
# 比特排序: idx = (b3<<3)|(b2<<2)|(b1<<1)|b0, b3b2 控 I, b1b0 控 Q
# 沿用 QPSK 约定: bit=0 → 正向, bit=1 → 负向 ( I/Q < 0 ↔ bit=1 )
# I 轴 Gray: 00→+3, 01→+1, 11→-1, 10→-3 (Q 同理)
_QAM16_LUT = np.array([
     3 + 3j,  3 + 1j,  3 - 3j,  3 - 1j,
     1 + 3j,  1 + 1j,  1 - 3j,  1 - 1j,
    -3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
    -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
], dtype=np.complex64) / np.sqrt(10)


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    nibbles = bits.reshape(-1, 4)
    idx = ((nibbles[:, 0] << 3) | (nibbles[:, 1] << 2)
           | (nibbles[:, 2] << 1) | nibbles[:, 3])
    return _QAM16_LUT[idx]


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 4, dtype=np.int8)
    I = np.real(symbols) * np.sqrt(10)
    Q = np.imag(symbols) * np.sqrt(10)
    bits[0::4] = (I < 0).astype(np.int8)
    bits[1::4] = (np.abs(I) < 2).astype(np.int8)
    bits[2::4] = (Q < 0).astype(np.int8)
    bits[3::4] = (np.abs(Q) < 2).astype(np.int8)
    return bits


# 调制方式分发: 默认 QPSK
_MOD_BITS_PER_SYM = 2
_MOD_FN = qpsk_modulate
_DEMOD_FN = qpsk_demodulate
_MOD_NAME = "qpsk"


def set_modulation(name: str):
    """切换 QPSK / 16QAM. 必须在 TX 和 RX 端使用相同设置."""
    global _MOD_BITS_PER_SYM, _MOD_FN, _DEMOD_FN, _MOD_NAME
    name = name.lower()
    if name == "qpsk":
        _MOD_BITS_PER_SYM = 2
        _MOD_FN = qpsk_modulate
        _DEMOD_FN = qpsk_demodulate
    elif name in ("16qam", "qam16"):
        _MOD_BITS_PER_SYM = 4
        _MOD_FN = qam16_modulate
        _DEMOD_FN = qam16_demodulate
        name = "16qam"
    else:
        raise ValueError(f"Unknown modulation: {name}")
    _MOD_NAME = name


# ============================================================================
# 前导码生成 (L-STF / L-LTF)
# ============================================================================

def _build_freq_grid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """把 (子载波索引, 值) 填到 NFFT 长度的频域格上."""
    grid = np.zeros(NFFT, dtype=np.complex64)
    for k, v in zip(indices, values):
        grid[k % NFFT] = v
    return grid


def make_lstf_time() -> np.ndarray:
    """生成 L-STF 时域: 10 个相同的 16 sample 短训练符号, 总长 160."""
    grid = _build_freq_grid(LSTF_VAL, LSTF_INDEX)
    t = np.fft.ifft(grid) * NFFT / np.sqrt(12)
    short = t[:16]
    return np.tile(short, 10).astype(np.complex64)


def make_lltf_time() -> np.ndarray:
    """生成 L-LTF: 32 sample GI + 2 个 64 sample 长训练, 总长 160."""
    indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    values = np.concatenate([LLTF_VAL_NEG, LLTF_VAL_POS]).astype(np.complex64)
    grid = _build_freq_grid(values, indices)
    t = np.fft.ifft(grid) * NFFT / np.sqrt(52)
    sym = t.astype(np.complex64)        # 64 samples
    gi = sym[-32:]
    return np.concatenate([gi, sym, sym]).astype(np.complex64)


def lltf_freq_reference() -> np.ndarray:
    """L-LTF 频域参考: 用于信道估计."""
    indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    values = np.concatenate([LLTF_VAL_NEG, LLTF_VAL_POS]).astype(np.complex64)
    grid = _build_freq_grid(values, indices)
    return grid


LSTF_TIME = make_lstf_time()
LLTF_TIME = make_lltf_time()
LLTF_FREQ_REF = lltf_freq_reference()
PREAMBLE = np.concatenate([LSTF_TIME, LLTF_TIME]).astype(np.complex64)
LEN_LSTF = 160
LEN_LLTF = 160
LEN_PREAMBLE = LEN_LSTF + LEN_LLTF       # 320


# ============================================================================
# OFDM 调制 / 解调
# ============================================================================

def ofdm_modulate(data_symbols: np.ndarray) -> np.ndarray:
    """
    把 QPSK 符号流映射到 OFDM 时域. 矢量化实现.
    输入: 一维复符号数组, 长度必须是 N_DATA 的整数倍.
    输出: 复时域信号 (每符号 NSYM samples).
    """
    n_sym = len(data_symbols) // N_DATA
    if n_sym == 0:
        return np.zeros(0, dtype=np.complex64)
    data_symbols = data_symbols[: n_sym * N_DATA].reshape(n_sym, N_DATA)
    grid = np.zeros((n_sym, NFFT), dtype=np.complex64)
    grid[:, _DATA_IDX] = data_symbols
    sign = (1 - 2 * (np.arange(n_sym) % 2)).astype(np.complex64)
    grid[:, _PILOT_IDX] = sign[:, None] * PILOT_PATTERN[None, :]
    t = np.fft.ifft(grid, axis=1) * (NFFT / np.sqrt(N_DATA + N_PILOT))
    sym = np.concatenate([t[:, -NCP:], t], axis=1)
    return sym.astype(np.complex64).reshape(-1)


def ofdm_demodulate(time_signal: np.ndarray, h_est: np.ndarray) -> np.ndarray:
    """OFDM 解调 (频域单抽头均衡 + 导频残留 CFO 校正), 矢量化实现."""
    n_sym = len(time_signal) // NSYM
    if n_sym == 0:
        return np.zeros(0, dtype=np.complex64)
    time_signal = time_signal[: n_sym * NSYM].reshape(n_sym, NSYM)
    syms = time_signal[:, NCP:]
    Y = np.fft.fft(syms, axis=1) / np.sqrt(N_DATA + N_PILOT)
    Y = Y / (h_est[None, :] + 1e-9)
    pilots_rx = Y[:, _PILOT_IDX]
    sign = (1 - 2 * (np.arange(n_sym) % 2)).astype(np.complex64)
    pilots_tx = sign[:, None] * PILOT_PATTERN[None, :]
    phi = np.angle(np.sum(pilots_rx * np.conj(pilots_tx), axis=1))
    Y = Y * np.exp(-1j * phi)[:, None]
    return Y[:, _DATA_IDX].astype(np.complex64).reshape(-1)


# ============================================================================
# 包打包: bytes -> 完整 baseband 波形
# ============================================================================

def build_packet_waveform(payload_bytes: bytes,
                          frame_id: int,
                          packet_id: int,
                          total_packets: int):
    """组装一个包: [前导码 | header(1+ OFDM sym) | data] -> 时域波形."""
    payload_len = len(payload_bytes)
    header = struct.pack(HEADER_FMT, SYNC_MAGIC, frame_id & 0xFFFF,
                         total_packets & 0xFFFF, packet_id & 0xFFFFFFFF,
                         payload_len & 0xFFFFFFFF)
    crc = zlib.crc32(header + payload_bytes).to_bytes(4, "little")
    full = header + payload_bytes + crc

    bits = bytes_to_bits(full)
    bits_per_sym = N_DATA * _MOD_BITS_PER_SYM
    pad = (-len(bits)) % bits_per_sym
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    n_sym = len(bits) // bits_per_sym

    syms = _MOD_FN(bits)
    data_time = ofdm_modulate(syms)

    waveform = np.concatenate([PREAMBLE, data_time]).astype(np.complex64)
    return waveform, n_sym


# ============================================================================
# 包检测/同步/解码
# ============================================================================

def detect_packets(rx: np.ndarray, threshold: float = 0.5):
    """用 L-STF 16-sample 自相关做粗包检测, 返回 plateau 起始索引列表."""
    L = 16
    M = 64
    n = len(rx)
    if n < L * 12:
        return []
    a = rx[:-L]
    b = np.conj(rx[L:])
    prod = a * b
    energy = np.abs(rx[L:]) ** 2
    cumsum = np.concatenate([[0], np.cumsum(prod)])
    cumE = np.concatenate([[0], np.cumsum(energy)])
    csum = cumsum[M:] - cumsum[:-M]
    cE = cumE[M:] - cumE[:-M]
    metric = np.abs(csum) / (cE + 1e-9)
    above = metric > threshold
    indices = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            indices.append(i)              # 返回 plateau 左边沿 (≈ L-STF 起点)
            i = j + 32
        else:
            i += 1
    return indices


def apply_cfo(x: np.ndarray, freq_per_sample: float) -> np.ndarray:
    n = np.arange(len(x))
    return (x * np.exp(-1j * freq_per_sample * n)).astype(np.complex64)


def find_lts_position(rx: np.ndarray, search_center: int,
                      search_radius: int = 200):
    """
    用 L-LTF 64-sample 模板互相关找精确的 LTS 起点.
    返回 (lts1_pos, lts2_pos, peak_val), lts1_pos 是第一个 LTS 起点(绝对索引),
    或 None 表示找不到.
    """
    template = LLTF_TIME[32:32 + 64]   # 第一个 64 sample LTS
    s = max(0, search_center - search_radius)
    e = min(len(rx), search_center + search_radius + 64)
    if e - s < 128 + 64:
        return None
    seg = rx[s:e]
    corr = np.correlate(seg, template, mode="valid")
    abs_corr = np.abs(corr)
    if len(abs_corr) < 65:
        return None

    # 找最高峰
    peak1 = int(np.argmax(abs_corr))
    # 在距离 peak1 +64 处也应该有一个相近的峰 (第二个 LTS)
    cand2_a = peak1 + 64
    cand2_b = peak1 - 64
    second = None
    if 0 <= cand2_a < len(abs_corr):
        # 在 ±2 范围内找第二峰
        lo = max(0, cand2_a - 2)
        hi = min(len(abs_corr), cand2_a + 3)
        loc = lo + int(np.argmax(abs_corr[lo:hi]))
        if abs_corr[loc] > 0.5 * abs_corr[peak1]:
            second = loc
            lts1 = peak1
    if second is None and 0 <= cand2_b < len(abs_corr):
        lo = max(0, cand2_b - 2)
        hi = min(len(abs_corr), cand2_b + 3)
        loc = lo + int(np.argmax(abs_corr[lo:hi]))
        if abs_corr[loc] > 0.5 * abs_corr[peak1]:
            second = peak1
            lts1 = loc
            peak1 = loc
    if second is None:
        return None
    lts1_abs = s + lts1
    lts2_abs = s + second
    return lts1_abs, lts2_abs, float(abs_corr[peak1])


def coarse_cfo_from_lstf(rx_lstf: np.ndarray) -> float:
    """L-STF 自相关粗 CFO (rad/sample). 输入应为 ≥ 144 sample 的 L-STF."""
    L = 16
    n = min(len(rx_lstf) - L, 9 * L)
    if n <= 0:
        return 0.0
    s = np.sum(np.conj(rx_lstf[:n]) * rx_lstf[L:L + n])
    return np.angle(s) / L


def fine_cfo_from_lts(rx_lts1: np.ndarray, rx_lts2: np.ndarray) -> float:
    """两个 64 sample LTS 的相位差估计精 CFO (rad/sample)."""
    return np.angle(np.sum(np.conj(rx_lts1) * rx_lts2)) / 64.0


def estimate_channel_from_lts(rx_lts1: np.ndarray,
                              rx_lts2: np.ndarray) -> np.ndarray:
    """LS 信道估计."""
    avg = 0.5 * (rx_lts1 + rx_lts2)
    H = np.fft.fft(avg) / np.sqrt(N_DATA + N_PILOT)
    H_ref = LLTF_FREQ_REF
    h_est = np.ones(NFFT, dtype=np.complex64)
    for sc in np.concatenate([np.arange(-26, 0), np.arange(1, 27)]):
        if H_ref[sc % NFFT] != 0:
            h_est[sc % NFFT] = H[sc % NFFT] / H_ref[sc % NFFT]
    return h_est


def decode_packet(rx: np.ndarray, start: int, max_data_sym: int = 600):
    """
    给定一个粗略检测点 start (≈ L-STF 起点附近), 用 LTS 互相关精确定位并解码.
    返回 (frame_id, packet_id, total_packets, payload, ok) 或 None.
    """
    if start < 0 or start + LEN_PREAMBLE > len(rx):
        return None

    # 1. LTS 精确定位 (在 start 周围搜索)
    pos = find_lts_position(rx, start + 192, search_radius=120)
    if pos is None:
        return None
    lts1_pos, lts2_pos, _ = pos

    lstf_start = lts1_pos - 32 - 160
    if lstf_start < 0:
        return None
    data_start = lts2_pos + 64
    if data_start + NSYM * 2 > len(rx):
        return None

    # 2. 粗 CFO (用整段 L-STF, 但跳过最后 16 sample 边界)
    lstf_seg = rx[lstf_start:lstf_start + 144]
    coarse = coarse_cfo_from_lstf(lstf_seg)

    # 3. 取 LTS 段, 应用粗 CFO
    seg_lts = rx[lts1_pos - 32:lts1_pos - 32 + 160]
    seg_lts = apply_cfo(seg_lts, coarse)
    rx_lts1 = seg_lts[32:32 + 64]
    rx_lts2 = seg_lts[32 + 64:32 + 128]

    # 4. 精 CFO
    fine = fine_cfo_from_lts(rx_lts1, rx_lts2)
    total_cfo = coarse + fine

    # 5. 信道估计 (再用精 CFO 校正过的 LTS)
    rx_lts1_c = apply_cfo(rx_lts1, fine)
    rx_lts2_c = (apply_cfo(rx_lts2, fine)
                 * np.exp(-1j * fine * 64))   # 补偿 LTS2 起始相位
    h_est = estimate_channel_from_lts(rx_lts1_c, rx_lts2_c)

    bits_per_sym = N_DATA * _MOD_BITS_PER_SYM

    # 6. 先解 header (按当前 mod 计算 OFDM 符号数)
    head_sym_count = (HEADER_LEN * 8 + bits_per_sym - 1) // bits_per_sym
    if data_start + NSYM * head_sym_count > len(rx):
        return None
    # 应用总 CFO 到数据段, 起点对齐
    data_chunk = rx[data_start:data_start + NSYM * (max_data_sym + 4)].copy()
    if len(data_chunk) < NSYM * head_sym_count:
        return None
    n_off = data_start - lstf_start          # 已经过的样本数, 用于相位连续
    phase = np.exp(-1j * total_cfo *
                   (np.arange(len(data_chunk)) + n_off)).astype(np.complex64)
    data_chunk = data_chunk * phase

    head_time = data_chunk[:NSYM * head_sym_count]
    head_syms = ofdm_demodulate(head_time, h_est)
    head_bits = _DEMOD_FN(head_syms)
    head_bytes = bits_to_bytes(head_bits[: HEADER_LEN * 8])

    try:
        magic, frame_id, total_packets, packet_id, payload_len = struct.unpack(
            HEADER_FMT, head_bytes)
    except struct.error:
        return None

    if magic != SYNC_MAGIC:
        return None
    if payload_len > 60_000 or total_packets > 4096 or total_packets == 0:
        return None
    if packet_id >= total_packets:
        return None

    full_bytes_needed = HEADER_LEN + payload_len + CRC_LEN
    n_sym_total = (full_bytes_needed * 8 + bits_per_sym - 1) // bits_per_sym
    if NSYM * n_sym_total > len(data_chunk):
        return None
    data_time = data_chunk[:NSYM * n_sym_total]
    data_syms = ofdm_demodulate(data_time, h_est)
    data_bits = _DEMOD_FN(data_syms)
    full = bits_to_bytes(data_bits[: full_bytes_needed * 8])

    payload = full[HEADER_LEN: HEADER_LEN + payload_len]
    crc_rx = int.from_bytes(full[HEADER_LEN + payload_len:
                                  HEADER_LEN + payload_len + CRC_LEN],
                            "little")
    crc_calc = zlib.crc32(full[: HEADER_LEN + payload_len])
    ok = (crc_rx == crc_calc)
    return frame_id, packet_id, total_packets, payload, ok


# ============================================================================
# 过采样 / 重采样 (基带 <-> 30MHz 空口)
# ============================================================================

def upsample_to_rf(x: np.ndarray) -> np.ndarray:
    """20MHz -> 30MHz, 用有理变换 3/2."""
    return sps.resample_poly(x, up=3, down=2).astype(np.complex64)


def downsample_to_bb(x: np.ndarray) -> np.ndarray:
    """30MHz -> 20MHz, 用有理变换 2/3."""
    return sps.resample_poly(x, up=2, down=3).astype(np.complex64)


# ============================================================================
# 视频帧 <-> 字节流 (JPEG 压缩)
# ============================================================================

JPEG_QUALITY = 75
MAX_PAYLOAD_PER_PKT = 1200


def frame_to_jpeg_bytes(frame: np.ndarray, width: int, height: int,
                        quality: int = JPEG_QUALITY,
                        codec: str = "jpg") -> bytes:
    """编码一帧到字节流. codec='jpg' 兼容旧行为, 'webp' 同字节数下画质更好.

    cv2.imdecode 自动识别两种格式, 接收端不用改.
    WebP @ q60 ~= JPEG @ q80 视觉质量, 字节数差不多.
    """
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    if codec == "webp":
        ok, buf = cv2.imencode(".webp", frame,
                               [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def jpeg_bytes_to_frame(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # auto-detect jpg/webp


def split_payload(data: bytes, max_size: int = MAX_PAYLOAD_PER_PKT):
    return [data[i:i + max_size] for i in range(0, len(data), max_size)]


# ============================================================================
# Pluto 硬件接口
# ============================================================================

class PlutoTX:
    """TX-only Pluto. 默认 *非 cyclic 流式* 模式: 每次 push 同尺寸 buffer,
    libiio 内部复用 DMA, 没有 destroy/create 间隙. 主线程预先把 baseband
    upsample+normalize+cast 成 int IQ 传进来, worker 只调 sdr.tx()."""

    def __init__(self, uri: str, freq: float, tx_gain: float,
                 streaming: bool = True):
        if not HAS_PLUTO:
            raise RuntimeError("pyadi-iio 未安装")
        print(f"[TX]  连接 Pluto {uri} ...")
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = RF_RATE
        self.sdr.tx_lo = int(freq)
        self.sdr.tx_rf_bandwidth = int(RF_BANDWIDTH)
        self.sdr.tx_hardwaregain_chan0 = float(tx_gain)
        self.sdr.tx_cyclic_buffer = not streaming
        # 关掉 TX 板自己的 RX AGC, 这块板只发不收
        try:
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = 0
        except Exception:
            pass
        self.streaming = streaming
        self._pushed = False
        mode = "streaming(non-cyclic)" if streaming else "cyclic"
        print(f"[TX]  就绪  Freq={freq/1e9:.2f}GHz  TX={tx_gain}dB  {mode}")

    def push_iq(self, tx_iq: np.ndarray):
        """喂已经 upsample/normalize/cast 好的 int 复数 IQ. cyclic 时仍需 destroy."""
        if not self.streaming and self._pushed:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
        self.sdr.tx(tx_iq)
        self._pushed = True

    def push(self, baseband: np.ndarray):
        """兼容旧接口: 拿 baseband 自己 upsample (cyclic 模式仍可用)."""
        rf = upsample_to_rf(baseband)
        peak = np.max(np.abs(rf))
        if peak > 0:
            rf = rf / peak * 0.7
        tx_iq = (rf * (2 ** 14)).astype(np.complex64)
        if not self.streaming and self._pushed:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
        self.sdr.tx(tx_iq)
        self._pushed = True

    def close(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass


class PlutoRX:
    """RX-only Pluto wrapper. capture() 阻塞至 rx_buffer_size 填满."""

    def __init__(self, uri: str, freq: float, rx_gain: float,
                 rx_buffer: int = 2 ** 18):
        if not HAS_PLUTO:
            raise RuntimeError("pyadi-iio 未安装")
        print(f"[RX]  连接 Pluto {uri} ...")
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = RF_RATE
        self.sdr.rx_lo = int(freq)
        self.sdr.rx_rf_bandwidth = int(RF_BANDWIDTH)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = float(rx_gain)
        self.sdr.rx_buffer_size = rx_buffer
        # *** 关键 ***: 静音 RX 板自己的 TX 通道.
        # Pluto 的 TX DAC 永远在跑, 默认 tx_hardwaregain_chan0 = 0dB (满功率),
        # 即使没 push 任何 sample, TX_LO 泄漏 + DAC 噪底也会从 TX 天线辐射出来,
        # 直接耦合进自己的 RX 前端, 把对面 Pluto 的目标信号淹没.
        # 双 Pluto 必须把 RX 板的 TX_GAIN 拉到 -89.75 (有效静音).
        try:
            self.sdr.tx_hardwaregain_chan0 = -89.75
            # 同时把 TX_LO 挪到远离 RX_LO 的频段, 进一步降低 LO 泄漏
            far_lo = int(freq) + int(200e6)
            if 70e6 < far_lo < 6e9:
                self.sdr.tx_lo = far_lo
        except Exception as e:
            print(f"[RX]  警告: 无法静音 TX 通道 ({e}), 信号可能被自干扰")
        for _ in range(3):
            try:
                self.sdr.rx()
            except Exception:
                pass
        print(f"[RX]  就绪  Freq={freq/1e9:.2f}GHz  RX={rx_gain}dB  "
              f"buf={rx_buffer}  TX_GAIN=-89.75dB(silenced)")

    def capture(self) -> np.ndarray:
        rf = self.sdr.rx()
        rf = np.asarray(rf, dtype=np.complex64) / (2 ** 14)
        return downsample_to_bb(rf)

    def close(self):
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass


class PlutoTRX:
    def __init__(self, uri: str, freq: float, tx_gain: float, rx_gain: float,
                 rx_buffer: int = 2 ** 18):
        if not HAS_PLUTO:
            raise RuntimeError("pyadi-iio 未安装")
        print(f"[TRX] 连接 Pluto {uri} ...")
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = RF_RATE
        self.sdr.tx_lo = int(freq)
        self.sdr.rx_lo = int(freq)
        self.sdr.tx_rf_bandwidth = int(RF_BANDWIDTH)
        self.sdr.rx_rf_bandwidth = int(RF_BANDWIDTH)
        self.sdr.tx_hardwaregain_chan0 = float(tx_gain)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = float(rx_gain)
        self.sdr.rx_buffer_size = rx_buffer
        self.sdr.tx_cyclic_buffer = True
        self._tx_pushed = False
        for _ in range(3):
            try:
                self.sdr.rx()
            except Exception:
                pass
        print(f"[TRX] 就绪  Freq={freq/1e9:.2f}GHz  SR={RF_RATE/1e6:.1f}MHz  "
              f"TX={tx_gain}dB  RX={rx_gain}dB")

    def push_tx(self, baseband: np.ndarray):
        rf = upsample_to_rf(baseband)
        peak = np.max(np.abs(rf))
        if peak > 0:
            rf = rf / peak * 0.7
        tx_iq = (rf * (2 ** 14)).astype(np.complex64)
        if self._tx_pushed:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
        self.sdr.tx(tx_iq)
        # 关键: 重建 RX 队列, 丢弃 push 之前堆积的陈旧采样.
        # 否则迭代 ~0.4s 但每次只 capture ~17ms, libiio ring buffer
        # 几十次后溢出, sdr.rx() 持续返回几秒前的 IQ → 解码到旧 burst,
        # 数据无法对齐, 接收完全停滞.
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self._tx_pushed = True

    def capture(self) -> np.ndarray:
        rf = self.sdr.rx()
        rf = np.asarray(rf, dtype=np.complex64) / (2 ** 14)
        bb = downsample_to_bb(rf)
        return bb

    def close(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass


# ============================================================================
# AWGN 信道 (仿真模式)
# ============================================================================

def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(np.abs(signal) ** 2) + 1e-12
    n_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(n_p / 2) * (np.random.randn(len(signal))
                                + 1j * np.random.randn(len(signal)))
    return (signal + noise).astype(np.complex64)


# ============================================================================
# 视频源
# ============================================================================

def open_video_source(source: str, video_path: str = ""):
    if source == "camera":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
        cap.release()
        print("[警告] 摄像头不可用, 改用 movie.mp4")
        source = "file"
    if source == "file":
        path = video_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "movie.mp4")
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                return cap
            cap.release()
        print(f"[警告] 视频文件 {path} 不可用, 改用测试图案")
    return None


def get_next_frame(cap, frame_idx: int):
    if cap is None:
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            img[y, :, 0] = (y * 2 + frame_idx * 5) % 256
            img[y, :, 1] = (y + frame_idx * 3) % 256
            img[y, :, 2] = (255 - y + frame_idx * 7) % 256
        cv2.putText(img, f"#{frame_idx:04d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


# ============================================================================
# 仿真模式: AWGN
# ============================================================================

def run_simulation(args):
    cap = open_video_source(args.source, args.input)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)

    frame_idx = 0
    tx_count = 0
    rx_count = 0
    t0 = time.time()
    print(f"[SIM] 仿真 SNR={args.snr}dB  分辨率={args.fwidth}x{args.fheight}  "
          f"JPEG质量={args.jpeg}")

    while time.time() - t0 < args.duration:
        frame = get_next_frame(cap, frame_idx)
        if frame is None:
            break

        jpg = frame_to_jpeg_bytes(frame, args.fwidth, args.fheight,
                                      args.jpeg, codec=args.codec)
        chunks = split_payload(jpg)
        n_pkt = len(chunks)

        recv_chunks = [None] * n_pkt
        for pid, chunk in enumerate(chunks):
            wf, _ = build_packet_waveform(chunk, frame_idx, pid, n_pkt)
            pad_pre = np.random.randint(50, 500)
            pad_post = np.random.randint(50, 500)
            air = np.concatenate([
                np.zeros(pad_pre, dtype=np.complex64), wf,
                np.zeros(pad_post, dtype=np.complex64)
            ])
            air = awgn(air, args.snr)

            indices = detect_packets(air)
            for st in indices:
                res = decode_packet(air, st)
                if res is None:
                    continue
                fid, pkt_id, tot, payload, ok = res
                if ok and fid == (frame_idx & 0xFFFF) and pkt_id == pid:
                    recv_chunks[pid] = payload
                    break
        tx_count += 1
        if all(c is not None for c in recv_chunks):
            jpg_rx = b"".join(recv_chunks)
            rx_frame = jpeg_bytes_to_frame(jpg_rx)
            if rx_frame is not None:
                rx_count += 1
                cv2.imshow("RX", cv2.resize(rx_frame, (480, 360), interpolation=cv2.INTER_CUBIC))

        cv2.imshow("TX", cv2.resize(frame, (480, 360)))
        key = cv2.waitKey(1)
        if key == 27:
            break
        elapsed = time.time() - t0
        print(f"\r[SIM] TX#{frame_idx} 包={n_pkt} JPEG={len(jpg)}B  "
              f"成功={rx_count}/{tx_count}  {elapsed:.1f}s", end="")
        frame_idx += 1

    print()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# 单台 Pluto 收发回环
# ============================================================================

def run_loopback(args):
    if not HAS_PLUTO:
        print("[错误] 未安装 pyadi-iio, 无法使用 loopback")
        return
    trx = PlutoTRX(args.uri, args.freq * 1e6, args.gain, args.rx_gain,
                   rx_buffer=2 ** 18)
    cap = open_video_source(args.source, args.input)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)

    frame_idx = 0
    tx_count = 0
    rx_count = 0
    last_rx_frame = None
    t0 = time.time()
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1

    try:
        while time.time() - t0 < args.duration:
            frame = get_next_frame(cap, frame_idx)
            if frame is None:
                break
            jpg = frame_to_jpeg_bytes(frame, args.fwidth, args.fheight,
                                      args.jpeg, codec=args.codec)
            chunks = split_payload(jpg)
            n_pkt = len(chunks)

            silence = np.zeros(NSYM * 4, dtype=np.complex64)
            burst_parts = []
            for pid, chunk in enumerate(chunks):
                wf, _ = build_packet_waveform(chunk, frame_idx, pid, n_pkt)
                burst_parts.append(wf)
                burst_parts.append(silence)
            burst = np.concatenate(burst_parts).astype(np.complex64)

            max_tx_samples = 2 ** 18
            if len(burst) > max_tx_samples:
                print(f"\n[WARN] burst {len(burst)} > {max_tx_samples}, "
                      f"已截断, 请降低 jpeg/分辨率或减小 MAX_PAYLOAD_PER_PKT")
                burst = burst[:max_tx_samples]

            try:
                trx.push_tx(burst)
                tx_count += 1
            except Exception as e:
                print(f"\n[ERR] TX: {e}")
                continue

            try:
                rx = trx.capture()
            except Exception:
                continue

            indices = detect_packets(rx, threshold=0.55)
            frame_done = False
            for st in indices:
                if frame_done:
                    break
                res = decode_packet(rx, st)
                if res is None:
                    continue
                fid, pkt_id, tot, payload, ok = res
                if not ok:
                    continue
                if tot == 0 or pkt_id >= tot:
                    continue
                if fid != last_frame_id:
                    payloads_buf = {}
                    last_total = tot
                    last_frame_id = fid
                payloads_buf[pkt_id] = payload
                if last_total > 0 and all(
                        i in payloads_buf for i in range(last_total)):
                    full = b"".join(payloads_buf[i]
                                    for i in range(last_total))
                    rx_frame = jpeg_bytes_to_frame(full)
                    if rx_frame is not None:
                        last_rx_frame = rx_frame
                        rx_count += 1
                    payloads_buf = {}
                    last_frame_id = -1
                    frame_done = True

            cv2.imshow("TX", cv2.resize(frame, (480, 360)))
            if last_rx_frame is not None:
                cv2.imshow("RX", cv2.resize(last_rx_frame, (480, 360), interpolation=cv2.INTER_CUBIC))
            key = cv2.waitKey(1)
            if key == 27:
                break

            elapsed = time.time() - t0
            print(f"\r[LOOP] TX#{frame_idx} 包={n_pkt} JPEG={len(jpg)}B  "
                  f"成功帧={rx_count}/{tx_count}  {elapsed:.1f}s", end="")
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        trx.close()
    print(f"\n[结果] 发射 {tx_count} 帧, 成功 {rx_count} 帧")


# ============================================================================
# 异步线程化模式 (单台 / 双台 Pluto 共用解码逻辑)
# ============================================================================

_BURST_LIMIT = 2 ** 18      # baseband sample 上限, 与 push_tx 容量一致


class FrameProducer:
    """主线程: 取一帧 -> JPEG 压缩 -> 切包 -> 拼成空口 burst.

    fixed_bb_size: 非零时把 baseband burst 零填充到固定长度.
    pre_upsample: True 时主线程做完 upsample/normalize/cast, 把准备好的
        int 复数 IQ 给 worker; False 时返回 baseband, 由 worker (PlutoTRX)
        自己 upsample.
    """

    def __init__(self, args, cap, fixed_bb_size: int = 0,
                 pre_upsample: bool = False):
        self.args = args
        self.cap = cap
        self.idx = 0
        self.fixed_bb_size = fixed_bb_size
        self.pre_upsample = pre_upsample

    def next(self):
        frame = get_next_frame(self.cap, self.idx)
        if frame is None:
            return None
        jpg = frame_to_jpeg_bytes(frame, self.args.fwidth, self.args.fheight,
                                  self.args.jpeg, codec=self.args.codec)
        chunks = split_payload(jpg)
        silence = np.zeros(NSYM * 4, dtype=np.complex64)
        parts = []
        for pid, chunk in enumerate(chunks):
            wf, _ = build_packet_waveform(chunk, self.idx, pid, len(chunks))
            parts.append(wf)
            parts.append(silence)
        burst = np.concatenate(parts).astype(np.complex64)
        if len(burst) > _BURST_LIMIT:
            print(f"\n[WARN] burst {len(burst)} > {_BURST_LIMIT}, 截断")
            burst = burst[:_BURST_LIMIT]
        # 流式模式: 零填充到固定 size, 让 DMA buffer 可复用
        if self.fixed_bb_size > 0 and len(burst) < self.fixed_bb_size:
            pad = np.zeros(self.fixed_bb_size - len(burst), dtype=np.complex64)
            burst = np.concatenate([burst, pad])
        elif self.fixed_bb_size > 0 and len(burst) > self.fixed_bb_size:
            burst = burst[:self.fixed_bb_size]
        # 仅在 pre_upsample 模式下 (dual) 主线程做 upsample/normalize/cast.
        # async 模式下 PlutoTRX.push_tx 自己做, 这里要保留 baseband.
        if self.pre_upsample:
            rf = upsample_to_rf(burst)
            peak = float(np.max(np.abs(rf)))
            if peak > 0:
                rf = rf / peak * 0.7
            burst = (rf * (2 ** 14)).astype(np.complex64)
        out = (frame, burst, self.idx, len(chunks), len(jpg))
        self.idx += 1
        return out


def _replace_latest(q: queue.Queue, item) -> bool:
    """有界 queue 上的 drop-oldest put: 总是让 q 里只剩最新的 item."""
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            return False


def _drain_to_latest(q: queue.Queue):
    """阻塞拿最早一个, 然后非阻塞继续拿到队空, 返回最后一个 (= 最新)."""
    item = q.get()
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def _rx_decode_worker(capture_fn, rx_q: queue.Queue,
                      stop_event: threading.Event,
                      stats: dict):
    """专职捕包+解码线程. capture_fn() 阻塞返回基带 IQ."""
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    while not stop_event.is_set():
        try:
            samples = capture_fn()
        except Exception:
            continue
        stats["captures"] = stats.get("captures", 0) + 1
        # 滚动平均的信号强度, 帮助诊断 RX 是否真的有信号进来
        rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
        peak = float(np.max(np.abs(samples)))
        stats["last_rms"] = rms
        stats["last_peak"] = peak
        indices = detect_packets(samples, threshold=0.55)
        stats["last_dets"] = len(indices)
        frame_done = False
        for st in indices:
            if frame_done:
                break
            res = decode_packet(samples, st)
            if res is None:
                continue
            fid, pkt_id, tot, payload, ok = res
            if not ok or tot == 0 or pkt_id >= tot:
                continue
            if fid != last_frame_id:
                payloads_buf = {}
                last_total = tot
                last_frame_id = fid
            payloads_buf[pkt_id] = payload
            if last_total > 0 and all(i in payloads_buf
                                       for i in range(last_total)):
                full = b"".join(payloads_buf[i]
                                for i in range(last_total))
                rx_frame = jpeg_bytes_to_frame(full)
                if rx_frame is not None:
                    _replace_latest(rx_q, rx_frame)
                payloads_buf = {}
                last_frame_id = -1
                frame_done = True


def _tx_stream_worker(push_fn, tx_q: queue.Queue,
                      stop_event: threading.Event,
                      stats: dict):
    """非 cyclic 流式: 紧循环反复 push 最新 burst, libiio 队列控速.

    硬件以 sample_rate 消费, 每次 tx() 成功就是一次 buffer-emit;
    内容更新 = 主线程把新 burst 放进 tx_q, worker 切换到新指针.
    stats 统计:
      pushed       = 内容更新次数 (= 真正空口"独立帧"数)
      stream_calls = sdr.tx() 调用次数 (= 硬件 buffer emit 次数, 受 USB 速率限)
    """
    current = None
    while not stop_event.is_set():
        # 非阻塞 drain 到最新内容
        new = None
        while True:
            try:
                new = tx_q.get_nowait()
            except queue.Empty:
                break
        if new is not None:
            current = new
            stats["pushed"] = stats.get("pushed", 0) + 1
        if current is None:
            time.sleep(0.005)
            continue
        try:
            push_fn(current)
            stats["stream_calls"] = stats.get("stream_calls", 0) + 1
        except Exception as e:
            print(f"\n[ERR] stream push: {e}")
            time.sleep(0.005)


def _tx_push_worker(push_fn, tx_q: queue.Queue,
                    stop_event: threading.Event,
                    stats: dict,
                    min_period: float = 0.0):
    """专职发射线程. 阻塞拿 burst, drain 到最新, 调用 push_fn.

    min_period: 两次 push 之间最短间隔. cyclic TX 切 buffer 时
    destroy+create 大约 30-40ms 静默, 期间 RX 抓不到信号.
    强制留出 >=2x 这个时间, 确保 cyclic 有足够稳态窗口给 RX 解包.
    """
    last_push = 0.0
    while not stop_event.is_set():
        try:
            burst = tx_q.get(timeout=0.1)
        except queue.Empty:
            continue
        # drain stale bursts queued while previous push was running
        while True:
            try:
                burst = tx_q.get_nowait()
            except queue.Empty:
                break
        if min_period > 0:
            wait = min_period - (time.time() - last_push)
            if wait > 0:
                time.sleep(wait)
        try:
            push_fn(burst)
            last_push = time.time()
            stats["pushed"] = stats.get("pushed", 0) + 1
        except Exception as e:
            print(f"\n[ERR] TX push: {e}")


def _run_threaded(args, push_fn, capture_fn, close_fns, mode_label: str,
                  streaming: bool = False, fixed_bb_size: int = 0,
                  pre_upsample: bool = False):
    """异步主体: 主线程产帧+显示, TX/RX 各一个 worker.

    streaming=True 时走 _tx_stream_worker (非 cyclic, 紧循环 push), 配合
    fixed_bb_size 让每个 burst 都是同 size, libiio 复用 DMA buffer.
    pre_upsample: producer 是否在主线程预先 upsample/normalize/cast.
    """
    cap = open_video_source(args.source, args.input)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)

    tx_q: queue.Queue = queue.Queue(maxsize=2)
    rx_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    tx_stats: dict = {}
    rx_stats: dict = {}

    if streaming:
        print(f"[{mode_label}] TX 模式: streaming (non-cyclic, fixed buffer "
              f"{fixed_bb_size} bb samples). TX 内容更新率随主线程产帧速度.")
        tx_thread = threading.Thread(
            target=_tx_stream_worker,
            args=(push_fn, tx_q, stop_event, tx_stats),
            daemon=True)
    else:
        tx_min_period = 1.0 / args.tx_fps if args.tx_fps > 0 else 0.0
        print(f"[{mode_label}] TX 节流: {args.tx_fps:.1f} fps "
              f"(min_period={tx_min_period*1000:.0f}ms)")
        tx_thread = threading.Thread(
            target=_tx_push_worker,
            args=(push_fn, tx_q, stop_event, tx_stats, tx_min_period),
            daemon=True)
    rx_thread = threading.Thread(
        target=_rx_decode_worker, args=(capture_fn, rx_q, stop_event, rx_stats),
        daemon=True)
    tx_thread.start()
    rx_thread.start()

    producer = FrameProducer(args, cap, fixed_bb_size=fixed_bb_size,
                             pre_upsample=pre_upsample)
    last_rx = None
    tx_submit = 0
    rx_done = 0
    t0 = time.time()
    last_report = t0
    last_pushed = 0

    try:
        while time.time() - t0 < args.duration:
            built = producer.next()
            if built is None:
                break
            frame, burst, fid, n_pkt, jpg_len = built

            if _replace_latest(tx_q, burst):
                tx_submit += 1

            while True:
                try:
                    last_rx = rx_q.get_nowait()
                    rx_done += 1
                except queue.Empty:
                    break

            cv2.imshow("TX", cv2.resize(frame, (480, 360)))
            if last_rx is not None:
                cv2.imshow("RX", cv2.resize(last_rx, (480, 360), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) == 27:
                break

            now = time.time()
            if now - last_report > 0.5:
                elapsed = now - t0
                pushed = tx_stats.get("pushed", 0)
                tx_air_fps = (pushed - last_pushed) / max(now - last_report,
                                                          1e-3)
                last_pushed = pushed
                last_report = now
                rx_fps = rx_done / max(elapsed, 1e-3)
                rms = rx_stats.get("last_rms", 0.0)
                peak = rx_stats.get("last_peak", 0.0)
                dets = rx_stats.get("last_dets", 0)
                print(f"\r[{mode_label}] fid#{fid} pkts={n_pkt} "
                      f"JPEG={jpg_len}B  TX_air={tx_air_fps:.1f}fps  "
                      f"RX={rx_fps:.1f}fps ({rx_done}/{tx_submit})  "
                      f"caps={rx_stats.get('captures',0)}  "
                      f"rms={rms:.3f} peak={peak:.2f} det={dets}  "
                      f"{elapsed:.1f}s",
                      end="")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        tx_thread.join(timeout=2)
        rx_thread.join(timeout=2)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for fn in close_fns:
            try:
                fn()
            except Exception:
                pass

    elapsed = time.time() - t0
    pushed = tx_stats.get("pushed", 0)
    streamed = tx_stats.get("stream_calls", 0)
    extra = (f", 硬件 emit {streamed} 次 ({streamed/elapsed:.1f}/s)"
             if streaming else "")
    print(f"\n[结果] {mode_label}: 主线程提交 {tx_submit} 帧, 实际空口发出 "
          f"{pushed} 帧 ({pushed/elapsed:.1f}fps), RX 解出 {rx_done} 帧 "
          f"({rx_done/elapsed:.1f}fps), 捕获 {rx_stats.get('captures',0)} 次"
          f"{extra}")


def _serial_sdr_worker(push_fn, capture_fn, tx_q, rx_q,
                       stop_event, tx_stats, rx_stats,
                       min_period: float = 0.0):
    """单 Pluto 必须 serialize: 一个 worker 串行 push + capture + decode.

    实测两个 worker 并发会让 RX 饿死 (libiio context 锁 + USB 仅一路).
    所以这里只在 main 给了新 burst 时才 push, 平时纯 capture.
    min_period 用来限制 push 频率, 同 dual 的理由 (避免 destroy 间隙吃光 cyclic 稳态).
    """
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    last_push = 0.0

    while not stop_event.is_set():
        # 1. 非阻塞拿最新 burst (drop intermediate)
        burst = None
        while True:
            try:
                burst = tx_q.get_nowait()
            except queue.Empty:
                break
        if burst is not None:
            if min_period > 0:
                wait = min_period - (time.time() - last_push)
                if wait > 0:
                    time.sleep(wait)
            try:
                push_fn(burst)
                last_push = time.time()
                tx_stats["pushed"] = tx_stats.get("pushed", 0) + 1
            except Exception as e:
                print(f"\n[ERR] push: {e}")

        # 2. 捕一帧 (cyclic TX 在持续重复最近一次的 burst)
        try:
            samples = capture_fn()
        except Exception:
            continue
        rx_stats["captures"] = rx_stats.get("captures", 0) + 1

        # 3. 解
        indices = detect_packets(samples, threshold=0.55)
        frame_done = False
        for st in indices:
            if frame_done:
                break
            res = decode_packet(samples, st)
            if res is None:
                continue
            fid, pkt_id, tot, payload, ok = res
            if not ok or tot == 0 or pkt_id >= tot:
                continue
            if fid != last_frame_id:
                payloads_buf = {}
                last_total = tot
                last_frame_id = fid
            payloads_buf[pkt_id] = payload
            if last_total > 0 and all(i in payloads_buf
                                       for i in range(last_total)):
                full = b"".join(payloads_buf[i]
                                for i in range(last_total))
                rx_frame = jpeg_bytes_to_frame(full)
                if rx_frame is not None:
                    _replace_latest(rx_q, rx_frame)
                payloads_buf = {}
                last_frame_id = -1
                frame_done = True


def run_async(args):
    """单 Pluto 异步: 主线程做 camera+jpeg+display, 单 worker 做 SDR.

    单 Pluto 下 TX/RX 必须串行 (共享 libiio context). 异步只能解耦
    UI 和 SDR, 实测 ~3-5 fps. 想要 15 fps 用 --mode dual (两台 Pluto).
    """
    if not HAS_PLUTO:
        print("[错误] 未安装 pyadi-iio")
        return
    trx = PlutoTRX(args.uri, args.freq * 1e6, args.gain, args.rx_gain,
                   rx_buffer=2 ** args.rx_bits)

    cap = open_video_source(args.source, args.input)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)

    tx_q: queue.Queue = queue.Queue(maxsize=2)
    rx_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    tx_stats: dict = {}
    rx_stats: dict = {}

    tx_min_period = 1.0 / args.tx_fps if args.tx_fps > 0 else 0.0
    worker = threading.Thread(
        target=_serial_sdr_worker,
        args=(trx.push_tx, trx.capture, tx_q, rx_q, stop_event,
              tx_stats, rx_stats, tx_min_period),
        daemon=True)
    worker.start()

    producer = FrameProducer(args, cap)
    last_rx = None
    tx_submit = 0
    rx_done = 0
    t0 = time.time()
    last_report = t0
    last_pushed = 0

    try:
        while time.time() - t0 < args.duration:
            built = producer.next()
            if built is None:
                break
            frame, burst, fid, n_pkt, jpg_len = built
            if _replace_latest(tx_q, burst):
                tx_submit += 1

            while True:
                try:
                    last_rx = rx_q.get_nowait()
                    rx_done += 1
                except queue.Empty:
                    break

            cv2.imshow("TX", cv2.resize(frame, (480, 360)))
            if last_rx is not None:
                cv2.imshow("RX", cv2.resize(last_rx, (480, 360), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) == 27:
                break

            now = time.time()
            if now - last_report > 0.5:
                elapsed = now - t0
                pushed = tx_stats.get("pushed", 0)
                tx_air_fps = (pushed - last_pushed) / max(now - last_report,
                                                          1e-3)
                last_pushed = pushed
                last_report = now
                rx_fps = rx_done / max(elapsed, 1e-3)
                print(f"\r[ASYNC] fid#{fid} pkts={n_pkt} JPEG={jpg_len}B  "
                      f"TX_air={tx_air_fps:.1f}fps  RX={rx_fps:.1f}fps "
                      f"({rx_done}/{tx_submit})  "
                      f"caps={rx_stats.get('captures',0)}  {elapsed:.1f}s",
                      end="")
            # 主线程不要刷得太快, 让 worker 实际跑起来
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker.join(timeout=2)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        trx.close()

    elapsed = time.time() - t0
    pushed = tx_stats.get("pushed", 0)
    print(f"\n[结果] ASYNC: 主线程提交 {tx_submit} 帧, 空口发出 "
          f"{pushed} 帧 ({pushed/elapsed:.1f}fps), RX 解出 {rx_done} 帧 "
          f"({rx_done/elapsed:.1f}fps), 捕获 {rx_stats.get('captures',0)} 次")


def run_dual(args):
    """双 Pluto: TX 在 args.tx_uri, RX 在 args.rx_uri, 两个独立 context."""
    if not HAS_PLUTO:
        print("[错误] 未安装 pyadi-iio")
        return
    if args.tx_uri == args.rx_uri:
        print(f"[错误] dual 模式 --tx-uri 和 --rx-uri 不能相同 ({args.tx_uri})")
        print("       同一台 Pluto 用 --mode async 即可")
        return
    tx = PlutoTX(args.tx_uri, args.freq * 1e6, args.gain, streaming=False)
    rx = PlutoRX(args.rx_uri, args.freq * 1e6, args.rx_gain,
                 rx_buffer=2 ** args.rx_bits)
    # cyclic 模式但 producer 预 upsample, worker 只调 sdr.tx() (没有 numpy
    # CPU 时间在 push 关键路径上). 关键: capture 窗口 < TX swap 周期 ->
    # 每次 capture 落在单一 fid 的 cyclic 稳态, frame 不会被 fid 跳变打碎.
    _run_threaded(args, tx.push_iq, rx.capture,
                  [tx.close, rx.close], "DUAL",
                  streaming=False, fixed_bb_size=0, pre_upsample=True)


def run_txbeacon(args):
    """TX 诊断: 在 args.tx_uri 上 cyclic 发一个固定测试 burst, 不抢 RX context.

    跟另一台机器/终端的 rxprobe 配合, 验证空口链路是否打通.
    burst 内容是 frame_id=0 的彩色测试图样, 5 包, ~6KB JPEG.
    """
    if not HAS_PLUTO:
        print("[错误] 未安装 pyadi-iio")
        return
    uri = args.tx_uri if args.mode == "txbeacon" else args.uri
    tx = PlutoTX(uri, args.freq * 1e6, args.gain)

    # 用 192x144 测试图样生成一个 burst (固定内容, cyclic 重复发)
    h, w = 144, 192
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :, 0] = (y * 3) % 256
        img[y, :, 1] = (y * 5 + 80) % 256
        img[y, :, 2] = (255 - y * 3) % 256
    cv2.putText(img, "TX BEACON", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    jpg = frame_to_jpeg_bytes(img, w, h, 60)
    chunks = split_payload(jpg)
    silence = np.zeros(NSYM * 4, dtype=np.complex64)
    parts = []
    for pid, chunk in enumerate(chunks):
        wf, _ = build_packet_waveform(chunk, 0, pid, len(chunks))
        parts.append(wf)
        parts.append(silence)
    burst = np.concatenate(parts).astype(np.complex64)
    print(f"[BEACON] 固定 burst: {len(chunks)} 包, JPEG={len(jpg)}B, "
          f"{len(burst)} baseband samples")

    try:
        tx.push(burst)
        print(f"[BEACON] 持续发射中 (cyclic, freq={args.freq:.1f}MHz, "
              f"TX_GAIN={args.gain}dB). Ctrl+C 停止.")
        t0 = time.time()
        while time.time() - t0 < args.duration:
            time.sleep(0.5)
            print(f"\r[BEACON] elapsed {time.time()-t0:.1f}s", end="")
    except KeyboardInterrupt:
        pass
    finally:
        tx.close()
    print("\n[BEACON] 已停止")


def run_rxprobe(args):
    """RX 诊断: 不仅测能量, 还把每一捕的 detection/LTS/Header/CRC 各阶段成功率拆开打."""
    if not HAS_PLUTO:
        print("[错误] 未安装 pyadi-iio")
        return
    uri = args.rx_uri if args.mode == "rxprobe" else args.uri
    rx = PlutoRX(uri, args.freq * 1e6, args.rx_gain, rx_buffer=2 ** 18)
    print(f"[PROBE] 监听 {uri} @ {args.freq:.1f} MHz, RX={args.rx_gain}dB.")
    print("        阶段意义: det = L-STF autocorr 触发数 (含噪声假阳性);")
    print("                  LTS = 找到合法 64-sample LTS 模板的检测点数;")
    print("                  HDR = header magic 命中 (说明 OFDM 解出来字节对了);")
    print("                  CRC = 整包通过 CRC32 (说明真的成功);")
    print("        det 高但 LTS=0 → 噪声假阳性, 信号没真正进来;")
    print("        LTS>0 但 HDR=0 → 信号在但 OFDM 解调坏 (CFO/采样率漂移);")
    print("        HDR>0 但 CRC=0 → 包大体能解, BER 偏高.")
    t0 = time.time()
    n = 0
    cum = {"det": 0, "lts": 0, "hdr": 0, "crc": 0}
    try:
        while time.time() - t0 < args.duration:
            samples = rx.capture()
            n += 1
            rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
            peak = float(np.max(np.abs(samples)))
            # 单次 metric 峰值: 反映 autocorr 信号-噪声比, 真信号一般 >0.85
            L = 16
            M = 64
            if len(samples) > L + M:
                a = samples[:-L]
                b = np.conj(samples[L:])
                prod = a * b
                energy = np.abs(samples[L:]) ** 2
                cs = np.concatenate([[0], np.cumsum(prod)])
                ce = np.concatenate([[0], np.cumsum(energy)])
                csum = cs[M:] - cs[:-M]
                cE = ce[M:] - ce[:-M]
                # 保护小 cE 下的数值噪声 (1e-9 太小会把 metric 推到几千)
                metric_peak = float(np.max(np.abs(csum) /
                                           np.maximum(cE, 1e-3)))
                metric_peak = min(metric_peak, 2.0)
            else:
                metric_peak = 0.0

            indices = detect_packets(samples, threshold=0.55)
            this_det = len(indices)
            this_lts = this_hdr = this_crc = 0
            for st in indices:
                if st + LEN_PREAMBLE > len(samples):
                    continue
                pos = find_lts_position(samples, st + 192,
                                        search_radius=120)
                if pos is None:
                    continue
                this_lts += 1
                res = decode_packet(samples, st)
                if res is None:
                    continue
                # decode_packet 内部已校 magic; res 不为 None 即 HDR ok
                this_hdr += 1
                if res[4]:
                    this_crc += 1
            cum["det"] += this_det
            cum["lts"] += this_lts
            cum["hdr"] += this_hdr
            cum["crc"] += this_crc
            print(f"\r[PROBE #{n:3d}] rms={rms:.4f} peak={peak:.3f} "
                  f"metric={metric_peak:.2f}  "
                  f"det={this_det:3d} LTS={this_lts:3d} HDR={this_hdr:2d} "
                  f"CRC={this_crc:2d}  cum CRC={cum['crc']}  "
                  f"t={time.time()-t0:.1f}s", end="")
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
    print(f"\n[结果] {n} 次 capture, 累计 det={cum['det']} LTS={cum['lts']} "
          f"HDR={cum['hdr']} CRC={cum['crc']}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="ADALM-Pluto OFDM 彩色视频传输 (5.8 GHz, 802.11a 风格)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pluto_ofdm_video.py --mode sim --snr 20                          # 仿真
  python pluto_ofdm_video.py --mode loopback                              # 单台同步 (调试用)
  python pluto_ofdm_video.py --mode async                                 # 单台异步线程
  python pluto_ofdm_video.py --mode dual --tx-uri ip:192.168.2.1 \
                              --rx-uri ip:192.168.3.1                     # 两台 Pluto, 15+ fps 目标
  python pluto_ofdm_video.py --mode dual --jpeg 80 --fwidth 320 --fheight 240
""")
    p.add_argument("--mode", default="sim",
                   choices=["sim", "loopback", "async", "dual",
                            "rxprobe", "txbeacon"])
    p.add_argument("--source", default="file",
                   choices=["file", "camera", "pattern"])
    p.add_argument("--input", default="", help="视频文件路径")
    p.add_argument("--uri", default="ip:192.168.2.1",
                   help="单台 Pluto URI (sim/loopback/async)")
    p.add_argument("--tx-uri", default="ip:192.168.2.1",
                   help="dual 模式下 TX Pluto URI")
    p.add_argument("--rx-uri", default="ip:192.168.3.1",
                   help="dual 模式下 RX Pluto URI")
    p.add_argument("--freq", type=float, default=CARRIER_FREQ_DEFAULT / 1e6,
                   help="中心频率 MHz")
    p.add_argument("--gain", type=float, default=TX_GAIN_DEFAULT,
                   help="TX 增益 dB")
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DEFAULT,
                   help="RX 增益 dB")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--fwidth", type=int, default=320, help="视频帧宽度 (彩色)")
    p.add_argument("--fheight", type=int, default=240, help="视频帧高度 (彩色)")
    p.add_argument("--jpeg", type=int, default=JPEG_QUALITY,
                   help="编码质量 1-100 (JPEG/WebP 共用此参数)")
    p.add_argument("--codec", default="jpg", choices=["jpg", "webp"],
                   help="帧编码: jpg (默认) 或 webp (同字节下画质更好 ~30-50%)")
    p.add_argument("--snr", type=float, default=20, help="仿真 SNR (dB)")
    p.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam"],
                   help="子载波调制: qpsk (默认, 鲁棒) 或 16qam (吞吐 x2, BER 高)")
    p.add_argument("--tx-fps", type=float, default=12.0,
                   help="async/dual 模式 TX 节流上限 (fps). 默认 12; "
                        "TX 推太快会被 buffer destroy 间隙吃掉 cyclic 稳态时间")
    p.add_argument("--rx-bits", type=int, default=18,
                   help="rx_buffer = 2**rx-bits RF 样本数. 默认 18 (8.7ms); "
                        "dual 推荐 19 (17.5ms, 跨 cyclic 周期更稳)")

    args = p.parse_args()
    set_modulation(args.mod)
    print(f"[CFG] modulation={_MOD_NAME}  bits/sym={_MOD_BITS_PER_SYM}")

    if args.mode == "sim":
        run_simulation(args)
    elif args.mode == "loopback":
        run_loopback(args)
    elif args.mode == "async":
        run_async(args)
    elif args.mode == "dual":
        run_dual(args)
    elif args.mode == "rxprobe":
        run_rxprobe(args)
    elif args.mode == "txbeacon":
        run_txbeacon(args)


if __name__ == "__main__":
    main()
