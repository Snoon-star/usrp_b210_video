#!/usr/bin/env python3
"""
USRP x310 OFDM 彩色视频/图像传输 (5.8 GHz)
===============================================
基于 pluto_ofdm_video.py 移植到 Ettus USRP x310, 协议与上层 100% 一致:
  - L-STF / L-LTF 前导码 (粗/精时序, 粗/精 CFO, LS 信道估计)
  - 64-FFT OFDM, 48 数据子载波 + 4 导频, CP = 16
  - QPSK / 16QAM 调制
  - 每包帧头携带序号 + 长度 + CRC32
  - 30 MHz 空口采样 (20 * 1.5 过采样)
  - 视频帧 JPEG / WebP / H.264 压缩后分包传输, RX 端解码并显示

x310 相对 Pluto 的硬件差异 (代码自动利用):
  - 10 GbE 链路, 真实持续吞吐 ~125 MB/s (Pluto 仅 ~100 MB/s 且需 cyclic)
  - 30 MS/s × 8B 复样 = 240 MB/s 单向数据, 跑在 10 GbE 上无压力
  - TX 不再依赖 Pluto 的 cyclic buffer, UsrpTX 内部用 daemon 线程
    持续 send() 实现等效 cyclic, 无 destroy/create 间隙
  - x310 单台即可双通道 (TX0/RX0/TX1/RX1), 同一设备做 dual 也可,
    但当前实现保留双 USRP 模式 (不同 IP) 以匹配 Pluto 端的双板拓扑

依赖: pip install uhd numpy scipy opencv-python av
       (uhd 通常随 UHD 安装包附带, 注意 Python 版本要匹配)

典型 x310 网卡配置:
  10 GbE NIC 静态 IP 192.168.10.1/24, 连 x310 SFP+0 接口 (默认 192.168.10.2)
  第二条 10 GbE NIC 静态 IP 192.168.20.1/24, 连第二台 x310 (192.168.20.2)
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

# pyFFTW: 批量 FFT 比 numpy 快数倍 (用于 ofdm_demodulate). 没装则回退 np.fft.
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
    HAS_PYFFTW = True
except Exception:
    HAS_PYFFTW = False

# Numba JIT 加速解码热点 (检测/互相关). 没装 numba 时回退为纯 Python (功能不变,
# 仅慢). 安装: conda install -c conda-forge numba
try:
    from numba import njit as _njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorate(fn):
            return fn
        return _decorate

try:
    import fec
    HAS_FEC = True
except Exception:
    HAS_FEC = False

try:
    import uhd
    HAS_USRP = True
except ImportError:
    HAS_USRP = False
    print("[警告] 未安装 uhd Python 绑定, 仅支持仿真模式. "
          "安装方法: 从 https://files.ettus.com 下载 UHD 并启用 Python 绑定, "
          "或 conda install -c conda-forge uhd")


# ============================================================================
# 全局射频/OFDM 配置
# ============================================================================y

CARRIER_FREQ_DEFAULT = 5.8e9           # 5.8 GHz (B210 量程 70MHz~6GHz)
SAMPLE_RATE_BB = 5e6                   # 基带采样率 (B210 over USB3 降速防溢出)
OSF = 1.5                              # 过采样因子 (运行时由 --osf 覆盖)
RF_RATE = int(SAMPLE_RATE_BB * OSF)    # B210 空口 (= OSF * 基带率)
RF_BANDWIDTH = 6e6
# BB<->RF 有理重采样比 (= OSF). UP==DOWN 时 (OSF=1) 重采样直通, 省掉 scipy
# resample_poly 这个 RX 端最大头开销 (实测占接收计算 ~70%, 且在采集线程里会
# 撕出接收盲区). OFDM 自带 12/64 空子载波做保护带, 不过采样也不混叠 (即 802.11a
# 的做法). 由 main() 据 --osf 设置.
_RESAMPLE_UP = 3
_RESAMPLE_DOWN = 2
# UBX-160 / SBX-120 / CBX-120 子板典型增益范围 0-31.5 dB.
# 注意 x310 的"增益"是放大量 (Pluto 的负数是衰减), 这里默认中等值.
TX_GAIN_DEFAULT = 10
RX_GAIN_DEFAULT = 20

# x310 默认 SFP+0 接口 IP, 主机端 NIC 应配 192.168.10.1/24
USRP_DEFAULT_ADDR = "addr=192.168.10.2"
USRP_SECOND_ADDR = "addr=192.168.20.2"

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


# 64QAM Gray, 每符号 6 bit (I/Q 各 8 电平 ±1,±3,±5,±7), 平均功率归一为 1.
# 比特约定沿用 QPSK/16QAM: bit=0 -> 正向. 用 LUT 构造保证 mod/demod 严格互逆.
def _build_qam64_lut() -> np.ndarray:
    # 3-bit Gray -> 8-PAM 电平, 与下面 demod 切片规则一致:
    #   b_sign=(v<0), b_mid=(|v|<4), b_low=(|v|<2 或 |v|>6)
    levels = {}
    for lv in (7, 5, 3, 1, -1, -3, -5, -7):
        a = 1 if lv < 0 else 0
        b = 1 if abs(lv) < 4 else 0
        c = 1 if (abs(lv) < 2 or abs(lv) > 6) else 0
        levels[(a, b, c)] = lv
    lut = np.zeros(64, dtype=np.complex64)
    for idx in range(64):
        bi = ((idx >> 5) & 1, (idx >> 4) & 1, (idx >> 3) & 1)
        bq = ((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)
        lut[idx] = levels[bi] + 1j * levels[bq]
    return (lut / np.sqrt(42.0)).astype(np.complex64)


_QAM64_LUT = _build_qam64_lut()


def qam64_modulate(bits: np.ndarray) -> np.ndarray:
    pad = (-len(bits)) % 6
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    g = bits.reshape(-1, 6)
    idx = ((g[:, 0] << 5) | (g[:, 1] << 4) | (g[:, 2] << 3)
           | (g[:, 3] << 2) | (g[:, 4] << 1) | g[:, 5])
    return _QAM64_LUT[idx]


def qam64_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 6, dtype=np.int8)
    I = np.real(symbols) * np.sqrt(42)
    Q = np.imag(symbols) * np.sqrt(42)
    aI = np.abs(I)
    aQ = np.abs(Q)
    bits[0::6] = (I < 0).astype(np.int8)
    bits[1::6] = (aI < 4).astype(np.int8)
    bits[2::6] = ((aI < 2) | (aI > 6)).astype(np.int8)
    bits[3::6] = (Q < 0).astype(np.int8)
    bits[4::6] = (aQ < 4).astype(np.int8)
    bits[5::6] = ((aQ < 2) | (aQ > 6)).astype(np.int8)
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
    elif name in ("64qam", "qam64"):
        _MOD_BITS_PER_SYM = 6
        _MOD_FN = qam64_modulate
        _DEMOD_FN = qam64_demodulate
        name = "64qam"
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
    _ifft = _fftw.ifft if HAS_PYFFTW else np.fft.ifft
    t = _ifft(grid, axis=1) * (NFFT / np.sqrt(N_DATA + N_PILOT))
    sym = np.concatenate([t[:, -NCP:], t], axis=1)
    return sym.astype(np.complex64).reshape(-1)


def ofdm_demodulate(time_signal: np.ndarray, h_est: np.ndarray) -> np.ndarray:
    """OFDM 解调 (频域单抽头均衡 + 导频残留 CFO 校正), 矢量化实现."""
    n_sym = len(time_signal) // NSYM
    if n_sym == 0:
        return np.zeros(0, dtype=np.complex64)
    time_signal = time_signal[: n_sym * NSYM].reshape(n_sym, NSYM)
    syms = time_signal[:, NCP:]
    _fft = _fftw.fft if HAS_PYFFTW else np.fft.fft
    Y = _fft(syms, axis=1) / np.sqrt(N_DATA + N_PILOT)
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

@_njit(cache=True)
def _detect_core_njit(rx, L, M, threshold):
    """单趟滑窗自相关 + plateau 扫描 (njit). 返回 plateau 起始索引数组.
    等价于原 cumsum 实现, 但单趟、零大数组分配, 显著更快."""
    n = rx.shape[0]
    nmetric = n - L - M + 1
    if nmetric <= 0:
        return np.empty(0, dtype=np.int64)
    above = np.empty(nmetric, dtype=np.uint8)
    S = 0j                                  # 滑窗内 prod 的复数和 (complex128)
    E = 0.0                                 # 滑窗内能量和
    for j in range(M):
        S += rx[j] * np.conj(rx[j + L])
        v = rx[j + L]
        E += v.real * v.real + v.imag * v.imag
    for k in range(nmetric):
        above[k] = 1 if (abs(S) / (E + 1e-9)) > threshold else 0
        if k + 1 < nmetric:                 # 滑动: 去掉 prod[k], 加入 prod[k+M]
            S = S - rx[k] * np.conj(rx[k + L]) \
                  + rx[k + M] * np.conj(rx[k + M + L])
            vr = rx[k + L]
            va = rx[k + M + L]
            E = E - (vr.real * vr.real + vr.imag * vr.imag) \
                  + (va.real * va.real + va.imag * va.imag)
    starts = np.empty(nmetric, dtype=np.int64)
    cnt = 0
    i = 0
    while i < nmetric:
        if above[i] == 1:
            j = i
            while j < nmetric and above[j] == 1:
                j += 1
            starts[cnt] = i                 # plateau 左边沿 (≈ L-STF 起点)
            cnt += 1
            i = j + 32
        else:
            i += 1
    return starts[:cnt]


def detect_packets(rx: np.ndarray, threshold: float = 0.5):
    """用 L-STF 16-sample 自相关做粗包检测, 返回 plateau 起始索引数组."""
    L = 16
    M = 64
    if len(rx) < L * 12:
        return np.empty(0, dtype=np.int64)
    if HAS_NUMBA:
        return _detect_core_njit(np.ascontiguousarray(rx), L, M, float(threshold))
    # 纯 numpy 回退 (无 numba 时; 比 njit 慢但远快于纯 Python 循环)
    prod = rx[:-L] * np.conj(rx[L:])
    energy = np.abs(rx[L:]) ** 2
    cumsum = np.concatenate(([0], np.cumsum(prod)))
    cumE = np.concatenate(([0], np.cumsum(energy)))
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
            indices.append(i)
            i = j + 32
        else:
            i += 1
    return np.asarray(indices, dtype=np.int64)


def apply_cfo(x: np.ndarray, freq_per_sample: float) -> np.ndarray:
    n = np.arange(len(x))
    return (x * np.exp(-1j * freq_per_sample * n)).astype(np.complex64)


@_njit(cache=True)
def _xcorr_abs_njit(seg, template):
    """|valid 互相关| (njit), 等价于 np.abs(np.correlate(seg, template, 'valid'))."""
    ns = seg.shape[0]
    nt = template.shape[0]
    nout = ns - nt + 1
    if nout <= 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(nout, dtype=np.float64)
    for k in range(nout):
        acc = 0j
        for m in range(nt):
            acc += seg[k + m] * np.conj(template[m])
        out[k] = abs(acc)
    return out


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
    if HAS_NUMBA:
        abs_corr = _xcorr_abs_njit(np.ascontiguousarray(seg),
                                   np.ascontiguousarray(template))
    else:
        abs_corr = np.abs(np.correlate(seg, template, mode="valid"))
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


# 信道估计用到的子载波 FFT 索引 (预计算, 替代每次调用对 52 个子载波的 Python 循环)
_CH_IDX = (np.concatenate([np.arange(-26, 0), np.arange(1, 27)]) % NFFT).astype(np.intp)
_CH_IDX = _CH_IDX[LLTF_FREQ_REF[_CH_IDX] != 0]


def estimate_channel_from_lts(rx_lts1: np.ndarray,
                              rx_lts2: np.ndarray) -> np.ndarray:
    """LS 信道估计 (向量化, 无 Python 循环)."""
    avg = 0.5 * (rx_lts1 + rx_lts2)
    H = np.fft.fft(avg) / np.sqrt(N_DATA + N_PILOT)
    h_est = np.ones(NFFT, dtype=np.complex64)
    h_est[_CH_IDX] = (H[_CH_IDX] / LLTF_FREQ_REF[_CH_IDX]).astype(np.complex64)
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
    """BB -> RF 有理上采样 (OSF = UP/DOWN). OSF=1 时直通, 省掉 resample 开销."""
    if _RESAMPLE_UP == _RESAMPLE_DOWN:
        return np.ascontiguousarray(x, dtype=np.complex64)
    return sps.resample_poly(x, up=_RESAMPLE_UP,
                             down=_RESAMPLE_DOWN).astype(np.complex64)


def downsample_to_bb(x: np.ndarray) -> np.ndarray:
    """RF -> BB 有理下采样 (1/OSF). OSF=1 时直通 (省掉 RX 端最大头开销)."""
    if _RESAMPLE_UP == _RESAMPLE_DOWN:
        return np.ascontiguousarray(x, dtype=np.complex64)
    return sps.resample_poly(x, up=_RESAMPLE_DOWN,
                             down=_RESAMPLE_UP).astype(np.complex64)


# ============================================================================
# 视频帧 <-> 字节流 (JPEG 压缩)
# ============================================================================

JPEG_QUALITY = 75
MAX_PAYLOAD_PER_PKT = 1200


# ============================================================================
# H.264 编/解码 (PyAV / libx264). 同字节数下比 JPEG/WebP 更清, GOP=1 时
# 每帧独立 (~25% 增益, 鲁棒), GOP>1 时启用帧间预测 (压缩翻倍但丢帧会花屏
# 直到下一个 IDR; 解码侧检测 fid 跳变会自动 reset).
# ============================================================================

_FFMPEG_PRELOADED = False
_AV_CODEC = "h264"   # 实际 PyAV 编解码器名; main() 据 --codec 设为 "h264" 或 "hevc"(H.265)


def _preload_ffmpeg_dlls():
    """修复 Windows + Python<3.8 下 PyAV 'DLL load failed' (WinError 127):
    avformat 等的依赖会被系统其他同名 DLL 抢先加载导致符号不匹配. 启动时把
    conda 环境 <prefix>/Library/bin 里的 DLL 用全路径预加载进内存, 之后按名
    解析即复用正确版本. 仅 Windows 需要, 失败静默忽略, 只执行一次."""
    global _FFMPEG_PRELOADED
    if _FFMPEG_PRELOADED or os.name != "nt":
        return
    _FFMPEG_PRELOADED = True
    import ctypes
    import glob
    libbin = os.path.join(sys.prefix, "Library", "bin")
    if not os.path.isdir(libbin):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(libbin)
        except OSError:
            pass
    for _f in glob.glob(os.path.join(libbin, "*.dll")):
        try:
            ctypes.WinDLL(_f)
        except OSError:
            pass


class _H264Encoder:
    def __init__(self, width: int, height: int, bitrate: int, gop: int,
                 preset: str = "fast", crf: int = 0):
        try:
            _preload_ffmpeg_dlls()
            import av
            from fractions import Fraction
        except ImportError:
            raise RuntimeError(
                "H.264 需要 PyAV. 安装: pip install av")
        self._av = av
        self.width = (width // 2) * 2          # libx264 要求偶数
        self.height = (height // 2) * 2
        self.codec = av.CodecContext.create(_AV_CODEC, "w")
        self.codec.width = self.width
        self.codec.height = self.height
        self.codec.pix_fmt = "yuv420p"
        self.codec.time_base = Fraction(1, 30)
        self.codec.gop_size = int(gop)
        opts = {
            "tune": "zerolatency",
            "preset": preset,   # ultrafast / superfast / veryfast / faster /
                                # fast / medium / slow ; 越慢压缩越高
        }
        opts["bf"] = "0"        # 0 B-frames: 低延迟, 丢包不依赖未来帧 (h264/h265 通用)
        if _AV_CODEC == "hevc":
            # 零前瞻/零B帧低延迟; repeat-headers=1 让每个关键帧自带 SPS/PPS,
            # 这样丢帧/解码器 reset 后仍能独立解码 -> 消除 cu_qp_delta 报错/花屏.
            # frame-threads=1: tune=zerolatency 在本环境未真正关掉帧级线程(实测
            # 仍 4 线程), 会缓存 ~4 帧才吐出 -> 固定延迟+'发N帧只到N-4'掉帧假象.
            opts["x265-params"] = ("bframes=0:rc-lookahead=0:scenecut=0:"
                                   "repeat-headers=1:frame-threads=1")
        if crf and int(crf) > 0:
            # 恒定质量模式: 每帧画质固定, 与码率/fps 假设无关 (解决码率被
            # 按 30fps 平摊导致每帧过小/模糊). crf 越小越清晰, 18~28 常用.
            opts["crf"] = str(int(crf))
        else:
            self.codec.bit_rate = int(bitrate)
        self.codec.options = opts
        self._pts = 0

    def encode(self, bgr: np.ndarray) -> bytes:
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
        # cv2 输出 I420 layout, shape ((H*3)//2, W), 这是 PyAV 的 yuv420p 格式
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame = self._av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        frame.pts = self._pts
        self._pts += 1
        out = bytearray()
        for pkt in self.codec.encode(frame):
            out.extend(bytes(pkt))
        return bytes(out)


class _H264Decoder:
    def __init__(self):
        try:
            _preload_ffmpeg_dlls()
            import av
        except ImportError:
            raise RuntimeError("H.264 需要 PyAV. 安装: pip install av")
        self._av = av
        self.codec = av.CodecContext.create(_AV_CODEC, "r")
        self._last = None

    def reset(self):
        """fid 跳变后调用, 丢弃 P-frame 引用状态, 等待下一个 IDR."""
        self.codec = self._av.CodecContext.create(_AV_CODEC, "r")

    def decode(self, data: bytes):
        if not data:
            return self._last
        try:
            packets = self.codec.parse(data)
            for pkt in packets:
                frames = self.codec.decode(pkt)
                for f in frames:
                    self._last = f.to_ndarray(format="bgr24")
        except Exception:
            pass
        return self._last


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
# USRP x310 硬件接口
# ============================================================================
#
# 设计要点 (与 Pluto 版的差异):
#
# 1. cyclic 仿真: UHD 没有 Pluto 那种 tx_cyclic_buffer 硬件循环, 因此
#    UsrpTX/UsrpTRX 内部用 daemon 线程持续 send() 同一 burst, 主线程通过
#    锁原子替换 burst 内容. send() 是 push-based 的, 10 GbE 链路带宽足够,
#    不存在 Pluto cyclic destroy/create 间隙问题, RX 端不会再撞到静默期.
#
# 2. capture(): rx_streamer.recv() 阻塞读取 rx_buffer 个 RF 样本, 然后
#    downsample 回 baseband. UHD 内部维护连续 RX 流 (start_cont 命令).
#
# 3. 默认增益: UBX/SBX/CBX 子板范围 0-31.5 dB. 信号弱时 RX 调 25-30,
#    饱和时调到 10-15. 不像 Pluto 用负数衰减.
#
# 4. 静音空闲通道: x310 的 TX 路径只在 send() 调用期间激活, 默认 silent;
#    所以 UsrpRX 不需要专门把 TX gain 拉到极低. 不过 UsrpTX 的 RX 不用
#    时也不开 stream_cmd, 自然就是关闭的.
# ============================================================================


def _uhd_tune(freq: float):
    """构造 tune_request, 避免在多处重复写."""
    return uhd.libpyuhd.types.tune_request(float(freq))


def _make_stream_args(chan: int = 0):
    """fc32 (主机端 complex64) / sc16 (空口 int16 复数), 单通道.
    chan: B210 RF 前端通道. 0=RF A (TX/RX/RX2 口), 1=RF B (TX/RX/RX2 口)."""
    st = uhd.usrp.StreamArgs("fc32", "sc16")
    st.channels = [chan]
    return st


def _start_rx_continuous(rx_streamer):
    """开 RX 连续流, 必须在每次 recv 前已经 issue 过 stream_cmd."""
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)


def _stop_rx_continuous(rx_streamer):
    try:
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(cmd)
    except Exception:
        pass


class UsrpTX:
    """TX-only USRP x310. 通过 daemon 线程持续 send() 实现 cyclic 效果."""

    def __init__(self, uri: str, freq: float, tx_gain: float,
                 streaming: bool = True, chan: int = 0):
        if not HAS_USRP:
            raise RuntimeError("uhd 未安装")
        self.chan = int(chan)
        print(f"[TX]  连接 USRP {uri} (chan={self.chan}) ...")
        self.usrp = uhd.usrp.MultiUSRP(uri)
        self.usrp.set_tx_rate(float(RF_RATE), self.chan)
        self.usrp.set_tx_freq(_uhd_tune(freq), self.chan)
        self.usrp.set_tx_gain(float(tx_gain), self.chan)
        try:
            self.usrp.set_tx_bandwidth(float(RF_BANDWIDTH), self.chan)
        except Exception:
            pass
        try:
            # B210: chan0=RF A 的 TX/RX 口, chan1=RF B 的 TX/RX 口. 天线名都是
            # "TX/RX" (能发能收), 选哪个物理 SMA 由 chan 决定.
            self.usrp.set_tx_antenna("TX/RX", self.chan)
        except Exception:
            pass
        self.tx_streamer = self.usrp.get_tx_stream(_make_stream_args(self.chan))
        self.streaming = streaming
        # cyclic 仿真用的状态
        self._burst = None
        self._lock = threading.Lock()
        self._loop_thread = None
        self._loop_stop = threading.Event()
        self._send_count = 0           # 累计成功 send() 次数 (= cyclic 发出的 burst 数)
        mode = "streaming(one-shot)" if streaming else "cyclic(thread-emul)"
        port = "RF B(TX/RX)" if self.chan == 1 else "RF A(TX/RX)"
        print(f"[TX]  就绪  Freq={freq/1e9:.2f}GHz  SR={RF_RATE/1e6:.1f}MHz  "
              f"TX={tx_gain}dB  chan={self.chan}({port})  {mode}")

    def _cyclic_loop(self):
        """daemon 线程: 反复 send() 当前 burst 模拟 cyclic 发射."""
        md = uhd.types.TXMetadata()
        md.has_time_spec = False
        md.start_of_burst = False
        md.end_of_burst = False
        while not self._loop_stop.is_set():
            with self._lock:
                burst = self._burst
            if burst is None:
                time.sleep(0.001)
                continue
            try:
                self.tx_streamer.send(burst, md, 0.5)
                self._send_count += 1
            except Exception:
                time.sleep(0.001)

    def push_iq(self, tx_iq: np.ndarray):
        """喂已经 upsample/normalize 好的 complex64 IQ. UHD fc32 已经是 -1..+1 范围."""
        if self.streaming:
            # one-shot: 发一次就停, 适合非 cyclic 流式探针
            md = uhd.types.TXMetadata()
            md.has_time_spec = False
            md.start_of_burst = True
            md.end_of_burst = True
            self.tx_streamer.send(tx_iq, md, 0.5)
        else:
            # cyclic 仿真: 更新 burst, 让后台线程持续发
            with self._lock:
                self._burst = tx_iq
            if self._loop_thread is None or not self._loop_thread.is_alive():
                self._loop_stop.clear()
                self._loop_thread = threading.Thread(
                    target=self._cyclic_loop, daemon=True)
                self._loop_thread.start()

    def push(self, baseband: np.ndarray):
        """兼容旧接口: 拿 baseband 自己 upsample. UHD 不需要 ×2^14 缩放."""
        rf = upsample_to_rf(baseband)
        peak = np.max(np.abs(rf))
        if peak > 0:
            rf = rf / peak * 0.7
        self.push_iq(rf.astype(np.complex64))

    def close(self):
        self._loop_stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1)


class UsrpRX:
    """RX-only USRP x310. capture() 阻塞读 rx_buffer 个 RF 样本后下采样."""

    def __init__(self, uri: str, freq: float, rx_gain: float,
                 rx_buffer: int = 2 ** 18, chan: int = 0):
        if not HAS_USRP:
            raise RuntimeError("uhd 未安装")
        self.chan = int(chan)
        print(f"[RX]  连接 USRP {uri} (chan={self.chan}) ...")
        self.usrp = uhd.usrp.MultiUSRP(uri)
        self.usrp.set_rx_rate(float(RF_RATE), self.chan)
        self.usrp.set_rx_freq(_uhd_tune(freq), self.chan)
        self.usrp.set_rx_gain(float(rx_gain), self.chan)
        try:
            self.usrp.set_rx_bandwidth(float(RF_BANDWIDTH), self.chan)
        except Exception:
            pass
        try:
            # B210: chan0=RF A 的 TX/RX 口, chan1=RF B 的 TX/RX 口 (RX 用 TX/RX
            # 天线名收发; 若想用纯收的 RX2 物理口可改成 "RX2").
            self.usrp.set_rx_antenna("TX/RX", self.chan)
        except Exception:
            pass
        self.rx_streamer = self.usrp.get_rx_stream(_make_stream_args(self.chan))
        self.rx_buffer = int(rx_buffer)
        self._md = uhd.types.RXMetadata()
        # 不再用连续流: capture() 每次按需有限取样, 避免处理间隙溢出
        # 暖机, 把启动瞬态丢掉
        for _ in range(3):
            try:
                self.capture()
            except Exception:
                pass
        port = "RF B(TX/RX)" if self.chan == 1 else "RF A(TX/RX)"
        print(f"[RX]  就绪  Freq={freq/1e9:.2f}GHz  RX={rx_gain}dB  "
              f"chan={self.chan}({port})  buf={rx_buffer}")

    def capture_rf(self) -> np.ndarray:
        """只 recv 原始 RF 样本, 不下采样. 把下采样留给消费线程: 采集线程
        专注 recv 即可把空口占空比拉满 (下采样若放在采集线程里, 每次 ~20ms
        都是接收盲区, 会丢掉大量空口数据 -> 帧难凑齐). OSF=1 时下采样本身直通."""
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = self.rx_buffer
        cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(cmd)
        rf = np.zeros(self.rx_buffer, dtype=np.complex64)
        filled = 0
        while filled < self.rx_buffer:
            n = self.rx_streamer.recv(rf[filled:], self._md, 1.0)
            if n <= 0:
                break
            filled += n
        if filled == 0:
            return np.zeros(0, dtype=np.complex64)
        return rf[:filled] if filled < self.rx_buffer else rf

    def capture(self) -> np.ndarray:
        """recv + 下采样 (兼容旧调用方: rxprobe / 暖机等)."""
        return downsample_to_bb(self.capture_rf())

    def close(self):
        _stop_rx_continuous(self.rx_streamer)


class UsrpTRX:
    """单台 x310 同时 TX + RX (channel 0 双向). 用于 loopback / async 模式.

    cyclic TX 用 daemon 线程仿真 (同 UsrpTX 思路); RX 用连续流模式.
    """

    def __init__(self, uri: str, freq: float, tx_gain: float, rx_gain: float,
                 rx_buffer: int = 2 ** 18):
        if not HAS_USRP:
            raise RuntimeError("uhd 未安装")
        print(f"[TRX] 连接 USRP {uri} ...")
        self.usrp = uhd.usrp.MultiUSRP(uri)
        # 共用同一 reference clock; x310 TX/RX 是分离子板各自 LO,
        # 配同一 freq 即可 loopback (LO 不完美同步, CFO 会有, 但前导能纠).
        self.usrp.set_tx_rate(float(RF_RATE), 0)
        self.usrp.set_rx_rate(float(RF_RATE), 0)
        self.usrp.set_tx_freq(_uhd_tune(freq), 0)
        self.usrp.set_rx_freq(_uhd_tune(freq), 0)
        self.usrp.set_tx_gain(float(tx_gain), 0)
        self.usrp.set_rx_gain(float(rx_gain), 0)
        try:
            self.usrp.set_tx_bandwidth(float(RF_BANDWIDTH), 0)
            self.usrp.set_rx_bandwidth(float(RF_BANDWIDTH), 0)
        except Exception:
            pass
        try:
            self.usrp.set_tx_antenna("TX/RX", 0)
            self.usrp.set_rx_antenna("TX/RX", 0)
        except Exception:
            pass
        self.tx_streamer = self.usrp.get_tx_stream(_make_stream_args())
        self.rx_streamer = self.usrp.get_rx_stream(_make_stream_args())
        self.rx_buffer = int(rx_buffer)
        self._md = uhd.types.RXMetadata()
        _start_rx_continuous(self.rx_streamer)
        for _ in range(3):
            try:
                self.capture()
            except Exception:
                pass
        # cyclic TX 仿真状态
        self._burst = None
        self._lock = threading.Lock()
        self._loop_thread = None
        self._loop_stop = threading.Event()
        print(f"[TRX] 就绪  Freq={freq/1e9:.2f}GHz  SR={RF_RATE/1e6:.1f}MHz  "
              f"TX={tx_gain}dB  RX={rx_gain}dB")

    def _cyclic_loop(self):
        md = uhd.types.TXMetadata()
        md.has_time_spec = False
        md.start_of_burst = False
        md.end_of_burst = False
        while not self._loop_stop.is_set():
            with self._lock:
                burst = self._burst
            if burst is None:
                time.sleep(0.001)
                continue
            try:
                self.tx_streamer.send(burst, md, 0.5)
            except Exception:
                time.sleep(0.001)

    def push_tx(self, baseband: np.ndarray):
        """Pluto 兼容接口: baseband → upsample → 投到 cyclic 线程."""
        rf = upsample_to_rf(baseband)
        peak = np.max(np.abs(rf))
        if peak > 0:
            rf = rf / peak * 0.7
        tx_iq = rf.astype(np.complex64)
        with self._lock:
            self._burst = tx_iq
        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._loop_stop.clear()
            self._loop_thread = threading.Thread(
                target=self._cyclic_loop, daemon=True)
            self._loop_thread.start()
        # 跟 Pluto 版同样语义: push 后让 RX 丢掉历史 buffer,
        # UHD 的 recv 不会缓存太多, 这里靠 stream_cmd 复位.
        # 不过实测 UHD 连续流不需要重置, 留空避免引入抖动.

    def capture(self) -> np.ndarray:
        rf = np.zeros(self.rx_buffer, dtype=np.complex64)
        filled = 0
        while filled < self.rx_buffer:
            n = self.rx_streamer.recv(rf[filled:], self._md, 0.5)
            if n <= 0:
                break
            filled += n
        if filled == 0:
            return np.zeros(0, dtype=np.complex64)
        if filled < self.rx_buffer:
            rf = rf[:filled]
        return downsample_to_bb(rf)

    def close(self):
        self._loop_stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1)
        _stop_rx_continuous(self.rx_streamer)


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
    h264_enc_sim = None
    h264_dec_sim = None
    if args.codec in ("h264", "h265"):
        h264_enc_sim = _H264Encoder(args.fwidth, args.fheight,
                                     args.h264_bitrate, args.h264_gop,
                                     preset=args.h264_preset)
        h264_dec_sim = _H264Decoder()
    print(f"[SIM] 仿真 SNR={args.snr}dB  分辨率={args.fwidth}x{args.fheight}  "
          f"codec={args.codec} 质量={args.jpeg}")

    while time.time() - t0 < args.duration:
        frame = get_next_frame(cap, frame_idx)
        if frame is None:
            break

        if h264_enc_sim is not None:
            jpg = h264_enc_sim.encode(frame)
            if not jpg:
                cv2.imshow("TX", cv2.resize(frame, (480, 360),
                                             interpolation=cv2.INTER_CUBIC))
                cv2.waitKey(1)
                frame_idx += 1
                continue
        else:
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
            rx_frame = (h264_dec_sim.decode(jpg_rx) if h264_dec_sim is not None
                        else jpeg_bytes_to_frame(jpg_rx))
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
    if not HAS_USRP:
        print("[错误] 未安装 uhd, 无法使用 loopback")
        return
    trx = UsrpTRX(args.uri, args.freq * 1e6, args.gain, args.rx_gain,
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
    last_completed_fid = -1
    h264_enc_lb = None
    h264_dec_lb = None
    if args.codec in ("h264", "h265"):
        h264_enc_lb = _H264Encoder(args.fwidth, args.fheight,
                                    args.h264_bitrate, args.h264_gop,
                                    preset=args.h264_preset)
        h264_dec_lb = _H264Decoder()

    try:
        while time.time() - t0 < args.duration:
            frame = get_next_frame(cap, frame_idx)
            if frame is None:
                break
            if h264_enc_lb is not None:
                jpg = h264_enc_lb.encode(frame)
                if not jpg:
                    cv2.imshow("TX", cv2.resize(frame, (480, 360),
                                                 interpolation=cv2.INTER_CUBIC))
                    cv2.waitKey(1)
                    frame_idx += 1
                    continue
            else:
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
                    if h264_dec_lb is not None:
                        if last_completed_fid != -1:
                            gap = (fid - last_completed_fid - 1) & 0xFFFF
                            if 0 < gap < 0x8000:
                                h264_dec_lb.reset()
                        rx_frame = h264_dec_lb.decode(full)
                        last_completed_fid = fid
                    else:
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

_BURST_LIMIT = 2 ** 21      # baseband sample 上限 (抬高以容纳 720p 等大帧, 防截断)


class FrameProducer:
    """主线程: 取一帧 -> JPEG 压缩 -> 切包 -> 拼成空口 burst.

    fixed_bb_size: 非零时把 baseband burst 零填充到固定长度.
    pre_upsample: True 时主线程做完 upsample/normalize/cast, 把准备好的
        int 复数 IQ 给 worker; False 时返回 baseband, 由 worker (UsrpTRX)
        自己 upsample.
    """

    def __init__(self, args, cap, fixed_bb_size: int = 0,
                 pre_upsample: bool = False):
        self.args = args
        self.cap = cap
        self.idx = 0
        self.fixed_bb_size = fixed_bb_size
        self.pre_upsample = pre_upsample
        self.h264_enc = None
        if args.codec in ("h264", "h265"):
            self.h264_enc = _H264Encoder(args.fwidth, args.fheight,
                                         args.h264_bitrate, args.h264_gop,
                                         preset=args.h264_preset)
            print(f"[ENC] H.264 libx264  {args.fwidth}x{args.fheight}  "
                  f"bitrate={args.h264_bitrate}  gop={args.h264_gop}  "
                  f"preset={args.h264_preset}")

    def next(self):
        frame = get_next_frame(self.cap, self.idx)
        if frame is None:
            return None
        if self.h264_enc is not None:
            jpg = self.h264_enc.encode(frame)
            if not jpg:
                # 编码器 warm-up 阶段可能返回空 packet; idx 不前进, skip
                out = (frame, None, self.idx, 0, 0)
                return out
        else:
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
        # async 模式下 UsrpTRX.push_tx 自己做, 这里要保留 baseband.
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
                      stats: dict,
                      codec: str = "jpg"):
    """专职捕包+解码线程. capture_fn() 阻塞返回基带 IQ.

    codec='h264' 时维持一个长生命周期 PyAV 解码器, 在 fid 跳变 (跨 GOP 丢
    帧) 时重置, 等待下一个 IDR.
    """
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    last_completed_fid = -1
    h264_dec = _H264Decoder() if codec in ("h264", "h265") else None
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
                if h264_dec is not None:
                    # fid 正向跳变 -> 跨 GOP 丢帧, P-frame 引用断, 必须 reset
                    if last_completed_fid != -1:
                        gap = (fid - last_completed_fid - 1) & 0xFFFF
                        if 0 < gap < 0x8000:
                            h264_dec.reset()
                    rx_frame = h264_dec.decode(full)
                    last_completed_fid = fid
                else:
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
        target=_rx_decode_worker,
        args=(capture_fn, rx_q, stop_event, rx_stats, args.codec),
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
            if burst is None:
                # encoder warm-up 或本帧无 packet 输出, skip 但保持 UI 活
                cv2.imshow("TX", cv2.resize(frame, (480, 360),
                                             interpolation=cv2.INTER_CUBIC))
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.005)
                continue

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
                       min_period: float = 0.0,
                       codec: str = "jpg"):
    """单 Pluto 必须 serialize: 一个 worker 串行 push + capture + decode.

    实测两个 worker 并发会让 RX 饿死 (libiio context 锁 + USB 仅一路).
    所以这里只在 main 给了新 burst 时才 push, 平时纯 capture.
    min_period 用来限制 push 频率, 同 dual 的理由 (避免 destroy 间隙吃光 cyclic 稳态).
    """
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    last_completed_fid = -1
    last_push = 0.0
    h264_dec = _H264Decoder() if codec in ("h264", "h265") else None

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
                if h264_dec is not None:
                    if last_completed_fid != -1:
                        gap = (fid - last_completed_fid - 1) & 0xFFFF
                        if 0 < gap < 0x8000:
                            h264_dec.reset()
                    rx_frame = h264_dec.decode(full)
                    last_completed_fid = fid
                else:
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
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    trx = UsrpTRX(args.uri, args.freq * 1e6, args.gain, args.rx_gain,
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
              tx_stats, rx_stats, tx_min_period, args.codec),
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
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    if args.tx_uri == args.rx_uri:
        print(f"[错误] dual 模式 --tx-uri 和 --rx-uri 不能相同 ({args.tx_uri})")
        print("       同一台 Pluto 用 --mode async 即可")
        return
    tx = UsrpTX(args.tx_uri, args.freq * 1e6, args.gain, streaming=False)
    rx = UsrpRX(args.rx_uri, args.freq * 1e6, args.rx_gain,
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
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    uri = args.tx_uri if args.mode == "txbeacon" else args.uri
    tx = UsrpTX(uri, args.freq * 1e6, args.gain, streaming=False,
                chan=args.tx_chan)

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

    n_pkt_per_burst = len(chunks)      # 每次 cyclic send 发出的包数 (K)
    try:
        tx.push(burst)
        print(f"[BEACON] 持续发射中 (cyclic, freq={args.freq:.1f}MHz, "
              f"TX_GAIN={args.gain}dB, K={n_pkt_per_burst}pkt/burst). Ctrl+C 停止.",
              flush=True)
        t0 = time.time()
        while time.time() - t0 < args.duration:
            time.sleep(0.3)
            # 周期换行上报累计已发包数, 供 power_tune 取窗口内 Δ 发射包数算 PDR
            print(f"[BEACON] elapsed {time.time()-t0:.1f}s "
                  f"txpkts={tx._send_count * n_pkt_per_burst}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        tx.close()
    print(f"[BEACON] 已停止 (共 {tx._send_count} bursts, "
          f"{tx._send_count * n_pkt_per_burst} pkts)", flush=True)


def run_rxprobe(args):
    """RX 诊断: 不仅测能量, 还把每一捕的 detection/LTS/Header/CRC 各阶段成功率拆开打."""
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    uri = args.rx_uri if args.mode == "rxprobe" else args.uri
    rx = UsrpRX(uri, args.freq * 1e6, args.rx_gain, rx_buffer=2 ** 18,
                chan=args.rx_chan)
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
    rf_samps = 0                       # 累计采集的 RF 样本数, 用于算占空比
    cum = {"det": 0, "lts": 0, "hdr": 0, "crc": 0}
    try:
        while time.time() - t0 < args.duration:
            samples = rx.capture()
            n += 1
            rf_samps += rx.rx_buffer
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
    elapsed = max(time.time() - t0, 1e-9)
    # 占空比 = 实际采集的 RF 样本数 / 同期空口总样本数; 调谐器据此把 RX 盲区
    # (含高增益时解码变慢导致的更长处理间隙) 从 PDR 里除掉 -> 还原真链路成功率.
    duty = rf_samps / (elapsed * RF_RATE) if RF_RATE > 0 else 0.0
    print(f"\n[结果] {n} 次 capture, 累计 det={cum['det']} LTS={cum['lts']} "
          f"HDR={cum['hdr']} CRC={cum['crc']} duty={duty*100:.1f}%")


# ============================================================================
# 主入口
# ============================================================================

def _build_frame_burst(jpg: bytes, frame_idx: int, args):
    """把一帧已编码字节流切包 -> OFDM baseband 波形 burst. 返回 (burst, n_pkt).

    纯 numpy/FFT, 无视频编码; 用于实时与预编码两条发射路径共用."""
    silence = np.zeros(NSYM * 4, dtype=np.complex64)
    parts = []
    if args.fec:
        # FEC: 切 k 个数据块 -> 编码 n 个块, 每块前置 5 字节子头(k, frame_len)
        k, n_pkt, frame_len, enc = fec.fec_pack_frame(
            jpg, args.fec_block, args.fec_overhead)
        sub = struct.pack("<BI", k, frame_len)
        for i in range(n_pkt):
            wf, _ = build_packet_waveform(
                sub + enc[i].tobytes(), frame_idx, i, n_pkt)
            parts.append(wf)
            parts.append(silence)
    else:
        chunks = split_payload(jpg)
        n_pkt = len(chunks)
        for pid, chunk in enumerate(chunks):
            wf, _ = build_packet_waveform(chunk, frame_idx, pid, n_pkt)
            parts.append(wf)
            parts.append(silence)
    burst = np.concatenate(parts).astype(np.complex64)
    if len(burst) > _BURST_LIMIT:
        print(f"\n[WARN] burst {len(burst)} > {_BURST_LIMIT}, 截断 "
              f"(提高 --h264-crf 或降低分辨率以缩小最大帧)")
        burst = burst[:_BURST_LIMIT]
    return burst, n_pkt


def _preencode_file(args):
    """离线把整段视频逐帧编码成 [(缩略图, 编码字节)] 列表, 一次性付清编码 CPU.

    仅文件源可用 (摄像头/图案无法预读未来). 之后发射循环只切包/调制/推送,
    不再受编码器 fps (如 720p ~10fps) 限制 -> 发射帧率可冲到空口/装配上限.
    缩略图存 320x240 省内存; --preencode-max 限制帧数防超大文件吃光内存."""
    cap = open_video_source(args.source, args.input)
    if cap is None:
        return None
    enc = None
    if args.codec in ("h264", "h265"):
        enc = _H264Encoder(args.fwidth, args.fheight, args.h264_bitrate,
                           args.h264_gop, preset=args.h264_preset,
                           crf=args.h264_crf)
    frames = []
    t0 = time.time()
    print(f"[TXVID] 预编码整段视频中 (一次性, codec={args.codec} "
          f"{args.fwidth}x{args.fheight} crf={args.h264_crf}) ...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if enc is not None:
                jpg = enc.encode(frame)
                if not jpg:                 # 编码器预热, 跳过该源帧
                    continue
            else:
                jpg = frame_to_jpeg_bytes(frame, args.fwidth, args.fheight,
                                          args.jpeg, codec=args.codec)
            frames.append((cv2.resize(frame, (320, 240)), jpg))
            if len(frames) % 30 == 0:
                print(f"\r  已编码 {len(frames)} 帧 ({time.time()-t0:.1f}s)",
                      end="")
            if len(frames) >= args.preencode_max:
                print(f"\n  达到 --preencode-max={args.preencode_max} 上限, 停止")
                break
    finally:
        cap.release()
    if frames:
        sizes = [len(j) for _, j in frames]
        print(f"\n[TXVID] 预编码完成: {len(frames)} 帧, {time.time()-t0:.1f}s, "
              f"帧字节 avg={sum(sizes)//len(sizes)} max={max(sizes)}")
    return frames


def run_txvideo(args):
    """纯发射进程: 持续把视频帧编码/切包/cyclic 发射. 配合另一台/另一进程的
    rxvideo 使用. 收发分进程可避免 dual 单进程里 TX 线程与 RX 重解码的
    GIL/USB 争用 (那会导致 TX 欠载、RX 捕获被打碎而解不出).

    --preencode (仅文件源): 启动时一次性编码整段视频, 之后发射不再占用编码 CPU,
    发射帧率不再被编码器吞吐 (如 720p ~10fps) 卡住."""
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    pre = None
    if args.preencode:
        if args.source != "file":
            print("[TXVID] --preencode 仅支持 --source file, 已忽略 (回退实时编码)")
        else:
            pre = _preencode_file(args)
            if not pre:
                print("[TXVID] 预编码无输出, 回退实时编码")
                pre = None

    tx = UsrpTX(args.tx_uri, args.freq * 1e6, args.gain, streaming=False,
                chan=args.tx_chan)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    min_period = 1.0 / max(args.tx_fps, 0.1)
    mode_tag = "预编码缓冲" if pre is not None else "实时编码"
    print(f"[TXVID] 发射视频 {args.tx_uri} @ {args.freq:.1f}MHz TX={args.gain}dB "
          f"codec={args.codec} {args.fwidth}x{args.fheight} ({mode_tag})")

    # 实时路径所需 (预编码路径不用)
    cap = None
    h264_enc = None
    if pre is None:
        cap = open_video_source(args.source, args.input)
        if args.codec in ("h264", "h265"):
            h264_enc = _H264Encoder(args.fwidth, args.fheight,
                                    args.h264_bitrate, args.h264_gop,
                                    preset=args.h264_preset, crf=args.h264_crf)

    frame_idx = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            tframe = time.time()
            if pre is not None:
                # 预编码缓冲: 取已编码字节 (循环播放), 只切包/调制/推送
                thumb, jpg = pre[frame_idx % len(pre)]
                disp = thumb
            else:
                frame = get_next_frame(cap, frame_idx)
                if frame is None:
                    break
                if h264_enc is not None:
                    jpg = h264_enc.encode(frame)
                    if not jpg:             # 编码器预热, 本源帧无输出
                        cv2.imshow("TX", cv2.resize(frame, (480, 360)))
                        cv2.waitKey(1)
                        frame_idx += 1
                        continue
                else:
                    jpg = frame_to_jpeg_bytes(frame, args.fwidth, args.fheight,
                                              args.jpeg, codec=args.codec)
                disp = cv2.resize(frame, (480, 360))

            burst, n_pkt = _build_frame_burst(jpg, frame_idx, args)
            # baseband -> upsample/normalize -> cyclic 线程持续发当前帧
            tx.push(burst)
            cv2.imshow("TX", disp)
            if cv2.waitKey(1) == 27:
                break
            elapsed = time.time() - t0
            print(f"\r[TXVID] frame#{frame_idx} 包={n_pkt} 帧={len(jpg)}B  "
                  f"{elapsed:.1f}s", end="")
            frame_idx += 1
            dt = time.time() - tframe
            if dt < min_period:
                time.sleep(min_period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        tx.close()
    el = max(time.time() - t0, 1e-3)
    print(f"\n[TXVID] 结束, 共发射 {frame_idx} 帧 ({frame_idx/el:.1f} fps)")


def run_rxvideo(args):
    """纯接收进程: 持续捕获+解码+重组+显示视频. 配合 txvideo 使用.

    采集与解码分线程 (方案 B): 采集线程背靠背 capture() 喂队列(丢最旧保最新),
    主线程取队列做 detect/decode/重组/显示. 单线程同步版会在解码的几十毫秒里
    停止采集, 监听占空比仅 ~15% -> 表现为'偶尔才收到信号'. 分线程后采集几乎
    不被打断, 占空比提到 ~90%+, 帧率稳定。"""
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    rx = UsrpRX(args.rx_uri, args.freq * 1e6, args.rx_gain,
                rx_buffer=2 ** args.rx_bits, chan=args.rx_chan)
    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)
    h264_dec = _H264Decoder() if args.codec in ("h264", "h265") else None

    # 采集线程: 不停 capture() 塞进有界队列, 满了丢最旧 (永远处理最新采集).
    cap_q: queue.Queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    cap_count = {"n": 0}

    def _capture_loop():
        # 采集线程只做 recv (capture_rf), 下采样移到主线程, 把空口占空比拉满.
        while not stop_event.is_set():
            try:
                s = rx.capture_rf()
            except Exception:
                continue
            if s is None or len(s) == 0:
                continue
            cap_count["n"] += 1
            try:
                cap_q.put_nowait(s)
            except queue.Full:
                try:
                    cap_q.get_nowait()          # 丢最旧
                except queue.Empty:
                    pass
                try:
                    cap_q.put_nowait(s)
                except queue.Full:
                    pass

    cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()

    frames_unique = 0
    last_rx_frame = None
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    shown_fid = -1
    last_completed_fid = -1
    n_cap = 0
    n_hdr = 0          # 成功解出帧头(magic 命中)的包数
    n_crc = 0          # 其中 CRC 通过的包数
    last_bytes = 0
    fps_win = 1.5      # 实时 fps 统计窗口 (秒)
    fps_times: list = []
    last_print = 0.0
    t0 = time.time()
    print(f"[RXVID] 接收视频 {args.rx_uri} @ {args.freq:.1f}MHz RX={args.rx_gain}dB "
          f"codec={args.codec}  (采集/解码分线程)")
    try:
        while time.time() - t0 < args.duration:
            try:
                samples = cap_q.get(timeout=0.5)
            except queue.Empty:
                if last_rx_frame is not None:    # 无新采集也保持 UI 活
                    cv2.imshow("RX", last_rx_frame)
                if cv2.waitKey(1) == 27:
                    break
                continue
            n_cap = cap_count["n"]
            # 下采样在此 (消费线程) 做, 不占用采集线程的 recv 时间. OSF=1 时直通.
            samples = downsample_to_bb(samples)
            for st in detect_packets(samples, threshold=0.55):
                res = decode_packet(samples, st)
                if res is None:
                    continue
                n_hdr += 1
                fid, pkt_id, tot, payload, ok = res
                if ok:
                    n_crc += 1
                if not ok or tot == 0 or pkt_id >= tot:
                    continue
                if fid == shown_fid:
                    continue          # 该帧已显示过 (cyclic 重复), 跳过省算力
                if fid != last_frame_id:
                    payloads_buf = {}
                    last_total = tot
                    last_frame_id = fid
                payloads_buf[pkt_id] = payload
                full = None
                if args.fec:
                    # FEC: 收齐 >= k 个块即可纠错恢复整帧 (k 在每块 5 字节子头里)
                    if len(payload) >= 5:
                        k_blk, frame_len = struct.unpack("<BI", payload[:5])
                        if len(payloads_buf) >= k_blk:
                            try:
                                bd = {idx: np.frombuffer(p[5:], dtype=np.uint8)
                                      for idx, p in payloads_buf.items()
                                      if len(p) >= 5}
                                full = fec.fec_unpack_frame(
                                    bd, k_blk, last_total, frame_len)
                            except Exception:
                                full = None
                elif last_total > 0 and all(
                        i in payloads_buf for i in range(last_total)):
                    full = b"".join(payloads_buf[i] for i in range(last_total))
                if full is not None:
                    if h264_dec is not None:
                        if last_completed_fid != -1:
                            gap = (fid - last_completed_fid - 1) & 0xFFFF
                            if 0 < gap < 0x8000:
                                h264_dec.reset()
                        rx_frame = h264_dec.decode(full)
                        last_completed_fid = fid
                    else:
                        rx_frame = jpeg_bytes_to_frame(full)
                    payloads_buf = {}
                    last_frame_id = -1
                    if rx_frame is not None:
                        last_rx_frame = rx_frame
                        frames_unique += 1
                        last_bytes = len(full)
                        fps_times.append(time.time())
                        shown_fid = fid
                        # 出一帧就停止扫描本 buffer 余下的 cyclic 重复包: 一个
                        # capture 窗口里同一帧通常重复 ~6 次 (~120 个检测点全解会
                        # 吃满 ~80ms, 拖慢 buffer 周转 -> 唯一帧率被压到 ~7fps).
                        # 解到第一帧新帧即 break, buffer 周转快数倍, 唯一帧率大涨.
                        # 下一帧由后续 capture (它在那帧 hold 窗口里占主导) 收下.
                        break
            # 原分辨率显示 (窗口可手动拖拽缩放), 不再强制缩到 480x360
            if last_rx_frame is not None:
                cv2.imshow("RX", last_rx_frame)
            if cv2.waitKey(1) == 27:
                break
            now = time.time()
            fps_times = [t for t in fps_times if now - t <= fps_win]
            if now - last_print >= 0.3:
                last_print = now
                elapsed = now - t0
                rt_fps = len(fps_times) / fps_win
                avg_fps = frames_unique / max(elapsed, 1e-3)
                quality = 100.0 * n_crc / max(n_hdr, 1)
                print(f"\r[RXVID] 实时fps={rt_fps:4.1f} 平均fps={avg_fps:4.1f}  "
                      f"唯一帧={frames_unique}  链路质量={quality:3.0f}%(CRC通过)  "
                      f"帧={last_bytes / 1024:5.1f}KB  cap={n_cap}  {elapsed:4.0f}s   ",
                      end="")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap_thread.join(timeout=2)
        cv2.destroyAllWindows()
        rx.close()
    el = max(time.time() - t0, 1e-3)
    print(f"\n[RXVID] 结束: 唯一帧 {frames_unique}, 平均 {frames_unique / el:.1f} fps, "
          f"链路质量 {100.0 * n_crc / max(n_hdr, 1):.0f}% (CRC通过率)")


def main():
    global SAMPLE_RATE_BB, RF_RATE, RF_BANDWIDTH, _AV_CODEC
    global OSF, _RESAMPLE_UP, _RESAMPLE_DOWN
    p = argparse.ArgumentParser(
        description="USRP x310 OFDM 彩色视频传输 (5.8 GHz, 802.11a 风格)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python usrp_ofdm_video.py --mode sim --snr 20                          # 仿真
  python usrp_ofdm_video.py --mode loopback                              # 单台同步 (调试用)
  python usrp_ofdm_video.py --mode async                                 # 单台异步线程
  python usrp_ofdm_video.py --mode dual --tx-uri {USRP_DEFAULT_ADDR} \\
                            --rx-uri {USRP_SECOND_ADDR}                  # 两台 x310
  python usrp_ofdm_video.py --mode dual --codec h264 --h264-bitrate 1500000 \\
                            --fwidth 640 --fheight 480 --tx-fps 30
""")
    p.add_argument("--mode", default="sim",
                   choices=["sim", "loopback", "async", "dual",
                            "rxprobe", "txbeacon", "txvideo", "rxvideo"])
    p.add_argument("--source", default="file",
                   choices=["file", "camera", "pattern"])
    p.add_argument("--input", default="", help="视频文件路径")
    p.add_argument("--uri", default=USRP_DEFAULT_ADDR,
                   help="单台 USRP UHD 地址 (sim/loopback/async), "
                        f"默认 {USRP_DEFAULT_ADDR}")
    p.add_argument("--tx-uri", default=USRP_DEFAULT_ADDR,
                   help="dual 模式 TX USRP 地址, 默认 "
                        f"{USRP_DEFAULT_ADDR}")
    p.add_argument("--rx-uri", default=USRP_SECOND_ADDR,
                   help="dual 模式 RX USRP 地址, 默认 "
                        f"{USRP_SECOND_ADDR}")
    p.add_argument("--freq", type=float, default=CARRIER_FREQ_DEFAULT / 1e6,
                   help="中心频率 MHz")
    p.add_argument("--gain", type=float, default=TX_GAIN_DEFAULT,
                   help="TX 增益 dB")
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DEFAULT,
                   help="RX 增益 dB")
    p.add_argument("--tx-chan", type=int, default=0, choices=[0, 1],
                   help="B210 发射 RF 前端通道: 0=RF A 的 TX/RX 口(默认), "
                        "1=RF B 的 TX/RX 口. 原 TX/RX 口损坏时改用 1.")
    p.add_argument("--rx-chan", type=int, default=0, choices=[0, 1],
                   help="B210 接收 RF 前端通道: 0=RF A 的 TX/RX 口(默认), "
                        "1=RF B 的 TX/RX 口.")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--fwidth", type=int, default=320, help="视频帧宽度 (彩色)")
    p.add_argument("--fheight", type=int, default=240, help="视频帧高度 (彩色)")
    p.add_argument("--jpeg", type=int, default=JPEG_QUALITY,
                   help="编码质量 1-100 (JPEG/WebP 共用此参数)")
    p.add_argument("--codec", default="jpg",
                   choices=["jpg", "webp", "h264", "h265"],
                   help="帧编码: jpg / webp (~+30%%质量) / h264 (~+50%%质量)")
    p.add_argument("--h264-bitrate", type=int, default=500_000,
                   help="H.264 目标比特率 (bit/s). 默认 500kbps. "
                        "320x240 推荐 500-800k, 480x360 推荐 800-1500k")
    p.add_argument("--h264-gop", type=int, default=1,
                   help="H.264 GOP 长度. 默认 1 = 每帧独立 (intra-only), "
                        "丢帧不影响后续, 9%% 帧丢失环境下唯一可行选择. "
                        "GOP>1 在丢帧链路上会卡屏到下次 IDR, 不推荐.")
    p.add_argument("--h264-preset", default="fast",
                   choices=["ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow"],
                   help="libx264 编码 preset. 越慢压缩率越高 (同字节画质更好). "
                        "默认 fast. 主线程 CPU 闲置, 可推到 medium/slow 拿 "
                        "更高画质, 编码时间从 ~2ms 涨到 ~15ms 通常无碍.")
    p.add_argument("--h264-crf", type=int, default=0,
                   help="H.264 恒定质量 CRF: 0=关(用 --h264-bitrate); >0=恒定质量, "
                        "每帧画质固定且与 fps 无关 (解决码率被按30fps平摊导致每帧过糊). "
                        "越小越清晰, 推荐 20~26; 设了它就忽略 --h264-bitrate.")
    p.add_argument("--fec", action="store_true",
                   help="开启前向纠错(Reed-Solomon erasure): 收齐任意 k/n 个包即恢复整帧, "
                        "在 16QAM 等丢包链路(CRC 80-95%%)下稳定出帧. 收发两端须一致.")
    p.add_argument("--fec-overhead", type=float, default=0.4,
                   help="FEC 冗余比例 (默认 0.4 = 40%% 冗余包). 丢包越多需越大; "
                        "0.4 可容忍约 28%% 丢包.")
    p.add_argument("--fec-block", type=int, default=1100,
                   help="FEC 每块字节数 (默认 1100). 越小块越多、单块越易收到但开销略增.")
    p.add_argument("--snr", type=float, default=20, help="仿真 SNR (dB)")
    p.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam", "64qam"],
                   help="子载波调制: qpsk (默认, 鲁棒) 或 16qam (吞吐 x2, BER 高)")
    p.add_argument("--tx-fps", type=float, default=12.0,
                   help="async/dual 模式 TX 节流上限 (fps). 默认 12; "
                        "TX 推太快会被 buffer destroy 间隙吃掉 cyclic 稳态时间")
    p.add_argument("--rx-bits", type=int, default=18,
                   help="rx_buffer = 2**rx-bits RF 样本数. 默认 18 (8.7ms); "
                        "dual 推荐 19 (17.5ms, 跨 cyclic 周期更稳)")
    p.add_argument("--samp-rate", type=float, default=SAMPLE_RATE_BB / 1e6,
                   help="基带采样率 MHz (默认 5). 提高=增带宽/吞吐, 支撑更高 fps "
                        "和更大分辨率. B210 over USB3 建议 ≤20 (RF=1.5x, ≤30MHz); "
                        "过高会因主机 CPU/USB 跟不上而掉帧. 收发两端必须一致.")
    p.add_argument("--preencode", action="store_true",
                   help="txvideo 文件源专用: 启动时一次性编码整段视频, 之后发射只切包/"
                        "调制/推送, 不再受编码器吞吐 (如 720p ~10fps) 限制 -> 发射帧率冲到 "
                        "空口/RX装配上限. 编码会在开头集中花一次时间.")
    p.add_argument("--preencode-max", type=int, default=2000,
                   help="--preencode 最多缓存帧数 (默认 2000, 防超大文件吃光内存).")
    p.add_argument("--osf", type=float, default=1.5, choices=[1.0, 1.5],
                   help="过采样因子 (RF=OSF*基带). 1.5=旧默认; "
                        "1.0=不过采样 (RF=BB, 同 802.11a, 空子载波即保护带), 省掉 RX 端 "
                        "~70%% 重采样开销 -> 采集占空比/帧率大涨, 强烈推荐. 同采样率下 "
                        "1.0 与 1.5 的空口数据率完全相同. 收发两端必须一致.")

    args = p.parse_args()
    set_modulation(args.mod)

    # 据 --samp-rate / --osf 覆盖全局采样率 (收发两端必须一致). OFDM 处理按样本数,
    # 与速率无关. OSF=1 时重采样直通 (UP==DOWN); OSF=1.5 时用 3/2.
    SAMPLE_RATE_BB = args.samp_rate * 1e6
    OSF = args.osf
    if abs(OSF - 1.0) < 1e-6:
        _RESAMPLE_UP = _RESAMPLE_DOWN = 1
    else:                                  # 1.5 (argparse choices 已限定)
        _RESAMPLE_UP, _RESAMPLE_DOWN = 3, 2
    RF_RATE = int(round(SAMPLE_RATE_BB * OSF))
    RF_BANDWIDTH = min(RF_RATE * 0.9, 56e6)
    _AV_CODEC = "hevc" if args.codec == "h265" else "h264"

    print(f"[CFG] modulation={_MOD_NAME}  bits/sym={_MOD_BITS_PER_SYM}")
    print(f"[CFG] 采样率 BB={SAMPLE_RATE_BB/1e6:.1f}MHz  RF={RF_RATE/1e6:.1f}MHz  "
          f"OSF={OSF:g}{' (无重采样)' if _RESAMPLE_UP == _RESAMPLE_DOWN else ''}  "
          f"BW={RF_BANDWIDTH/1e6:.1f}MHz")

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
    elif args.mode == "txvideo":
        run_txvideo(args)
    elif args.mode == "rxvideo":
        run_rxvideo(args)


if __name__ == "__main__":
    main()
