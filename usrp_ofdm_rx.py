#!/usr/bin/env python3
"""
USRP X310 OFDM 彩色视频接收端 (5.8 GHz)
===============================================
由 usrp_ofdm_video.py 拆出的纯 RX 版本, 与 usrp_ofdm_tx.py 配合在两台电脑
上分别运行收/发. 协议与 dual 模式 100% 一致.

典型用法:
  python usrp_ofdm_rx.py --uri addr=192.168.20.2 \\
                         --freq 5800 --rx-gain 20 \\
                         --codec h264

诊断模式:
  python usrp_ofdm_rx.py --mode rxprobe                  # 分阶段统计 det/LTS/HDR/CRC

依赖: pip install uhd numpy scipy opencv-python av
"""

import argparse
import queue
import struct
import threading
import time
import zlib

import cv2
import numpy as np
from scipy import signal as sps

try:
    import uhd
    HAS_USRP = True
except ImportError:
    HAS_USRP = False
    print("[警告] 未安装 uhd Python 绑定. "
          "安装: 从 https://files.ettus.com 下载 UHD 启用 Python 绑定, "
          "或 conda install -c conda-forge uhd")


# ============================================================================
# 全局射频/OFDM 配置
# ============================================================================

CARRIER_FREQ_DEFAULT = 5.8e9
SAMPLE_RATE_BB = 20e6
OSF = 1.5
RF_RATE = int(SAMPLE_RATE_BB * OSF)
RF_BANDWIDTH = 18e6
RX_GAIN_DEFAULT = 20
USRP_DEFAULT_ADDR = "addr=192.168.20.2"

NFFT = 64
NCP = 16
NSYM = NFFT + NCP

DATA_SC = np.array([
    -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -13, -12,
    -11, -10, -9, -8, -6, -5, -4, -3, -2, -1,
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 23, 24, 25, 26,
])
PILOT_SC = np.array([-21, -7, 7, 21])
N_DATA = len(DATA_SC)
N_PILOT = len(PILOT_SC)

LSTF_INDEX = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
LSTF_VAL = np.sqrt(13.0 / 6.0) * np.array([
    1 + 1j, -1 - 1j, 1 + 1j, -1 - 1j, -1 - 1j, 1 + 1j,
    -1 - 1j, -1 - 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j,
])

LLTF_VAL_NEG = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1,
                         -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
LLTF_VAL_POS = np.array([1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1,
                         -1, -1, 1, -1, 1, -1, 1, 1, 1, 1])

PILOT_PATTERN = np.array([1, 1, 1, -1], dtype=np.complex64)

_DATA_IDX = (DATA_SC % NFFT).astype(np.intp)
_PILOT_IDX = (PILOT_SC % NFFT).astype(np.intp)

SYNC_MAGIC = 0xA55A3CC3
HEADER_FMT = "<IHHII"
HEADER_LEN = struct.calcsize(HEADER_FMT)
CRC_LEN = 4


# ============================================================================
# 比特 / 解调 (RX 只需 decode 侧)
# ============================================================================

def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 2, dtype=np.int8)
    bits[0::2] = (np.real(symbols) < 0).astype(np.int8)
    bits[1::2] = (np.imag(symbols) < 0).astype(np.int8)
    return bits


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 4, dtype=np.int8)
    I = np.real(symbols) * np.sqrt(10)
    Q = np.imag(symbols) * np.sqrt(10)
    bits[0::4] = (I < 0).astype(np.int8)
    bits[1::4] = (np.abs(I) < 2).astype(np.int8)
    bits[2::4] = (Q < 0).astype(np.int8)
    bits[3::4] = (np.abs(Q) < 2).astype(np.int8)
    return bits


_MOD_BITS_PER_SYM = 2
_DEMOD_FN = qpsk_demodulate
_MOD_NAME = "qpsk"


def set_modulation(name: str):
    global _MOD_BITS_PER_SYM, _DEMOD_FN, _MOD_NAME
    name = name.lower()
    if name == "qpsk":
        _MOD_BITS_PER_SYM = 2
        _DEMOD_FN = qpsk_demodulate
    elif name in ("16qam", "qam16"):
        _MOD_BITS_PER_SYM = 4
        _DEMOD_FN = qam16_demodulate
        name = "16qam"
    else:
        raise ValueError(f"Unknown modulation: {name}")
    _MOD_NAME = name


# ============================================================================
# 前导码 (RX 侧: 需要时域 LLTF 模板 + 频域参考)
# ============================================================================

def _build_freq_grid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    grid = np.zeros(NFFT, dtype=np.complex64)
    for k, v in zip(indices, values):
        grid[k % NFFT] = v
    return grid


def make_lltf_time() -> np.ndarray:
    indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    values = np.concatenate([LLTF_VAL_NEG, LLTF_VAL_POS]).astype(np.complex64)
    grid = _build_freq_grid(values, indices)
    t = np.fft.ifft(grid) * NFFT / np.sqrt(52)
    sym = t.astype(np.complex64)
    gi = sym[-32:]
    return np.concatenate([gi, sym, sym]).astype(np.complex64)


def lltf_freq_reference() -> np.ndarray:
    indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    values = np.concatenate([LLTF_VAL_NEG, LLTF_VAL_POS]).astype(np.complex64)
    return _build_freq_grid(values, indices)


LLTF_TIME = make_lltf_time()
LLTF_FREQ_REF = lltf_freq_reference()
LEN_LSTF = 160
LEN_LLTF = 160
LEN_PREAMBLE = LEN_LSTF + LEN_LLTF


# ============================================================================
# OFDM 解调
# ============================================================================

def ofdm_demodulate(time_signal: np.ndarray, h_est: np.ndarray) -> np.ndarray:
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
# 包检测 / 同步 / 解码
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
            indices.append(i)
            i = j + 32
        else:
            i += 1
    return indices


def apply_cfo(x: np.ndarray, freq_per_sample: float) -> np.ndarray:
    n = np.arange(len(x))
    return (x * np.exp(-1j * freq_per_sample * n)).astype(np.complex64)


def find_lts_position(rx: np.ndarray, search_center: int,
                      search_radius: int = 200):
    template = LLTF_TIME[32:32 + 64]
    s = max(0, search_center - search_radius)
    e = min(len(rx), search_center + search_radius + 64)
    if e - s < 128 + 64:
        return None
    seg = rx[s:e]
    corr = np.correlate(seg, template, mode="valid")
    abs_corr = np.abs(corr)
    if len(abs_corr) < 65:
        return None

    peak1 = int(np.argmax(abs_corr))
    cand2_a = peak1 + 64
    cand2_b = peak1 - 64
    second = None
    if 0 <= cand2_a < len(abs_corr):
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
    return s + lts1, s + second, float(abs_corr[peak1])


def coarse_cfo_from_lstf(rx_lstf: np.ndarray) -> float:
    L = 16
    n = min(len(rx_lstf) - L, 9 * L)
    if n <= 0:
        return 0.0
    s = np.sum(np.conj(rx_lstf[:n]) * rx_lstf[L:L + n])
    return np.angle(s) / L


def fine_cfo_from_lts(rx_lts1: np.ndarray, rx_lts2: np.ndarray) -> float:
    return np.angle(np.sum(np.conj(rx_lts1) * rx_lts2)) / 64.0


def estimate_channel_from_lts(rx_lts1: np.ndarray,
                              rx_lts2: np.ndarray) -> np.ndarray:
    avg = 0.5 * (rx_lts1 + rx_lts2)
    H = np.fft.fft(avg) / np.sqrt(N_DATA + N_PILOT)
    H_ref = LLTF_FREQ_REF
    h_est = np.ones(NFFT, dtype=np.complex64)
    for sc in np.concatenate([np.arange(-26, 0), np.arange(1, 27)]):
        if H_ref[sc % NFFT] != 0:
            h_est[sc % NFFT] = H[sc % NFFT] / H_ref[sc % NFFT]
    return h_est


def decode_packet(rx: np.ndarray, start: int, max_data_sym: int = 600):
    """给定粗略检测点 start, 用 LTS 精确定位并解码.
    返回 (fid, pid, tot, payload, ok) 或 None."""
    if start < 0 or start + LEN_PREAMBLE > len(rx):
        return None

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

    lstf_seg = rx[lstf_start:lstf_start + 144]
    coarse = coarse_cfo_from_lstf(lstf_seg)

    seg_lts = rx[lts1_pos - 32:lts1_pos - 32 + 160]
    seg_lts = apply_cfo(seg_lts, coarse)
    rx_lts1 = seg_lts[32:32 + 64]
    rx_lts2 = seg_lts[32 + 64:32 + 128]

    fine = fine_cfo_from_lts(rx_lts1, rx_lts2)
    total_cfo = coarse + fine

    rx_lts1_c = apply_cfo(rx_lts1, fine)
    rx_lts2_c = (apply_cfo(rx_lts2, fine)
                 * np.exp(-1j * fine * 64))
    h_est = estimate_channel_from_lts(rx_lts1_c, rx_lts2_c)

    bits_per_sym = N_DATA * _MOD_BITS_PER_SYM

    head_sym_count = (HEADER_LEN * 8 + bits_per_sym - 1) // bits_per_sym
    if data_start + NSYM * head_sym_count > len(rx):
        return None
    data_chunk = rx[data_start:data_start + NSYM * (max_data_sym + 4)].copy()
    if len(data_chunk) < NSYM * head_sym_count:
        return None
    n_off = data_start - lstf_start
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
# 下采样 / 图像解码
# ============================================================================

def downsample_to_bb(x: np.ndarray) -> np.ndarray:
    return sps.resample_poly(x, up=2, down=3).astype(np.complex64)


class _H264Decoder:
    def __init__(self):
        try:
            import av
        except ImportError:
            raise RuntimeError("H.264 需要 PyAV. 安装: pip install av")
        self._av = av
        self.codec = av.CodecContext.create("h264", "r")
        self._last = None

    def reset(self):
        """fid 跳变后调用, 丢弃 P-frame 引用状态, 等待下一个 IDR."""
        self.codec = self._av.CodecContext.create("h264", "r")

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


def jpeg_bytes_to_frame(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ============================================================================
# USRP X310 RX 侧硬件接口
# ============================================================================

def _uhd_tune(freq: float):
    return uhd.libpyuhd.types.tune_request(float(freq))


def _make_stream_args():
    st = uhd.usrp.StreamArgs("fc32", "sc16")
    st.channels = [0]
    return st


def _start_rx_continuous(rx_streamer):
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)


def _stop_rx_continuous(rx_streamer):
    try:
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(cmd)
    except Exception:
        pass


class UsrpRX:
    """RX-only USRP X310. capture() 阻塞读 rx_buffer 个 RF 样本后下采样."""

    def __init__(self, uri: str, freq: float, rx_gain: float,
                 rx_buffer: int = 2 ** 18):
        if not HAS_USRP:
            raise RuntimeError("uhd 未安装")
        print(f"[RX]  连接 USRP {uri} ...")
        self.usrp = uhd.usrp.MultiUSRP(uri)
        self.usrp.set_rx_rate(float(RF_RATE), 0)
        self.usrp.set_rx_freq(_uhd_tune(freq), 0)
        self.usrp.set_rx_gain(float(rx_gain), 0)
        try:
            self.usrp.set_rx_bandwidth(float(RF_BANDWIDTH), 0)
        except Exception:
            pass
        try:
            self.usrp.set_rx_antenna("RX2", 0)
        except Exception:
            pass
        self.rx_streamer = self.usrp.get_rx_stream(_make_stream_args())
        self.rx_buffer = int(rx_buffer)
        self._md = uhd.types.RXMetadata()
        _start_rx_continuous(self.rx_streamer)
        for _ in range(3):
            try:
                self.capture()
            except Exception:
                pass
        print(f"[RX]  就绪  Freq={freq/1e9:.2f}GHz  RX={rx_gain}dB  "
              f"buf={rx_buffer}")

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
        _stop_rx_continuous(self.rx_streamer)


# ============================================================================
# RX worker
# ============================================================================

def _replace_latest(q: queue.Queue, item) -> bool:
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


def _rx_decode_worker(capture_fn, rx_q: queue.Queue,
                      stop_event: threading.Event,
                      stats: dict,
                      codec: str = "jpg"):
    """专职捕包+解码线程. codec='h264' 时维持长生命周期 PyAV 解码器,
    在 fid 跳变 (跨 GOP 丢帧) 时重置, 等待下一个 IDR."""
    payloads_buf: dict = {}
    last_total = 0
    last_frame_id = -1
    last_completed_fid = -1
    h264_dec = _H264Decoder() if codec == "h264" else None
    while not stop_event.is_set():
        try:
            samples = capture_fn()
        except Exception:
            continue
        stats["captures"] = stats.get("captures", 0) + 1
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


# ============================================================================
# 主流程
# ============================================================================

def run_rx(args):
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    rx = UsrpRX(args.uri, args.freq * 1e6, args.rx_gain,
                rx_buffer=2 ** args.rx_bits)

    cv2.namedWindow("RX", cv2.WINDOW_NORMAL)
    rx_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    rx_stats: dict = {}

    rx_thread = threading.Thread(
        target=_rx_decode_worker,
        args=(rx.capture, rx_q, stop_event, rx_stats, args.codec),
        daemon=True)
    rx_thread.start()

    last_rx = None
    rx_done = 0
    t0 = time.time()
    last_report = t0

    try:
        while time.time() - t0 < args.duration:
            while True:
                try:
                    last_rx = rx_q.get_nowait()
                    rx_done += 1
                except queue.Empty:
                    break

            if last_rx is not None:
                cv2.imshow("RX", cv2.resize(last_rx, (480, 360),
                                             interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) == 27:
                break

            now = time.time()
            if now - last_report > 0.5:
                elapsed = now - t0
                rx_fps = rx_done / max(elapsed, 1e-3)
                rms = rx_stats.get("last_rms", 0.0)
                peak = rx_stats.get("last_peak", 0.0)
                dets = rx_stats.get("last_dets", 0)
                caps = rx_stats.get("captures", 0)
                print(f"\r[RX] decoded={rx_done} ({rx_fps:.1f}fps)  "
                      f"caps={caps}  rms={rms:.3f} peak={peak:.2f} "
                      f"det={dets}  {elapsed:.1f}s", end="")
                last_report = now

            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        rx_thread.join(timeout=2)
        cv2.destroyAllWindows()
        rx.close()

    elapsed = time.time() - t0
    print(f"\n[结果] RX: 解出 {rx_done} 帧 "
          f"({rx_done/max(elapsed,1e-3):.1f}fps), 捕获 "
          f"{rx_stats.get('captures',0)} 次")


def run_rxprobe(args):
    """RX 诊断: 把每一捕的 detection/LTS/Header/CRC 各阶段成功率拆开打."""
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    rx = UsrpRX(args.uri, args.freq * 1e6, args.rx_gain, rx_buffer=2 ** 18)
    print(f"[PROBE] 监听 {args.uri} @ {args.freq:.1f} MHz, RX={args.rx_gain}dB.")
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
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="USRP X310 OFDM 视频接收端 (5.8 GHz, 802.11a 风格)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python usrp_ofdm_rx.py --uri {USRP_DEFAULT_ADDR} \\
                         --freq 5800 --rx-gain 20 --codec h264
  python usrp_ofdm_rx.py --mode rxprobe --rx-gain 25            # 链路诊断
""")
    p.add_argument("--mode", default="rx", choices=["rx", "rxprobe"],
                   help="rx=主流程; rxprobe=分阶段成功率诊断")
    p.add_argument("--uri", default=USRP_DEFAULT_ADDR,
                   help=f"RX USRP UHD 地址, 默认 {USRP_DEFAULT_ADDR}")
    p.add_argument("--freq", type=float, default=CARRIER_FREQ_DEFAULT / 1e6,
                   help="中心频率 MHz")
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DEFAULT,
                   help="RX 增益 dB")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--codec", default="jpg",
                   choices=["jpg", "webp", "h264"],
                   help="帧解码: 必须与 TX 端一致. jpg / webp 同用 cv2 自动识别")
    p.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam"],
                   help="子载波调制, 必须与 TX 端一致")
    p.add_argument("--rx-bits", type=int, default=18,
                   help="rx_buffer = 2**rx-bits RF 样本数. 默认 18 (8.7ms); "
                        "信号弱或抖动大时推荐 19 (17.5ms)")

    args = p.parse_args()
    set_modulation(args.mod)
    print(f"[CFG] modulation={_MOD_NAME}  bits/sym={_MOD_BITS_PER_SYM}")

    if args.mode == "rxprobe":
        run_rxprobe(args)
    else:
        run_rx(args)


if __name__ == "__main__":
    main()
