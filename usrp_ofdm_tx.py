#!/usr/bin/env python3
"""
USRP X310 OFDM 彩色视频发射端 (5.8 GHz)
===============================================
由 usrp_ofdm_video.py 拆出的纯 TX 版本, 与 usrp_ofdm_rx.py 配合在两台电脑
上分别运行收/发. 协议与 dual 模式 100% 一致:
  - L-STF / L-LTF 前导码
  - 64-FFT OFDM, 48 数据子载波 + 4 导频, CP=16
  - QPSK / 16QAM 调制
  - 每包帧头携带序号 + 长度 + CRC32
  - 30 MHz 空口采样 (20 * 1.5 过采样)

典型用法:
  python usrp_ofdm_tx.py --uri addr=192.168.10.2 \\
                         --freq 5800 --gain 10 \\
                         --codec h264 --h264-bitrate 1500000 \\
                         --fwidth 320 --fheight 240 --tx-fps 12

依赖: pip install uhd numpy scipy opencv-python av
"""

import argparse
import os
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
RF_RATE = int(SAMPLE_RATE_BB * OSF)        # 30 MHz
RF_BANDWIDTH = 18e6
TX_GAIN_DEFAULT = 10
USRP_DEFAULT_ADDR = "addr=192.168.10.2"

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

_QPSK_LUT = np.array([
    (1 + 1j), (1 - 1j), (-1 + 1j), (-1 - 1j),
], dtype=np.complex64) / np.sqrt(2)

_QAM16_LUT = np.array([
     3 + 3j,  3 + 1j,  3 - 3j,  3 - 1j,
     1 + 3j,  1 + 1j,  1 - 3j,  1 - 1j,
    -3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
    -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
], dtype=np.complex64) / np.sqrt(10)


# ============================================================================
# 比特 / 调制 (TX 只需 encode 侧)
# ============================================================================

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big").astype(np.int8)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)
    idx = (pairs[:, 0] << 1) | pairs[:, 1]
    return _QPSK_LUT[idx]


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    nibbles = bits.reshape(-1, 4)
    idx = ((nibbles[:, 0] << 3) | (nibbles[:, 1] << 2)
           | (nibbles[:, 2] << 1) | nibbles[:, 3])
    return _QAM16_LUT[idx]


_MOD_BITS_PER_SYM = 2
_MOD_FN = qpsk_modulate
_MOD_NAME = "qpsk"


def set_modulation(name: str):
    global _MOD_BITS_PER_SYM, _MOD_FN, _MOD_NAME
    name = name.lower()
    if name == "qpsk":
        _MOD_BITS_PER_SYM = 2
        _MOD_FN = qpsk_modulate
    elif name in ("16qam", "qam16"):
        _MOD_BITS_PER_SYM = 4
        _MOD_FN = qam16_modulate
        name = "16qam"
    else:
        raise ValueError(f"Unknown modulation: {name}")
    _MOD_NAME = name


# ============================================================================
# 前导码 (TX 侧时域波形)
# ============================================================================

def _build_freq_grid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    grid = np.zeros(NFFT, dtype=np.complex64)
    for k, v in zip(indices, values):
        grid[k % NFFT] = v
    return grid


def make_lstf_time() -> np.ndarray:
    grid = _build_freq_grid(LSTF_VAL, LSTF_INDEX)
    t = np.fft.ifft(grid) * NFFT / np.sqrt(12)
    short = t[:16]
    return np.tile(short, 10).astype(np.complex64)


def make_lltf_time() -> np.ndarray:
    indices = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    values = np.concatenate([LLTF_VAL_NEG, LLTF_VAL_POS]).astype(np.complex64)
    grid = _build_freq_grid(values, indices)
    t = np.fft.ifft(grid) * NFFT / np.sqrt(52)
    sym = t.astype(np.complex64)
    gi = sym[-32:]
    return np.concatenate([gi, sym, sym]).astype(np.complex64)


LSTF_TIME = make_lstf_time()
LLTF_TIME = make_lltf_time()
PREAMBLE = np.concatenate([LSTF_TIME, LLTF_TIME]).astype(np.complex64)


# ============================================================================
# OFDM 调制 + 包打包
# ============================================================================

def ofdm_modulate(data_symbols: np.ndarray) -> np.ndarray:
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


def build_packet_waveform(payload_bytes: bytes,
                          frame_id: int,
                          packet_id: int,
                          total_packets: int):
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

    syms = _MOD_FN(bits)
    data_time = ofdm_modulate(syms)
    return np.concatenate([PREAMBLE, data_time]).astype(np.complex64), 0


# ============================================================================
# 上采样 / 图像编码
# ============================================================================

def upsample_to_rf(x: np.ndarray) -> np.ndarray:
    return sps.resample_poly(x, up=3, down=2).astype(np.complex64)


JPEG_QUALITY = 75
MAX_PAYLOAD_PER_PKT = 1200


class _H264Encoder:
    def __init__(self, width: int, height: int, bitrate: int, gop: int,
                 preset: str = "fast"):
        try:
            import av
            from fractions import Fraction
        except ImportError:
            raise RuntimeError("H.264 需要 PyAV. 安装: pip install av")
        self._av = av
        self.width = (width // 2) * 2
        self.height = (height // 2) * 2
        self.codec = av.CodecContext.create("h264", "w")
        self.codec.width = self.width
        self.codec.height = self.height
        self.codec.pix_fmt = "yuv420p"
        self.codec.time_base = Fraction(1, 30)
        self.codec.bit_rate = int(bitrate)
        self.codec.gop_size = int(gop)
        self.codec.max_b_frames = 0
        self.codec.options = {
            "tune": "zerolatency",
            "preset": preset,
        }
        self._pts = 0

    def encode(self, bgr: np.ndarray) -> bytes:
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame = self._av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        frame.pts = self._pts
        self._pts += 1
        out = bytearray()
        for pkt in self.codec.encode(frame):
            out.extend(bytes(pkt))
        return bytes(out)


def frame_to_jpeg_bytes(frame: np.ndarray, width: int, height: int,
                        quality: int = JPEG_QUALITY,
                        codec: str = "jpg") -> bytes:
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    if codec == "webp":
        ok, buf = cv2.imencode(".webp", frame,
                               [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def split_payload(data: bytes, max_size: int = MAX_PAYLOAD_PER_PKT):
    return [data[i:i + max_size] for i in range(0, len(data), max_size)]


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
# USRP X310 TX 侧硬件接口
# ============================================================================

def _uhd_tune(freq: float):
    return uhd.libpyuhd.types.tune_request(float(freq))


def _make_stream_args():
    st = uhd.usrp.StreamArgs("fc32", "sc16")
    st.channels = [0]
    return st


class UsrpTX:
    """TX-only USRP X310. 通过 daemon 线程持续 send() 实现 cyclic 效果."""

    def __init__(self, uri: str, freq: float, tx_gain: float,
                 streaming: bool = False):
        if not HAS_USRP:
            raise RuntimeError("uhd 未安装")
        print(f"[TX]  连接 USRP {uri} ...")
        self.usrp = uhd.usrp.MultiUSRP(uri)
        self.usrp.set_tx_rate(float(RF_RATE), 0)
        self.usrp.set_tx_freq(_uhd_tune(freq), 0)
        self.usrp.set_tx_gain(float(tx_gain), 0)
        try:
            self.usrp.set_tx_bandwidth(float(RF_BANDWIDTH), 0)
        except Exception:
            pass
        try:
            self.usrp.set_tx_antenna("TX/RX", 0)
        except Exception:
            pass
        self.tx_streamer = self.usrp.get_tx_stream(_make_stream_args())
        self.streaming = streaming
        self._burst = None
        self._lock = threading.Lock()
        self._loop_thread = None
        self._loop_stop = threading.Event()
        mode = "streaming(one-shot)" if streaming else "cyclic(thread-emul)"
        print(f"[TX]  就绪  Freq={freq/1e9:.2f}GHz  SR={RF_RATE/1e6:.1f}MHz  "
              f"TX={tx_gain}dB  {mode}")

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

    def push_iq(self, tx_iq: np.ndarray):
        if self.streaming:
            md = uhd.types.TXMetadata()
            md.has_time_spec = False
            md.start_of_burst = True
            md.end_of_burst = True
            self.tx_streamer.send(tx_iq, md, 0.5)
        else:
            with self._lock:
                self._burst = tx_iq
            if self._loop_thread is None or not self._loop_thread.is_alive():
                self._loop_stop.clear()
                self._loop_thread = threading.Thread(
                    target=self._cyclic_loop, daemon=True)
                self._loop_thread.start()

    def push(self, baseband: np.ndarray):
        rf = upsample_to_rf(baseband)
        peak = np.max(np.abs(rf))
        if peak > 0:
            rf = rf / peak * 0.7
        self.push_iq(rf.astype(np.complex64))

    def close(self):
        self._loop_stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1)


# ============================================================================
# 主线程帧生产 + 发射 worker
# ============================================================================

_BURST_LIMIT = 2 ** 18


class FrameProducer:
    """主线程: 取一帧 -> 压缩 -> 切包 -> 拼成空口 burst -> 预 upsample."""

    def __init__(self, args, cap):
        self.args = args
        self.cap = cap
        self.idx = 0
        self.h264_enc = None
        if args.codec == "h264":
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
                return (frame, None, self.idx, 0, 0)
        else:
            jpg = frame_to_jpeg_bytes(frame, self.args.fwidth,
                                      self.args.fheight, self.args.jpeg,
                                      codec=self.args.codec)
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

        rf = upsample_to_rf(burst)
        peak = float(np.max(np.abs(rf)))
        if peak > 0:
            rf = rf / peak * 0.7
        burst = (rf * (2 ** 14)).astype(np.complex64)

        out = (frame, burst, self.idx, len(chunks), len(jpg))
        self.idx += 1
        return out


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


def _tx_push_worker(push_fn, tx_q: queue.Queue,
                    stop_event: threading.Event,
                    stats: dict,
                    min_period: float = 0.0):
    """专职发射线程: 阻塞拿 burst, drain 到最新, push.

    min_period: 两次 push 之间最短间隔, 保证 cyclic 有稳态时间给 RX 解包.
    """
    last_push = 0.0
    while not stop_event.is_set():
        try:
            burst = tx_q.get(timeout=0.1)
        except queue.Empty:
            continue
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


# ============================================================================
# 主流程
# ============================================================================

def run_tx(args):
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    tx = UsrpTX(args.uri, args.freq * 1e6, args.gain, streaming=False)

    cap = open_video_source(args.source, args.input)
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)

    tx_q: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    tx_stats: dict = {}

    tx_min_period = 1.0 / args.tx_fps if args.tx_fps > 0 else 0.0
    print(f"[TX]  节流: {args.tx_fps:.1f} fps "
          f"(min_period={tx_min_period*1000:.0f}ms)")
    tx_thread = threading.Thread(
        target=_tx_push_worker,
        args=(tx.push_iq, tx_q, stop_event, tx_stats, tx_min_period),
        daemon=True)
    tx_thread.start()

    producer = FrameProducer(args, cap)
    tx_submit = 0
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
                cv2.imshow("TX", cv2.resize(frame, (480, 360),
                                             interpolation=cv2.INTER_CUBIC))
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.005)
                continue

            if _replace_latest(tx_q, burst):
                tx_submit += 1

            cv2.imshow("TX", cv2.resize(frame, (480, 360)))
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
                print(f"\r[TX] fid#{fid} pkts={n_pkt} JPEG={jpg_len}B  "
                      f"TX_air={tx_air_fps:.1f}fps  submit={tx_submit}  "
                      f"{elapsed:.1f}s", end="")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        tx_thread.join(timeout=2)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        tx.close()

    elapsed = time.time() - t0
    pushed = tx_stats.get("pushed", 0)
    print(f"\n[结果] TX: 主线程提交 {tx_submit} 帧, 实际空口发出 "
          f"{pushed} 帧 ({pushed/max(elapsed,1e-3):.1f}fps)")


def run_txbeacon(args):
    """TX 诊断: cyclic 发一个固定 192x144 彩色测试图, 与 rxprobe 配合验证链路."""
    if not HAS_USRP:
        print("[错误] 未安装 uhd")
        return
    tx = UsrpTX(args.uri, args.freq * 1e6, args.gain)
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


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="USRP X310 OFDM 视频发射端 (5.8 GHz, 802.11a 风格)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python usrp_ofdm_tx.py --uri {USRP_DEFAULT_ADDR} \\
                         --freq 5800 --gain 10 \\
                         --codec h264 --h264-bitrate 1500000 \\
                         --fwidth 320 --fheight 240 --tx-fps 12
  python usrp_ofdm_tx.py --mode txbeacon --gain 10                # 链路诊断
""")
    p.add_argument("--mode", default="tx", choices=["tx", "txbeacon"],
                   help="tx=主流程; txbeacon=cyclic 发固定测试图样")
    p.add_argument("--source", default="file",
                   choices=["file", "camera", "pattern"])
    p.add_argument("--input", default="", help="视频文件路径")
    p.add_argument("--uri", default=USRP_DEFAULT_ADDR,
                   help=f"TX USRP UHD 地址, 默认 {USRP_DEFAULT_ADDR}")
    p.add_argument("--freq", type=float, default=CARRIER_FREQ_DEFAULT / 1e6,
                   help="中心频率 MHz")
    p.add_argument("--gain", type=float, default=TX_GAIN_DEFAULT,
                   help="TX 增益 dB")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--fwidth", type=int, default=320, help="视频帧宽度")
    p.add_argument("--fheight", type=int, default=240, help="视频帧高度")
    p.add_argument("--jpeg", type=int, default=JPEG_QUALITY,
                   help="编码质量 1-100 (JPEG/WebP 共用)")
    p.add_argument("--codec", default="jpg",
                   choices=["jpg", "webp", "h264"],
                   help="帧编码: jpg / webp / h264")
    p.add_argument("--h264-bitrate", type=int, default=500_000,
                   help="H.264 比特率 bit/s")
    p.add_argument("--h264-gop", type=int, default=1,
                   help="H.264 GOP. 默认 1 = intra-only, 抗丢帧最强")
    p.add_argument("--h264-preset", default="fast",
                   choices=["ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow"])
    p.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam"],
                   help="子载波调制, 必须与 RX 端一致")
    p.add_argument("--tx-fps", type=float, default=12.0,
                   help="TX 节流上限 fps")

    args = p.parse_args()
    set_modulation(args.mod)
    print(f"[CFG] modulation={_MOD_NAME}  bits/sym={_MOD_BITS_PER_SYM}")

    if args.mode == "txbeacon":
        run_txbeacon(args)
    else:
        run_tx(args)


if __name__ == "__main__":
    main()
