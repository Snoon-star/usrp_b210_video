#!/usr/bin/env python3
"""
ADALM-Pluto 单天线视频传输测试
- 读取视频文件,逐帧编码为 QPSK 符号
- 通过 PlutoSDR 发射
- 同步运行接收端(如果有第二个Pluto)或仅发射验证
- 单台 Pluto 收发回环验证模式 (loopback): TX/RX 口在同一台设备
  物理连接方式:
    1. 用 SMA 线 + 30dB 衰减器把 TX1A 接到 RX1A (推荐, 信号干净)
    2. 或两根天线分别插到 TX1A/RX1A, 近距离空中传输

依赖安装:
    pip install pyadi-iio numpy opencv-python

前置条件:
    - PlutoSDR 通过 USB 连接, 已安装驱动
    - 或通过网络连接: uri="ip:192.168.2.1"
"""

import time
import sys
import numpy as np
import cv2

try:
    import adi
except ImportError:
    print("请安装 pyadi-iio: pip install pyadi-iio")
    sys.exit(1)

# ============================================================
# 配置参数
# ============================================================
SAMPLE_RATE = 20e6          # 采样率 20 MHz
CENTER_FREQ = 2.4e9         # 中心频率 2.4 GHz (ISM频段)
TX_GAIN = 0                 # 发射增益 dB (最大 89.75,可变)
BANDWIDTH = 18e6            # RF带宽
FRAME_WIDTH = 64            # 每帧图像压缩宽度
FRAME_HEIGHT = 64           # 每帧图像压缩高度
FRAMES_PER_SECOND = 5       # 每秒传输帧数
SYNC_PREAMBLE_LEN = 256     # 同步前导码长度

# QPSK 星座图映射
QPSK_MAP = {
    (0, 0):  1 + 1j,
    (0, 1): -1 + 1j,
    (1, 0):  1 - 1j,
    (1, 1): -1 - 1j,
}

# ============================================================
# 工具函数
# ============================================================


def generate_sync_preamble(length):
    """生成 Zadoff-Chu 同步前导码"""
    u = 25
    n = np.arange(length)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / length)
    return zc.astype(np.complex64)


def bits_to_qpsk(bits):
    """比特流 -> QPSK 复数符号"""
    bits = np.array(bits).flatten()
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    symbols = np.zeros(len(bits) // 2, dtype=np.complex64)
    for i in range(0, len(bits), 2):
        key = (int(bits[i]), int(bits[i + 1]))
        symbols[i // 2] = QPSK_MAP[key]
    return symbols


def qpsk_to_bits(symbols):
    """QPSK 复数符号 -> 比特流 (接收端用)"""
    bits = []
    for s in symbols:
        re, im = np.real(s), np.imag(s)
        if re >= 0 and im >= 0:
            bits.extend([0, 0])
        elif re < 0 and im >= 0:
            bits.extend([0, 1])
        elif re >= 0 and im < 0:
            bits.extend([1, 0])
        else:
            bits.extend([1, 1])
    return np.array(bits)


def encode_frame_to_bits(frame):
    """
    将视频帧编码为比特流
    - 压缩到 FRAME_WIDTH x FRAME_HEIGHT
    - 转灰度
    - 直接取像素二值化
    返回: (bits, original_shape, mean_val)
    """
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary = (gray > 128).astype(np.uint8)
    bits = binary.flatten()
    return bits, gray.shape, np.mean(gray)


def decode_bits_to_frame(bits, shape):
    """比特流解码为视频帧"""
    total_pixels = shape[0] * shape[1]
    pixels = bits[:total_pixels].reshape(shape) * 255
    return pixels.astype(np.uint8)


def frame_header(frame_index, num_bits):
    """
    帧头部信息: [帧序号(16bit) | 比特数(24bit)]
    共 40 bit = 20 QPSK符号
    """
    header_bits = []
    for i in range(16):
        header_bits.append((frame_index >> (15 - i)) & 1)
    for i in range(24):
        header_bits.append((num_bits >> (23 - i)) & 1)
    return np.array(header_bits)


def parse_frame_header(bits):
    """解析帧头部"""
    frame_index = 0
    for i in range(16):
        frame_index = (frame_index << 1) | int(bits[i])
    num_bits = 0
    for i in range(16, 40):
        num_bits = (num_bits << 1) | int(bits[i])
    return frame_index, num_bits


def build_tx_burst(frame, frame_index):
    """
    构建一次发射突发:
    [前导码 | 帧头(QPSK) | 数据(QPSK)]
    """
    preamble = generate_sync_preamble(SYNC_PREAMBLE_LEN)

    bits, shape, _ = encode_frame_to_bits(frame)
    header_bits = frame_header(frame_index, len(bits))
    header_sym = bits_to_qpsk(header_bits)
    data_sym = bits_to_qpsk(bits)

    burst = np.concatenate([preamble, header_sym, data_sym])
    # 归一化功率
    burst = burst / np.sqrt(np.mean(np.abs(burst) ** 2))
    return burst.astype(np.complex64), bits, shape


# ============================================================
# 发射器类
# ============================================================


class PlutoVideoTX:
    """PlutoSDR 视频发射器"""

    def __init__(self, uri="ip:192.168.2.1", center_freq=CENTER_FREQ,
                 sample_rate=SAMPLE_RATE, tx_gain=TX_GAIN, bandwidth=BANDWIDTH):
        self.uri = uri
        print(f"[TX] 初始化 PlutoSDR ({uri}) ...")
        self.sdr = adi.Pluto(uri)
        self.sdr.tx_lo = int(center_freq)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.tx_rf_bandwidth = int(bandwidth)
        self.sdr.tx_hardwaregain_chan0 = tx_gain
        self.sdr.tx_cyclic_buffer = False
        print(f"[TX] 初始化完成  Freq={center_freq/1e6:.0f}MHz  "
              f"SR={sample_rate/1e6:.0f}MHz  BW={bandwidth/1e6:.0f}MHz  "
              f"Gain={tx_gain}dB")
        self.buffer_size = 2 ** 18

    def send_burst(self, burst):
        """发送一次突发数据"""
        burst_len = len(burst)
        if burst_len > self.buffer_size:
            raise ValueError(f"突发长度 {burst_len} 超过缓冲区 {self.buffer_size}")
        # 补零到缓冲区大小
        tx_data = np.zeros(self.buffer_size, dtype=np.complex64)
        tx_data[:burst_len] = burst
        self.sdr.tx(tx_data)

    def close(self):
        self.sdr.tx_destroy_buffer()
        print("[TX] PlutoSDR 已释放")


# ============================================================
# 接收器类 (单设备测试时可选)
# ============================================================


class PlutoVideoRX:
    """PlutoSDR 视频接收器 (需要第二个 Pluto)"""

    def __init__(self, uri="ip:192.168.2.1", center_freq=CENTER_FREQ,
                 sample_rate=SAMPLE_RATE, bandwidth=BANDWIDTH):
        self.uri = uri
        print(f"[RX] 初始化 PlutoSDR ({uri}) ...")
        self.sdr = adi.Pluto(uri)
        self.sdr.rx_lo = int(center_freq)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.rx_rf_bandwidth = int(bandwidth)
        self.sdr.rx_buffer_size = 2 ** 18
        self.sdr.gain_control_mode_chan0 = "slow_attack"
        print(f"[RX] 初始化完成  Freq={center_freq/1e6:.0f}MHz  "
              f"SR={sample_rate/1e6:.0f}MHz  BW={bandwidth/1e6:.0f}MHz")
        self.preamble_ref = generate_sync_preamble(SYNC_PREAMBLE_LEN)

    def detect_burst(self):
        """检测前导码并捕获一帧"""
        rx_data = self.sdr.rx()
        # 用互相关检测前导码
        corr = np.correlate(rx_data, self.preamble_ref, mode='valid')
        peak_idx = np.argmax(np.abs(corr))

        if np.abs(corr[peak_idx]) < 0.15 * len(self.preamble_ref):
            return None  # 未检测到有效信号

        # 从前导码结束位置提取帧头和载荷
        start = peak_idx + SYNC_PREAMBLE_LEN
        header_sym = rx_data[start:start + 20]
        num_data_sym = (len(rx_data) - start - 20)
        if num_data_sym <= 0:
            return None
        data_sym = rx_data[start + 20:start + 20 + num_data_sym]

        header_bits = qpsk_to_bits(header_sym)
        frame_idx, num_bits = parse_frame_header(header_bits)
        data_bits = qpsk_to_bits(data_sym[:num_bits // 2])

        return frame_idx, data_bits, num_bits

    def close(self):
        self.sdr.rx_destroy_buffer()
        print("[RX] PlutoSDR 已释放")


# ============================================================
# 单设备收发器 (TX/RX 在同一台 Pluto 上)
# ============================================================


class PlutoVideoTRX:
    """
    单台 PlutoSDR 同时收发: 用同一个 adi.Pluto 实例配置 TX 与 RX.
    PlutoSDR 是全双工 AD9363, TX/RX 各自独立 RFFE, 可同时开启.
    使用方式:
        - TX1A 用 SMA 线/衰减器接到 RX1A (有线回环)
        - 或两根天线插在 TX1A/RX1A 近距离空中传输
    """

    def __init__(self, uri="ip:192.168.2.1", center_freq=CENTER_FREQ,
                 sample_rate=SAMPLE_RATE, tx_gain=TX_GAIN, bandwidth=BANDWIDTH,
                 rx_buffer_size=2 ** 15):
        self.uri = uri
        print(f"[TRX] 初始化 PlutoSDR ({uri}) ...")
        self.sdr = adi.Pluto(uri)
        # 1) 先配置采样率 (影响 TX/RX 两条链路)
        self.sdr.sample_rate = int(sample_rate)
        # 2) TX 路配置
        self.sdr.tx_lo = int(center_freq)
        self.sdr.tx_rf_bandwidth = int(bandwidth)
        self.sdr.tx_hardwaregain_chan0 = tx_gain
        self.sdr.tx_cyclic_buffer = True   # 循环发射当前帧
        # 3) RX 路配置
        self.sdr.rx_lo = int(center_freq)
        self.sdr.rx_rf_bandwidth = int(bandwidth)
        self.sdr.rx_buffer_size = rx_buffer_size
        self.sdr.gain_control_mode_chan0 = "slow_attack"
        # 缓冲区参数
        self.tx_buffer_size = 2 ** 14      # 单次 TX 推入的总样本数
        self.rx_buffer_size = rx_buffer_size
        self.preamble_ref = generate_sync_preamble(SYNC_PREAMBLE_LEN)
        self._tx_pushed = False
        print(f"[TRX] 初始化完成  Freq={center_freq/1e6:.0f}MHz  "
              f"SR={sample_rate/1e6:.0f}MHz  BW={bandwidth/1e6:.0f}MHz  "
              f"TX Gain={tx_gain}dB")
        # 丢弃前几次 RX 让 AGC 稳定 (忽略可能的 None 返回值)
        for _ in range(3):
            try:
                self.sdr.rx()
            except Exception:
                pass

    def update_tx(self, burst):
        """更新循环 TX 缓冲为新的突发."""
        if len(burst) > self.tx_buffer_size:
            raise ValueError(f"突发长度 {len(burst)} 超过 TX 缓冲 "
                             f"{self.tx_buffer_size}")
        tx_data = np.zeros(self.tx_buffer_size, dtype=np.complex64)
        tx_data[:len(burst)] = burst
        if self._tx_pushed:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
        self.sdr.tx(tx_data)
        self._tx_pushed = True

    def receive_and_decode(self, expected_data_sym):
        """
        采集一个 RX buffer, 用前导码相关检测并解调.
        expected_data_sym: 期望的数据符号数 (用于截取数据段).
        返回: (frame_idx, data_bits, num_bits, snr_db) 或 None
        """
        try:
            rx_data = self.sdr.rx()
        except Exception:
            return None
        if rx_data is None or len(rx_data) < SYNC_PREAMBLE_LEN + 20:
            return None
        # 互相关定位前导码
        corr = np.correlate(rx_data, self.preamble_ref, mode='valid')
        abs_corr = np.abs(corr)
        peak_idx = int(np.argmax(abs_corr))
        peak_val = abs_corr[peak_idx]

        # 检测门限: 相关峰高于均值的若干倍
        noise_floor = np.median(abs_corr)
        if peak_val < max(0.1 * len(self.preamble_ref), 5 * noise_floor):
            return None

        # 用相关峰相位校正后续符号 (补偿信道相位旋转)
        phase = np.angle(corr[peak_idx])

        burst_len = SYNC_PREAMBLE_LEN + 20 + expected_data_sym
        start = peak_idx + SYNC_PREAMBLE_LEN
        if start + 20 + expected_data_sym > len(rx_data):
            return None

        header_sym = rx_data[start:start + 20] * np.exp(-1j * phase)
        data_sym = (rx_data[start + 20:start + 20 + expected_data_sym]
                    * np.exp(-1j * phase))

        header_bits = qpsk_to_bits(header_sym)
        frame_idx, num_bits = parse_frame_header(header_bits)
        # 防御性: 帧号过大或比特数异常时丢弃
        if num_bits <= 0 or num_bits > expected_data_sym * 2:
            return None
        data_bits = qpsk_to_bits(data_sym[:num_bits // 2])

        # 估算 SNR: 信号功率/噪声功率
        signal_power = np.mean(np.abs(data_sym) ** 2)
        # 用前导码理想形状反推噪声
        rx_pre = rx_data[peak_idx:peak_idx + SYNC_PREAMBLE_LEN]
        scale = np.vdot(self.preamble_ref, rx_pre) / SYNC_PREAMBLE_LEN
        noise = rx_pre - scale * self.preamble_ref
        noise_power = np.mean(np.abs(noise) ** 2) + 1e-12
        snr_db = 10 * np.log10(signal_power / noise_power)

        return frame_idx, data_bits, num_bits, snr_db

    def close(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        print("[TRX] PlutoSDR 已释放")


# ============================================================
# 视频源
# ============================================================


def test_with_camera(tx, duration=30):
    """
    使用摄像头实时采集并发射 (单PlutoSDR发射测试)
    在显示屏上同步展示正在发送的画面
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头, 切换到测试图案模式")
        cap.release()
        return test_with_pattern(tx, duration)

    print(f"\n[测试] 摄像头实时传输, 持续 {duration} 秒")
    print(f"[测试] 帧率={FRAMES_PER_SECOND} fps, 分辨率={FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[测试] 比特率≈{FRAME_WIDTH * FRAME_HEIGHT * FRAMES_PER_SECOND / 1000:.0f} kbps")
    print("-" * 50)

    frame_idx = 0
    frame_interval = 1.0 / FRAMES_PER_SECOND
    t_start = time.time()
    tx_count = 0

    cv2.namedWindow("TX Monitor", cv2.WINDOW_NORMAL)

    while time.time() - t_start < duration:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # 构建并发射
        burst, bits, shape = build_tx_burst(frame, frame_idx)
        tx.send_burst(burst)
        tx_count += 1

        # 显示压缩后要发送的画面
        display_frame = cv2.resize(
            decode_bits_to_frame(bits, shape),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("TX Monitor", display_frame)

        elapsed = time.time() - t_start
        fps_actual = tx_count / elapsed if elapsed > 0 else 0
        print(f"\r[TX] 帧 #{frame_idx:04d}  |  已发送 {tx_count} 帧  |  "
              f"实际 {fps_actual:.1f} fps  |  已运行 {elapsed:.0f}s", end="")

        frame_idx += 1

        # 维持帧率
        sleep_time = frame_interval - (time.time() - loop_start)
        if sleep_time > 0:
            key = cv2.waitKey(int(sleep_time * 1000))
        else:
            key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"\n[汇总] 发射 {tx_count} 帧, 用时 {elapsed:.1f}s, "
          f"平均 {tx_count/elapsed:.1f} fps")


def test_with_video_file(tx, video_path, duration=30):
    """从视频文件读取并发射"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_idx = 0
    frame_interval = 1.0 / FRAMES_PER_SECOND
    t_start = time.time()
    tx_count = 0

    cv2.namedWindow("TX Monitor", cv2.WINDOW_NORMAL)

    while time.time() - t_start < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
            continue

        burst, bits, shape = build_tx_burst(frame, frame_idx)
        tx.send_burst(burst)
        tx_count += 1

        display = cv2.resize(
            decode_bits_to_frame(bits, shape),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("TX Monitor", display)
        cv2.waitKey(int(frame_interval * 1000))
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"[汇总] 发射 {tx_count} 帧, 用时 {elapsed:.1f}s, "
          f"平均 {tx_count/elapsed:.1f} fps")


def test_with_pattern(tx, duration=30):
    """使用测试图案(移动条纹)发射"""
    print(f"\n[测试] 测试图案传输, 持续 {duration} 秒")
    print("-" * 50)

    frame_idx = 0
    frame_interval = 1.0 / FRAMES_PER_SECOND
    t_start = time.time()
    tx_count = 0

    cv2.namedWindow("TX Monitor", cv2.WINDOW_NORMAL)

    while time.time() - t_start < duration:
        # 生成移动条纹测试图案
        pattern = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            color = ((y + frame_idx * 8) % 255)
            cv2.line(pattern, (0, y), (640, y),
                     (color, 255 - color, (color + 128) % 255), 1)

        # 添加帧号文字
        cv2.putText(pattern, f"Frame: {frame_idx:04d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        burst, bits, shape = build_tx_burst(pattern, frame_idx)
        tx.send_burst(burst)
        tx_count += 1

        display = cv2.resize(
            decode_bits_to_frame(bits, shape),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("TX Monitor", display)

        key = cv2.waitKey(int(frame_interval * 1000))
        if key == 27:
            break

        frame_idx += 1

    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"[汇总] 发射 {tx_count} 帧, 用时 {elapsed:.1f}s, "
          f"平均 {tx_count/elapsed:.1f} fps")


def test_without_pluto(duration=10):
    """
    无 PlutoSDR 硬件时的演示模式
    自动尝试使用同目录下的 movie.mp4，无则使用测试图案
    展示完整的编码->调制->信道仿真->解调->解码流程
    """
    print("\n[演示] 无硬件仿真模式 (编码->AWGN信道->解码)")
    print("-" * 50)

    # 尝试使用同目录下的视频文件
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "movie.mp4")
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        print(f"[信息] 已加载视频: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        print(f"[信息] 视频: {total_frames} 帧, {fps_video:.1f} fps")
    else:
        cap.release()
        cap = None
        print("[信息] 未检测到 movie.mp4，使用测试图案")

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Received (SNR=10dB)", cv2.WINDOW_NORMAL)

    frame_idx = 0
    frame_interval = 1.0 / FRAMES_PER_SECOND
    t_start = time.time()
    total_ber = 0
    frame_count = 0

    while time.time() - t_start < duration:
        # 生成或采集一帧
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕则循环，摄像头失败则切换到测试图案
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    continue
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for y in range(480):
                color = ((y + frame_idx * 8) % 255)
                cv2.line(frame, (0, y), (640, y),
                         (color, 255 - color, (color + 128) % 255), 1)
            cv2.putText(frame, f"Frame: {frame_idx:04d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 编码
        original_bits, shape, _ = encode_frame_to_bits(frame)

        # QPSK 调制
        symbols = bits_to_qpsk(original_bits)

        # AWGN信道仿真 (10dB SNR)
        snr_db = 10
        signal_power = np.mean(np.abs(symbols) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
        )
        rx_symbols = symbols + noise.astype(np.complex64)

        # 解调
        rx_bits = qpsk_to_bits(rx_symbols)

        # 计算BER
        min_len = min(len(original_bits), len(rx_bits))
        ber = np.mean(original_bits[:min_len] != rx_bits[:min_len])
        total_ber += ber
        frame_count += 1

        # 解码
        rx_frame = decode_bits_to_frame(rx_bits, shape)

        cv2.imshow("Original", cv2.resize(frame, (256, 256)))
        cv2.imshow(f"Received (SNR={snr_db}dB)",
                   cv2.resize(rx_frame, (256, 256),
                              interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(int(frame_interval * 1000))
        if key == 27:
            break
        frame_idx += 1

    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    avg_ber = total_ber / frame_count if frame_count > 0 else 0
    print(f"[汇总] 仿真 {frame_count} 帧, 用时 {elapsed:.1f}s, "
          f"平均 BER={avg_ber:.2e} ({10*np.log10(0.5/avg_ber) if avg_ber > 0 else '∞'} dB 等效)")


# ============================================================
# 单设备回环测试
# ============================================================


def test_loopback(duration=30, source="pattern", video_path="",
                  uri="ip:192.168.2.1", center_freq=CENTER_FREQ,
                  sample_rate=SAMPLE_RATE, tx_gain=TX_GAIN,
                  bandwidth=BANDWIDTH, frame_interval_s=None):
    """
    单台 PlutoSDR 收发回环:
    TX 发射视频帧 -> RX 接收并解码显示.
    物理连接: TX1A (SMA) -> 30dB 衰减器 -> RX1A
    """
    from fractions import Fraction

    if frame_interval_s is None:
        frame_interval_s = 1.0 / FRAMES_PER_SECOND

    trx = PlutoVideoTRX(uri=uri, center_freq=center_freq,
                        sample_rate=sample_rate, tx_gain=tx_gain,
                        bandwidth=bandwidth, rx_buffer_size=2 ** 15)

    print(f"\n[Loopback] 单台 Pluto 收发回环测试, 持续 {duration} 秒")
    print(f"[Loopback] 帧率={1/frame_interval_s:.1f} fps, "
          f"分辨率={FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[Loopback] 物理连接: TX1A -> 30dB 衰减器 -> RX1A (或天线对连)")
    print("-" * 50)

    # 准备视频源
    cap = None
    if source == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("摄像头不可用, 切换到测试图案")
            cap = None
    elif source == "file":
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}, 切换到测试图案")
            cap = None

    cv2.namedWindow("TX (原始)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RX (接收)", cv2.WINDOW_NORMAL)

    def get_frame(frame_idx):
        """获取一帧图像, 返回 (frame, shape_info)"""
        nonlocal cap
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                if source == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    cap = None
                    return get_frame(frame_idx)
            return frame, None
        # 测试图案
        pattern = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            color = ((y + frame_idx * 8) % 255)
            cv2.line(pattern, (0, y), (640, y),
                     (color, 255 - color, (color + 128) % 255), 1)
        cv2.putText(pattern, f"Frame: {frame_idx:04d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return pattern, None

    frame_idx = 0
    tx_count = 0
    rx_count = 0
    bad_frame = 0
    total_snr = 0.0
    t_start = time.time()

    try:
        while time.time() - t_start < duration:
            loop_t = time.time()

            # 1) 获取并编码帧
            try:
                frame, _ = get_frame(frame_idx)
                burst, bits, shape = build_tx_burst(frame, frame_idx)
            except Exception:
                frame_idx += 1
                continue
            expected_data_sym = (20 + len(bits) // 2)  # 头20sym + 数据sym

            # 2) TX: 更新循环缓冲
            try:
                trx.update_tx(burst)
                tx_count += 1
            except Exception as e:
                print(f"\n[ERR] TX 失败: {e}")
                frame_idx += 1
                bad_frame += 1
                time.sleep(0.1)
                continue

            # 3) 短等待让信号飞过去
            time.sleep(0.01)

            # 4) RX: 采集并解码
            try:
                result = trx.receive_and_decode(expected_data_sym)
            except Exception:
                result = None
            if result is not None:
                rx_idx, rx_bits, num_bits, snr_db = result
                if rx_idx == frame_idx:
                    rx_frame = decode_bits_to_frame(rx_bits, shape)
                    total_snr += snr_db
                    rx_count += 1
                    # 显示
                    cv2.imshow("RX (接收)",
                               cv2.resize(rx_frame, (256, 256),
                                          interpolation=cv2.INTER_NEAREST))
                else:
                    bad_frame += 1
            else:
                bad_frame += 1

            # 5) 显示 TX 画面
            tx_small = cv2.resize(frame, (256, 256))
            cv2.imshow("TX (原始)", tx_small)

            elapsed = time.time() - t_start
            avg_snr = total_snr / rx_count if rx_count > 0 else 0
            rx_rate = rx_count / elapsed if elapsed > 0 else 0
            print(f"\r[LOOP] TX帧#{frame_idx:04d} | TX={tx_count} "
                  f"RX={rx_count} 失败={bad_frame} "
                  f"RX速率={rx_rate:.1f}fps "
                  f"SNR={avg_snr:.1f}dB"
                  f"  已运行{elapsed:.0f}s", end="")

            frame_idx += 1

            # 保持帧率
            remain = frame_interval_s - (time.time() - loop_t)
            if remain > 0:
                key = cv2.waitKey(int(remain * 1000))
            else:
                key = cv2.waitKey(1)
            if key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        trx.close()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 50}")
    print(f"[结果] 单台 Pluto 收发回环测试完成:")
    print(f"       发射帧: {tx_count}")
    print(f"       正确接收: {rx_count}")
    print(f"       丢失/错误: {bad_frame}")
    print(f"       成功率: {rx_count/tx_count*100:.1f}%"
          if tx_count > 0 else "N/A")
    print(f"       平均 SNR: {total_snr/rx_count:.1f} dB"
          if rx_count > 0 else "N/A")
    print(f"       用时: {elapsed:.0f}s")


# ============================================================
# 主程序
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ADALM-Pluto 单天线视频传输测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python pluto_video_tx.py                          # 仿真模式(无硬件)
  python pluto_video_tx.py --mode camera             # PlutoSDR 实时摄像头传输
  python pluto_video_tx.py --mode pattern            # PlutoSDR 测试图案传输
  python pluto_video_tx.py --mode file --input a.mp4 # PlutoSDR 视频文件传输
  python pluto_video_tx.py --mode rx --rx-uri ip:192.168.2.1  # 接收端
  python pluto_video_tx.py --mode loopback                    # 单台Pluto收发回环
  python pluto_video_tx.py --mode loopback --source camera    # 回环+摄像头
  python pluto_video_tx.py --mode loopback --source file --input a.mp4  # 回环+视频文件
        """
    )
    parser.add_argument("--mode", default="sim",
                        choices=["sim", "camera", "pattern", "file", "rx", "loopback"],
                        help="运行模式 (默认: sim)")
    parser.add_argument("--uri", default="ip:192.168.2.1",
                        help="PlutoSDR URI")
    parser.add_argument("--rx-uri", default="ip:192.168.2.1",
                        help="接收端 PlutoSDR URI")
    parser.add_argument("--freq", type=float, default=CENTER_FREQ / 1e6,
                        help="中心频率 MHz (默认: 2400)")
    parser.add_argument("--gain", type=float, default=TX_GAIN,
                        help="发射增益 dB (默认: 0)")
    parser.add_argument("--duration", type=int, default=30,
                        help="测试持续时间秒 (默认: 30)")
    parser.add_argument("--input", default="",
                        help="输入视频文件路径")
    parser.add_argument("--fps", type=int, default=None,
                        help="每秒帧数 (默认: 5)")
    parser.add_argument("--width", type=int, default=None,
                        help="帧宽度 (默认: 64)")
    parser.add_argument("--height", type=int, default=None,
                        help="帧高度 (默认: 64)")
    parser.add_argument("--source", default="pattern",
                        choices=["pattern", "camera", "file"],
                        help="loopback 模式的视频源 (默认: pattern)")

    args = parser.parse_args()

    if args.fps is not None:
        globals()["FRAMES_PER_SECOND"] = args.fps
    if args.width is not None:
        globals()["FRAME_WIDTH"] = args.width
    if args.height is not None:
        globals()["FRAME_HEIGHT"] = args.height
    center_freq = args.freq * 1e6

    if args.mode == "sim":
        test_without_pluto(args.duration)

    elif args.mode == "rx":
        rx = PlutoVideoRX(uri=args.rx_uri, center_freq=center_freq)
        print(f"\n[RX] 等待接收... (按 Ctrl+C 停止)")
        frame_idx_last = -1
        try:
            t_start = time.time()
            while time.time() - t_start < args.duration:
                result = rx.detect_burst()
                if result is not None:
                    frame_idx, data_bits, num_bits = result
                    if frame_idx != frame_idx_last:
                        frame = decode_bits_to_frame(
                            data_bits,
                            (FRAME_HEIGHT, FRAME_WIDTH)
                        )
                        cv2.imshow("RX Video",
                                   cv2.resize(frame, (256, 256),
                                              interpolation=cv2.INTER_NEAREST))
                        print(f"\r[RX] 收到帧 #{frame_idx}  "
                              f"({num_bits} bits)", end="")
                        frame_idx_last = frame_idx
                cv2.waitKey(10)
        except KeyboardInterrupt:
            pass
        finally:
            rx.close()
            cv2.destroyAllWindows()

    elif args.mode in ("camera", "pattern", "file"):
        tx = PlutoVideoTX(uri=args.uri, center_freq=center_freq,
                          tx_gain=args.gain)
        try:
            if args.mode == "camera":
                test_with_camera(tx, args.duration)
            elif args.mode == "pattern":
                test_with_pattern(tx, args.duration)
            elif args.mode == "file":
                if not args.input:
                    print("请指定输入视频: --input <path>")
                else:
                    test_with_video_file(tx, args.input, args.duration)
        finally:
            tx.close()

    elif args.mode == "loopback":
        video_path = args.input if args.input else ""
        test_loopback(
            duration=args.duration,
            source=args.source,
            video_path=video_path,
            uri=args.uri,
            center_freq=center_freq,
            sample_rate=SAMPLE_RATE,
            tx_gain=args.gain,
            bandwidth=BANDWIDTH,
        )


if __name__ == "__main__":
    main()
