#!/usr/bin/env python3
"""Per-module timing for the QPSK + H.265 receive pipeline.

Goal: decide whether replacing the PyAV H.265 decoder with celux is worth it.
celux ONLY accelerates the video-decode step, so we measure every stage and
see what fraction H.265-decode actually is.

Receive loop (run_rxvideo / _rx_decode_worker), per capture:
  samples = downsample(capture)            # once per capture
  for st in detect_packets(samples):       # once per capture (whole buffer)
      decode_packet(samples, st)           # once per detected packet
  ...reassemble...; h264_dec.decode(full)  # once per completed frame

No SDR hardware, no GUI. Run: python _profile_modules.py
"""
import time
import numpy as np
import usrp_ofdm_video as U

U.set_modulation("qpsk")
U._AV_CODEC = "hevc"

RF_BITS = 18                      # rx_buffer = 2**18 RF samples (default)
RESES = [(320, 240), (640, 480), (1280, 720)]


def timeit(fn, n, *a, **k):
    fn(*a, **k)                   # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a, **k)
    return (time.perf_counter() - t0) / n * 1e3   # ms/call


def make_frame(w, h, seed):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = (128 + 80 * np.sin(seed / 3.0)).astype(np.uint8)
    x0 = 20 + (seed * 17) % (w - 80)
    img[h // 3:2 * h // 3, x0:x0 + 60] = (0, 0, 255)
    return img


def encode_frames(w, h, n=12):
    """Encode n frames, return list of non-empty per-frame bitstreams + enc ms."""
    enc = U._H264Encoder(w, h, bitrate=1_500_000, gop=1, preset="fast")
    outs = []
    t0 = time.perf_counter()
    for i in range(n):
        b = enc.encode(make_frame(w, h, i))
        if b:
            outs.append(b)
    enc_ms = (time.perf_counter() - t0) / n * 1e3
    return outs, enc_ms


def build_capture_buffer(frame_bytes):
    """One frame's packets laid into a downsampled-capture-sized BB buffer."""
    chunks = U.split_payload(frame_bytes)
    parts = []
    silence = np.zeros(U.NSYM * 4, dtype=np.complex64)
    for pid, ch in enumerate(chunks):
        wf, _ = U.build_packet_waveform(ch, 0, pid, len(chunks))
        parts.append(wf)
        parts.append(silence)
    body = np.concatenate(parts).astype(np.complex64)
    # capture buffer post-downsample length ~ 2**RF_BITS * 2/3
    bb_len = int((2 ** RF_BITS) * 2 / 3)
    if len(body) < bb_len:
        body = np.concatenate([body, np.zeros(bb_len - len(body),
                                              dtype=np.complex64)])
    body = U.awgn(body, 25.0)     # clean channel: measure work, not retries
    return body, len(chunks)


def main():
    print(f"[profile] mod={U._MOD_NAME} codec=hevc  numba={U.HAS_NUMBA} "
          f"pyfftw={U.HAS_PYFFTW}  rx_bits={RF_BITS}")

    # ---- video codec cost vs resolution -----------------------------------
    print("\n=== 视频编解码 (H.265 / PyAV) ===")
    print(f"{'分辨率':>10} {'帧字节':>8} {'编码ms':>8} {'解码ms':>8}")
    codec_rows = {}
    for (w, h) in RESES:
        outs, enc_ms = encode_frames(w, h, n=12)
        if not outs:
            print(f"{w}x{h:<6} (no encoder output)")
            continue
        dec = U._H264Decoder()
        # prime decoder with a couple frames, then time steady-state decode
        for b in outs[:2]:
            dec.decode(b)
        steady = outs[2:] or outs
        t0 = time.perf_counter()
        for b in steady:
            dec.decode(b)
        dec_ms = (time.perf_counter() - t0) / len(steady) * 1e3
        sz = int(np.mean([len(b) for b in outs]))
        codec_rows[(w, h)] = (sz, enc_ms, dec_ms)
        print(f"{w}x{h:<6} {sz:>8d} {enc_ms:>8.2f} {dec_ms:>8.2f}")

    # ---- DSP cost (per capture / per packet) ------------------------------
    print("\n=== DSP (NumPy 回退路径) ===")
    # use the 640x480 frame as a representative payload
    rep_w, rep_h = 640, 480
    outs, _ = encode_frames(rep_w, rep_h, n=12)
    frame_bytes = outs[len(outs) // 2] if outs else b"\x00" * 20000
    buf, n_pkt = build_capture_buffer(frame_bytes)

    rf = np.zeros(2 ** RF_BITS, dtype=np.complex64)   # raw RF-sized buffer
    ds_ms = timeit(U.downsample_to_bb, 20, rf)
    det_ms = timeit(U.detect_packets, 20, buf)

    starts = U.detect_packets(buf)
    # time decode_packet averaged over detected starts
    def _dec_all():
        for st in starts:
            U.decode_packet(buf, st)
    n_starts = max(len(starts), 1)
    decall_ms = timeit(_dec_all, 20)
    dec1_ms = decall_ms / n_starts

    # isolated ofdm_demodulate (one packet's worth of symbols)
    h_dummy = np.ones(U.NFFT, dtype=np.complex64)
    one = U.build_packet_waveform(b"x" * 1200, 0, 0, 1)[0][U.LEN_PREAMBLE:]
    odm_ms = timeit(U.ofdm_demodulate, 50, one, h_dummy)

    print(f"downsample_to_bb  (每次capture): {ds_ms:8.2f} ms  (2^{RF_BITS} RF样本)")
    print(f"detect_packets    (每次capture): {det_ms:8.2f} ms  (BB缓冲~{len(buf)}样本)")
    print(f"decode_packet     (每个包)     : {dec1_ms:8.2f} ms  "
          f"(检出{len(starts)}个包, 该帧{n_pkt}包)")
    print(f"ofdm_demodulate   (每个包)     : {odm_ms:8.2f} ms")

    # ---- per-second budget at a target fps --------------------------------
    print("\n=== 每秒耗时预算 (示例: 640x480 @ 12 fps) ===")
    if (rep_w, rep_h) in codec_rows:
        sz, enc_ms, dec_ms = codec_rows[(rep_w, rep_h)]
        fps = 12
        # captures/sec: capture covers 2^RF_BITS RF samples @ RF_RATE
        cap_per_s = U.RF_RATE / (2 ** RF_BITS)
        pkts_per_frame = n_pkt
        pkts_per_s = pkts_per_frame * fps
        t_ds = ds_ms * cap_per_s
        t_det = det_ms * cap_per_s
        t_decpkt = dec1_ms * pkts_per_s
        t_h265 = dec_ms * fps
        total = t_ds + t_det + t_decpkt + t_h265
        print(f"capture速率 ~{cap_per_s:.1f}/s, 每帧~{pkts_per_frame}包, "
              f"~{pkts_per_s:.0f}包/s")
        for name, v in [("downsample", t_ds), ("detect_packets", t_det),
                        ("decode_packet", t_decpkt), ("H.265 decode", t_h265)]:
            print(f"  {name:<16}: {v:7.1f} ms/s  ({100*v/total:4.1f}%)")
        print(f"  {'合计':<16}: {total:7.1f} ms/s  (单核占用 {total/10:.1f}%)")
        print()
        print(f"[结论] H.265 解码占 RX 计算的 {100*t_h265/total:.1f}%。")
        if t_h265 / total > 0.4:
            print("  -> H.265 解码是主要开销, celux/硬解有明显收益, 值得更换。")
        elif t_h265 / total > 0.15:
            print("  -> H.265 解码占比中等, celux 有一定收益但非决定性; "
                  "优先装 numba/pyfftw 提 DSP 更划算。")
        else:
            print("  -> H.265 解码占比很小, 瓶颈在 DSP(detect/decode_packet)。"
                  "换 celux 几乎无用, 应先装 numba+pyfftw 或并行化 DSP。")


if __name__ == "__main__":
    main()
