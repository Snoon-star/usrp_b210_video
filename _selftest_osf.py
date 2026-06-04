#!/usr/bin/env python3
"""Headless round-trip self-test for the OSF (oversampling) change.

Exercises the real TX->channel->RX path with NO SDR hardware and NO GUI:
  frame -> H.265 encode -> packetize -> upsample_to_rf -> AWGN
        -> downsample_to_bb -> detect_packets -> decode_packet -> reassemble
        -> H.265 decode -> verify a frame came back.

Runs at OSF=1.0 (new, no-resample) and OSF=1.5 (legacy) to confirm the new
identity-resample path is correct and the change didn't regress 1.5.

Run: conda run -n u220 --no-capture-output python _selftest_osf.py
"""
import numpy as np
import usrp_ofdm_video as U

U.set_modulation("qpsk")
U._AV_CODEC = "hevc"


def make_frame(w, h, seed):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = (128 + 80 * np.sin(seed / 3.0)).astype(np.uint8)
    x0 = 20 + (seed * 17) % (w - 80)
    img[h // 3:2 * h // 3, x0:x0 + 60] = (0, 0, 255)
    return img


def set_osf(osf):
    U.OSF = osf
    if abs(osf - 1.0) < 1e-6:
        U._RESAMPLE_UP = U._RESAMPLE_DOWN = 1
    else:
        U._RESAMPLE_UP, U._RESAMPLE_DOWN = 3, 2


def roundtrip_one(frame_bytes, snr_db):
    """One frame's packets -> upsample -> AWGN -> downsample -> decode.
    Returns (recovered_full_bytes_or_None, n_pkt, n_ok)."""
    chunks = U.split_payload(frame_bytes)
    n_pkt = len(chunks)
    silence = np.zeros(U.NSYM * 4, dtype=np.complex64)
    parts = []
    for pid, ch in enumerate(chunks):
        wf, _ = U.build_packet_waveform(ch, 7, pid, n_pkt)
        parts.append(wf)
        parts.append(silence)
    bb = np.concatenate(parts).astype(np.complex64)

    # TX path: upsample (identity at OSF=1) + peak-normalize like push()
    rf = U.upsample_to_rf(bb)
    peak = np.max(np.abs(rf))
    if peak > 0:
        rf = rf / peak * 0.7
    # channel
    rf = U.awgn(rf, snr_db)
    # RX path: downsample (identity at OSF=1) then detect/decode
    rx = U.downsample_to_bb(rf)

    buf = {}
    n_ok = 0
    for st in U.detect_packets(rx, threshold=0.55):
        res = U.decode_packet(rx, st)
        if res is None:
            continue
        fid, pkt_id, tot, payload, ok = res
        if ok and fid == 7 and pkt_id < n_pkt:
            n_ok += 1
            buf[pkt_id] = payload
    full = None
    if len(buf) == n_pkt:
        full = b"".join(buf[i] for i in range(n_pkt))
    return full, n_pkt, n_ok


def main():
    print(f"[selftest] numba={U.HAS_NUMBA} pyfftw={U.HAS_PYFFTW}")
    w, h = 640, 480
    for osf in (1.0, 1.5):
        set_osf(osf)
        enc = U._H264Encoder(w, h, bitrate=0, gop=1, preset="fast", crf=26)
        dec = U._H264Decoder()
        ok_frames = 0
        total = 0
        crc_pkts = 0
        all_pkts = 0
        for i in range(10):
            fb = enc.encode(make_frame(w, h, i))
            if not fb:
                continue
            total += 1
            full, n_pkt, n_ok = roundtrip_one(fb, snr_db=25.0)
            all_pkts += n_pkt
            crc_pkts += n_ok
            if full is not None:
                # exact payload round-trip check
                assert full == fb, f"payload mismatch at frame {i}"
                img = dec.decode(full)
                if img is not None and img.shape[0] == h and img.shape[1] == w:
                    ok_frames += 1
        print(f"  OSF={osf:g} resample=({U._RESAMPLE_UP}/{U._RESAMPLE_DOWN})  "
              f"frames_ok={ok_frames}/{total}  pkt_CRC={crc_pkts}/{all_pkts}")
        assert ok_frames >= total - 1, (
            f"OSF={osf}: only {ok_frames}/{total} frames recovered")
    print("[selftest] PASS - both OSF paths round-trip correctly")


if __name__ == "__main__":
    main()
