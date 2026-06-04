# usrp_b210_video

QPSK + H.265 OFDM color-video transmission over two USRP B210 (1 GHz, separate TX/RX hosts).

## Run (conda env: u220)

```
RX: python usrp_ofdm_video.py --mode rxvideo --rx-uri "serial=U220213" --freq 1000 --rx-gain 35 --samp-rate 8 --osf 1.0 --codec h265 --rx-bits 19
TX: python usrp_ofdm_video.py --mode txvideo --tx-uri "serial=U220202" --freq 1000 --gain 55 --samp-rate 8 --osf 1.0 --codec h265 --h264-crf 26 --fwidth 640 --fheight 480 --tx-fps 25 --preencode
```

`--preencode` (file source only): encodes the whole clip once at startup, then the transmit loop only packetizes/modulates/pushes — so TX fps is no longer capped by encoder throughput (e.g. 720p H.265 ≈ 10 fps online). Big startup pause while it encodes, then it transmits the buffered frames in a loop. Raise `--samp-rate` (both ends) for more airtime headroom at high resolution.

Both ends must match --mod (default qpsk) / --codec / --freq / --samp-rate / **--osf**. Needs uhd + numba + pyfftw (all in the u220 conda env).

### `--osf 1.0` (key fps fix)

`--osf 1.0` disables the 1.5x oversampling. The OFDM signal already has a ~19% guard band (12/64 null subcarriers, exactly like 802.11a), so no oversampling is needed. This removes `scipy.resample_poly`, which profiled as **~70% of RX compute (~20 ms per capture at rx-bits 19)** and ran inside the capture thread — every resample millisecond was a receive blind-gap. The air data rate is **identical** to `--osf 1.5` at the same `--samp-rate`; you just stop paying for the resample. Result: the radio captures airtime continuously, so RX decodes far more unique frames. `--tx-fps` can be raised accordingly (encode caps ~28 fps at 640x480; lower resolution or `--h264-preset veryfast` for more).

Legacy behavior is still available with `--osf 1.5` (must match on both ends).
