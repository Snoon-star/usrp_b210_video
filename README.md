# usrp_b210_video

QPSK + H.265 OFDM color-video transmission over two USRP B210 (1 GHz, separate TX/RX hosts).

## Run (conda env: u220)

Verified working config (1280x640 H.265, ~stable):

```
RX: python usrp_ofdm_video.py --mode rxvideo --rx-uri "serial=U220213" --freq 1000 --rx-gain 30 --samp-rate 20 --osf 1.0 --codec h265 --rx-bits 18 --duration 60
TX: python usrp_ofdm_video.py --mode txvideo --tx-uri "serial=U220202" --freq 1000 --gain 55 --samp-rate 20 --osf 1.0 --codec h265 --h264-crf 28 --fwidth 1280 --fheight 640 --tx-fps 20 --preencode --duration 90
```

`--preencode` (file source only): encodes the whole clip once at startup, then the transmit loop only packetizes/modulates/pushes — so TX fps is no longer capped by encoder throughput (e.g. 720p H.265 ≈ 10 fps online). Big startup pause while it encodes, then it transmits the buffered frames in a loop. Raise `--samp-rate` (both ends) for more airtime headroom at high resolution.

### `--rx-bits` must fit inside the TX hold

The RX capture window (`2^rx-bits` RF samples) must be **shorter than one TX frame hold** (`1 / --tx-fps`) yet still span ≥1 cyclic packet period. If the window is *longer* than the hold, every capture straddles a frame boundary and the decoder keeps resetting its packet buffer → fps stalls. At `--samp-rate 20 --tx-fps 20` (50 ms hold, ~9 ms cyclic period), `--rx-bits 18` (≈13 ms window) is the sweet spot; `--rx-bits 19/20` (26/52 ms) straddle the hold and cap fps at ~7. Rule of thumb: window ≈ 1.5–2× the cyclic period, comfortably under the hold.

Both ends must match --mod (default qpsk) / --codec / --freq / --samp-rate / **--osf**. Needs uhd + numba + pyfftw (all in the u220 conda env).

### `--osf 1.0` (key fps fix)

`--osf 1.0` disables the 1.5x oversampling. The OFDM signal already has a ~19% guard band (12/64 null subcarriers, exactly like 802.11a), so no oversampling is needed. This removes `scipy.resample_poly`, which profiled as **~70% of RX compute (~20 ms per capture at rx-bits 19)** and ran inside the capture thread — every resample millisecond was a receive blind-gap. The air data rate is **identical** to `--osf 1.5` at the same `--samp-rate`; you just stop paying for the resample. Result: the radio captures airtime continuously, so RX decodes far more unique frames. `--tx-fps` can be raised accordingly (encode caps ~28 fps at 640x480; lower resolution or `--h264-preset veryfast` for more).

Legacy behavior is still available with `--osf 1.5` (must match on both ends).
