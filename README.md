# usrp_b210_video

QPSK + H.265 OFDM color-video transmission over two USRP B210 (1 GHz, separate TX/RX hosts).

## Run (conda env: u220)

```
RX: python usrp_ofdm_video.py --mode rxvideo --rx-uri "serial=U220213" --freq 1000 --rx-gain 35 --samp-rate 8 --codec h265 --rx-bits 19
TX: python usrp_ofdm_video.py --mode txvideo --tx-uri "serial=U220202" --freq 1000 --gain 55 --samp-rate 8 --codec h265 --h264-crf 26 --fwidth 640 --fheight 480 --tx-fps 15
```

Both ends must match --mod (default qpsk) / --codec / --freq / --samp-rate. Needs uhd + numba + pyfftw (all in the u220 conda env).
