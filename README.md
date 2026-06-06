# usrp_b210_video

QPSK + H.265 OFDM color-video transmission over two USRP B210 (1 GHz, separate TX/RX hosts).

## Run (conda env: u220)

Verified working config (1280x640 H.265, ~stable):

```
RX: python usrp_ofdm_video.py --mode rxvideo --rx-uri "serial=U220213" --freq 1000 --rx-gain 30 --samp-rate 20 --osf 1.0 --codec h265 --rx-bits 18 --duration 60
TX: python usrp_ofdm_video.py --mode txvideo --tx-uri "serial=U220202" --freq 1000 --gain 55 --samp-rate 20 --osf 1.0 --codec h265 --h264-crf 28 --fwidth 1280 --fheight 640 --tx-fps 20 --preencode --duration 90 --tx-chan 1
```

`--tx-chan 1` makes **U220202** transmit on **RF B's `TX/RX` jack** instead of RF A's — its original RF A `TX/RX` connector is damaged. On a B210 the front-end channels are `0 = RF A` / `1 = RF B`, and only the `TX/RX` jack of each can transmit (the `RX2` jack is receive-only). `--rx-chan` does the same for the RX host; both default to `0` (RF A). Remember to physically move the antenna/cable to the **RF B `TX/RX`** SMA on U220202.

### 4120 MHz config

Same as above but at 4.12 GHz — higher path loss, so bump the gains (`--gain 65` TX, `--rx-gain 45` RX):

```
RX: python usrp_ofdm_video.py --mode rxvideo --rx-uri "serial=U220213" --freq 4120 --rx-gain 45 --samp-rate 20 --osf 1.0 --codec h265 --rx-bits 18 --duration 60
TX: python usrp_ofdm_video.py --mode txvideo --tx-uri "serial=U220202" --freq 4120 --gain 65 --samp-rate 20 --osf 1.0 --codec h265 --h264-crf 28 --fwidth 1280 --fheight 640 --tx-fps 20 --preencode --duration 90 --tx-chan 1
```

`--preencode` (file source only): encodes the whole clip once at startup, then the transmit loop only packetizes/modulates/pushes — so TX fps is no longer capped by encoder throughput (e.g. 720p H.265 ≈ 10 fps online). Big startup pause while it encodes, then it transmits the buffered frames in a loop. 
`--preencode-max` control perencode frame ,default = 2000. 
Raise `--samp-rate` (both ends) for more airtime headroom at high resolution.

### `--rx-bits` must fit inside the TX hold

The RX capture window (`2^rx-bits` RF samples) must be **shorter than one TX frame hold** (`1 / --tx-fps`) yet still span ≥1 cyclic packet period. If the window is *longer* than the hold, every capture straddles a frame boundary and the decoder keeps resetting its packet buffer → fps stalls. At `--samp-rate 20 --tx-fps 20` (50 ms hold, ~9 ms cyclic period), `--rx-bits 18` (≈13 ms window) is the sweet spot; `--rx-bits 19/20` (26/52 ms) straddle the hold and cap fps at ~7. Rule of thumb: window ≈ 1.5–2× the cyclic period, comfortably under the hold.

Both ends must match --mod (default qpsk) / --codec / --freq / --samp-rate / **--osf**. Needs uhd + numba + pyfftw (all in the u220 conda env).

### `--osf 1.0` (key fps fix)

`--osf 1.0` disables the 1.5x oversampling. The OFDM signal already has a ~19% guard band (12/64 null subcarriers, exactly like 802.11a), so no oversampling is needed. This removes `scipy.resample_poly`, which profiled as **~70% of RX compute (~20 ms per capture at rx-bits 19)** and ran inside the capture thread — every resample millisecond was a receive blind-gap. The air data rate is **identical** to `--osf 1.5` at the same `--samp-rate`; you just stop paying for the resample. Result: the radio captures airtime continuously, so RX decodes far more unique frames. `--tx-fps` can be raised accordingly (encode caps ~28 fps at 640x480; lower resolution or `--h264-preset veryfast` for more).

Legacy behavior is still available with `--osf 1.5` (must match on both ends).

## Auto power tuning (`usrp_power_tune.py`)

The best `--gain` (TX) / `--rx-gain` (RX) depend on the environment (frequency,
distance, antennas), so finding them by hand is tedious. `usrp_power_tune.py`
sweeps a grid of `(tx_gain, rx_gain)` combos, measures real link quality for
each, and prints the best combo by **true link PDR** plus a ready-to-run
`txvideo`/`rxvideo` command pair.

It reuses two existing diagnostic modes (no new RF code): `--mode txbeacon`
transmits a fixed test burst at a given TX gain and periodically reports how many
packets it has sent (`txpkts=`), and `--mode rxprobe` reports `det / LTS / HDR /
CRC` stage counts plus its receive duty cycle (`duty=`). The tuner launches both
as subprocesses and, for each gain combo, samples the transmitted-packet delta
over the measurement window. (`txbeacon`/`rxprobe` now also honor
`--tx-chan`/`--rx-chan`, required for U220202's RF B port.)

Run it **inside the activated `u220` conda env** (workers inherit `sys.executable`):

```
conda activate u220
python usrp_power_tune.py --freq 1000            # auto grid for ~1 GHz
python usrp_power_tune.py --freq 4120            # auto grid for ~4.12 GHz
python usrp_power_tune.py --tx-gains 60:70:2 --rx-gains 35,40,45 --dwell 8
python usrp_power_tune.py --dry-run              # print the commands only, no radio
```

Defaults: TX=`serial=U220202` on `--tx-chan 1` (RF B, since U220202's RF A is
damaged), RX=`serial=U220213` on `--rx-chan 0`, `--samp-rate 20 --osf 1.0
--mod qpsk`. Gain lists accept `a,b,c` or `start:stop:step`; results are written
to `power_tune_results.csv`. `--rank` chooses the winning metric:

- **`pdr` (default)** — `CRC / (Δtx_packets × rx_duty)` = true per-packet link
  success. Dividing by the TX-measured packet count removes transmit-FPS
  fluctuation; dividing by the measured RX duty cycle removes receive blind-time
  (which also compensates the heavier decode load at high gain). This is the
  cleanest metric for *gain* tuning, since gain only changes per-packet success,
  not transmit rate. Combos below an opportunity floor are excluded so a
  tiny-sample fluke can't win.
- `crc` — most CRC-passing packets per `--dwell` = throughput (mixes in TX-FPS, so
  noisier for gain tuning, but matches the real time-bound video link).
- `crclts` — highest `crc/lts` (partial PDR: denominator is RX-detected preambles,
  so it over-credits weak gains). `balanced` — `crc × crc/lts`.

Each combo is sampled over `--repeat` windows (default 3) and scored by the
**median** of its per-window PDRs, so a single window hit by transient
interference (or a lucky one) can't decide the winner. The table and CSV report
the window count (`n`) and the per-combo PDR `min–max` spread; `--repeat 1`
restores the old single-window behavior (and roughly thirds the total sweep time).

`--role both` (default) assumes both B210s are on one host (like `--mode dual`) and
is required for `pdr` (it needs the local beacon's `txpkts`); two-host setups use
`--role tx` on the TX host and `--role rx` on the RX host (which falls back to
`crc`, since it can't see the TX count) with the same `--dwell`.
