#!/usr/bin/env python3
"""
USRP B210 收发功率自动调试 (U220202 TX / U220213 RX)
=====================================================
不同环境(频率/距离/天线)下, TX --gain 和 RX --rx-gain 的最佳点都不一样, 手动
一格一格试很麻烦. 本脚本把它自动化: 扫一遍 (tx_gain, rx_gain) 网格, 每个组合
用 usrp_ofdm_video.py 的两个诊断模式实测链路质量, 最后给出 CRC 通过最多的组合,
并直接打印可复制的 txvideo/rxvideo 命令.

原理 (复用现成的两个诊断模式, 不重写射频栈):
  - TX 侧: --mode txbeacon  在 tx_gain 上 cyclic 持续发一个固定测试 burst.
  - RX 侧: --mode rxprobe    在 rx_gain 上收, 逐捕统计 det/LTS/HDR/CRC 四级成功数,
            结束时打印 "[结果] ... 累计 det=.. LTS=.. HDR=.. CRC=..".
  本脚本把这两个模式按网格拉起来, 解析 rxprobe 的累计 CRC 作为该组合的得分.

四级指标含义 (来自 rxprobe):
  det = L-STF 自相关触发数(含噪声假阳性);  LTS = 找到合法 LTS 模板的点数;
  HDR = header magic 命中(OFDM 字节解对了);  CRC = 整包 CRC32 通过(真正成功).
  -> 排序主键是 CRC(每 dwell 收到的好包数), 次键 HDR; crc/lts 反映 BER 高低.

拓扑假设 (默认 --role both):
  两台 B210 接在同一台主机上(和 --mode dual 一样, 各自独立 USB context),
  本脚本在本机同时拉起 txbeacon + rxprobe 子进程来扫描. 若 TX/RX 在两台不同
  主机, 用 --role tx (只发, 可扫 tx_gain) 和 --role rx (只收并解析, 假设对端
  已在发) 分别在两台机器上跑, 用相同 --dwell 对齐时间.

硬件备注: U220202 的 RF A TX/RX 口已损坏, 故 --tx-chan 默认 1 (走 RF B 的
TX/RX 口); 记得把发射天线/馈线接到 U220202 的 **RF B TX/RX** SMA 上.
详见 memory/u220202-tx-port-damaged.md.

运行环境: 必须在已激活的 conda u220 环境里跑 (uhd/numba/pyfftw 都在那儿).
  conda activate u220
  python usrp_power_tune.py --freq 1000            # 1 GHz 自动扫
  python usrp_power_tune.py --freq 4120            # 4.12 GHz 自动扫
  python usrp_power_tune.py --tx-gains 60:70:2 --rx-gains 35,40,45 --dwell 8
  python usrp_power_tune.py --dry-run              # 只打印将要执行的命令

子进程用 sys.executable 拉起, 因此本脚本用哪个 python 跑, worker 就用哪个
(在 u220 环境里跑本脚本即可让 worker 也在 u220 环境).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time


# rxprobe 结束时的累计汇总行: "[结果] N 次 capture, 累计 det=.. LTS=.. HDR=.. CRC=.."
# 只锚 ASCII 的 det/LTS/HDR/CRC 字段, 但用 "累计" 把它和逐捕行(也含 det=..LTS=..)区分开.
_SUMMARY_RE = re.compile(
    r"累计\s*det=(\d+)\s+LTS=(\d+)\s+HDR=(\d+)\s+CRC=(\d+)")
# 兜底: 逐捕行尾的累计 CRC ("... cum CRC=NN ...")
_CUMCRC_RE = re.compile(r"cum CRC=(\d+)")
# 设备打开/解析失败的特征, 用于提示
_ERR_RE = re.compile(r"\[错误\]|Traceback|RuntimeError|LookupError|usb")


def _parse_gain_list(spec: str):
    """解析增益列表: "55,60,65" 或 "start:stop:step" (闭区间).

    step 模式生成 [start, start+step, ..., <=stop]. 返回去重升序的 float 列表.
    """
    spec = spec.strip()
    vals = []
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"增益区间格式应为 start:stop:step, 收到 {spec!r}")
        start, stop, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise ValueError("step 必须 > 0")
        v = start
        # 加 1e-6 容忍浮点误差, 让 stop 含进来
        while v <= stop + 1e-6:
            vals.append(round(v, 3))
            v += step
    else:
        for tok in spec.split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
    # 去重保序->升序
    seen = set()
    out = []
    for v in sorted(vals):
        if v not in seen:
            seen.add(v)
            out.append(v)
    if not out:
        raise ValueError(f"空增益列表: {spec!r}")
    return out


def _default_gains(freq_mhz: float):
    """据频率给一个合理的初扫网格 (路损越高, 增益基准越高). B210 量程内安全值."""
    if freq_mhz < 2000:          # ~1 GHz 近场
        return [50, 55, 60, 65], [25, 30, 35, 40, 45]
    elif freq_mhz < 3500:
        return [55, 60, 65, 70], [30, 35, 40, 45, 50]
    else:                        # ~4 GHz+ 高路损
        return [60, 65, 70, 75], [35, 40, 45, 50, 55]


def _common_rf_args(args):
    """收发两端都必须一致的射频/OFDM 参数."""
    return ["--freq", str(args.freq),
            "--samp-rate", str(args.samp_rate),
            "--osf", str(args.osf),
            "--mod", args.mod]


def _beacon_cmd(args, tx_gain, duration):
    return [sys.executable, args.script, "--mode", "txbeacon",
            "--tx-uri", args.tx_uri,
            "--gain", str(tx_gain),
            "--tx-chan", str(args.tx_chan),
            "--duration", str(duration)] + _common_rf_args(args)


def _probe_cmd(args, rx_gain, duration):
    return [sys.executable, args.script, "--mode", "rxprobe",
            "--rx-uri", args.rx_uri,
            "--rx-gain", str(rx_gain),
            "--rx-chan", str(args.rx_chan),
            "--duration", str(duration)] + _common_rf_args(args)


def _child_env():
    """强制 worker 用 UTF-8 写 stdout, 保证含中文的 '[结果]/累计' 行能被解析."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


class _Drainer:
    """后台读子进程 stdout, 防止管道写满阻塞; 记录最近若干行供诊断,
    并标记是否见到设备/解析错误."""

    def __init__(self, proc, keep=60):
        self.proc = proc
        self.lines = []
        self.keep = keep
        self.saw_error = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        for raw in iter(self.proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            if _ERR_RE.search(line):
                self.saw_error = True
            self.lines.append(line)
            if len(self.lines) > self.keep:
                self.lines = self.lines[-self.keep:]

    def join(self, timeout=2.0):
        self._t.join(timeout)


def _spawn(cmd):
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", env=_child_env(),
        bufsize=1)


def _stop(proc, timeout=5.0):
    """停掉子进程 (TerminateProcess); 退出即释放 USB 设备, 下个组合可重开."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=timeout)
        except Exception:
            pass


def _parse_probe_output(text: str):
    """从 rxprobe 全部 stdout 里抽出累计 (cap, det, lts, hdr, crc).

    优先用 '累计' 汇总行; 没有则退回最后一个 'cum CRC=' (只给 crc)."""
    m = None
    for m in _SUMMARY_RE.finditer(text):
        pass                     # 取最后一个匹配
    if m is not None:
        det, lts, hdr, crc = (int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4)))
        capm = re.search(r"\[结果\]\s*(\d+)\s*次", text)
        cap = int(capm.group(1)) if capm else 0
        return cap, det, lts, hdr, crc, True
    # 兜底
    last = None
    for last in _CUMCRC_RE.finditer(text):
        pass
    if last is not None:
        return 0, 0, 0, 0, int(last.group(1)), True
    return 0, 0, 0, 0, 0, False


def run_probe_once(args, rx_gain):
    """跑一次 rxprobe (dwell 秒, 自然结束), 返回解析后的指标 dict."""
    cmd = _probe_cmd(args, rx_gain, args.dwell)
    t0 = time.time()
    try:
        proc = _spawn(cmd)
    except FileNotFoundError:
        print(f"\n[错误] 找不到 python 或脚本: {cmd[0]} {args.script}")
        return None
    try:
        # 设备 open(~1-2s 暖机) + dwell 测量 + 余量
        out, _ = proc.communicate(timeout=args.dwell + 40)
    except subprocess.TimeoutExpired:
        _stop(proc)
        out, _ = proc.communicate()
    cap, det, lts, hdr, crc, ok = _parse_probe_output(out or "")
    crc_lts = (100.0 * crc / lts) if lts else 0.0
    return {
        "tx_gain": None, "rx_gain": rx_gain,
        "cap": cap, "det": det, "lts": lts, "hdr": hdr, "crc": crc,
        "crc_lts_pct": crc_lts, "parsed": ok,
        "elapsed": time.time() - t0,
        "tail": (out or "").strip().splitlines()[-3:],
    }


def sweep_both(args, tx_gains, rx_gains):
    """单机扫描: 外层 tx_gain (每个开一次 beacon), 内层 rx_gain (每个跑一次 rxprobe)."""
    results = []
    total = len(tx_gains) * len(rx_gains)
    done = 0
    # beacon 至少要盖住内层一轮 rxprobe 的总时长
    per_probe = args.dwell + 40 + args.settle
    beacon_dur = args.warmup + len(rx_gains) * per_probe + 15

    print(f"[扫描] 共 {total} 组合  (tx_gains={tx_gains}  rx_gains={rx_gains})  "
          f"dwell={args.dwell}s")
    print(f"[扫描] TX={args.tx_uri} chan{args.tx_chan}  "
          f"RX={args.rx_uri} chan{args.rx_chan}  "
          f"freq={args.freq}MHz samp={args.samp_rate} osf={args.osf} mod={args.mod}")
    print("-" * 78)

    beacon = None
    try:
        for tx_gain in tx_gains:
            print(f"\n[TX gain={tx_gain}dB] 启动 beacon ...")
            beacon = _spawn(_beacon_cmd(args, tx_gain, beacon_dur))
            drain = _Drainer(beacon)
            # 等 beacon 暖机(设备 open + cyclic 线程起来)
            t_warm = time.time()
            while time.time() - t_warm < args.warmup:
                if beacon.poll() is not None:
                    break
                time.sleep(0.1)
            if beacon.poll() is not None or drain.saw_error:
                print(f"  [警告] beacon 似乎未正常发射 (tx_gain={tx_gain}). "
                      f"末尾输出:")
                for ln in drain.lines[-5:]:
                    print(f"    | {ln}")

            for rx_gain in rx_gains:
                done += 1
                print(f"  [{done:2d}/{total}] tx={tx_gain:<4g} rx={rx_gain:<4g} "
                      f"测量 {args.dwell}s ...", end="", flush=True)
                r = run_probe_once(args, rx_gain)
                if r is None:
                    _stop(beacon)
                    return results
                r["tx_gain"] = tx_gain
                results.append(r)
                flag = "" if r["parsed"] else "  [未解析到结果!]"
                print(f"\r  [{done:2d}/{total}] tx={tx_gain:<4g} rx={rx_gain:<4g} "
                      f"| cap={r['cap']:<4d} det={r['det']:<4d} LTS={r['lts']:<4d} "
                      f"HDR={r['hdr']:<3d} CRC={r['crc']:<3d} "
                      f"crc/lts={r['crc_lts_pct']:5.1f}%{flag}")
                if not r["parsed"] and r["tail"]:
                    for ln in r["tail"]:
                        print(f"        | {ln}")
                time.sleep(args.settle)

            _stop(beacon)
            beacon = None
            drain.join()
            time.sleep(args.settle + 0.8)   # 让 TX 设备彻底释放再换 gain
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C, 停止扫描.")
    finally:
        _stop(beacon)
    return results


def run_role_tx(args, tx_gains):
    """两机拓扑的 TX 侧: 依次在每个 tx_gain 上发 dwell 秒 (打印对齐用的横幅).

    在 RX 主机上同步用 --role rx 对齐 --dwell 即可人工对应每段."""
    print(f"[TX-only] 将依次发射: {tx_gains}, 每段 {args.dwell}s "
          f"(freq={args.freq}MHz tx_chan={args.tx_chan})")
    for i, tx_gain in enumerate(tx_gains, 1):
        print(f"\n=== [{i}/{len(tx_gains)}] TX gain={tx_gain}dB  "
              f"发射 {args.dwell}s  {time.strftime('%H:%M:%S')} ===")
        beacon = _spawn(_beacon_cmd(args, tx_gain, args.dwell + 5))
        drain = _Drainer(beacon)
        try:
            t0 = time.time()
            while time.time() - t0 < args.dwell:
                if beacon.poll() is not None:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            _stop(beacon)
            print("\n[中断] 停止.")
            return
        _stop(beacon)
        drain.join()
        if drain.saw_error:
            for ln in drain.lines[-5:]:
                print(f"    | {ln}")
        time.sleep(args.settle)
    print("\n[TX-only] 全部段发射完毕.")


def run_role_rx(args, rx_gains):
    """两机拓扑的 RX 侧: 假设对端已在发射, 只扫 rx_gain 并解析排名."""
    results = []
    total = len(rx_gains)
    print(f"[RX-only] 假设对端已在 {args.freq}MHz 发射. 扫 rx_gains={rx_gains}, "
          f"每个 {args.dwell}s.")
    try:
        for i, rx_gain in enumerate(rx_gains, 1):
            print(f"  [{i:2d}/{total}] rx={rx_gain:<4g} 测量 {args.dwell}s ...",
                  end="", flush=True)
            r = run_probe_once(args, rx_gain)
            if r is None:
                return results
            results.append(r)
            flag = "" if r["parsed"] else "  [未解析到结果!]"
            print(f"\r  [{i:2d}/{total}] rx={rx_gain:<4g} | cap={r['cap']:<4d} "
                  f"det={r['det']:<4d} LTS={r['lts']:<4d} HDR={r['hdr']:<3d} "
                  f"CRC={r['crc']:<3d} crc/lts={r['crc_lts_pct']:5.1f}%{flag}")
            time.sleep(args.settle)
    except KeyboardInterrupt:
        print("\n[中断] 停止扫描.")
    return results


def _rank_results(results, mode):
    """按 --rank 选的指标对结果排序, 返回 (排序后列表, 表头说明).

    crc      = 累计 CRC 好包数最多 (默认, 各组合 dwell 相同 -> 直接代表吞吐).
    crclts   = 每包成功率 crc/lts 最高, 但只在 CRC≥下限的组合里挑 -> 滤掉
               "只收到两三个包但都对" 的高比率假象 (那种链路吞吐其实很差).
    balanced = CRC×(crc/lts), 同时奖励吞吐和每包可靠性.
    """
    max_crc = max((r["crc"] for r in results), default=0)
    if mode == "balanced":
        keyf = lambda r: (r["crc_lts_pct"] / 100.0 * r["crc"], r["crc"], r["hdr"])
        note = "按 balanced = CRC×(crc/lts) 降序 (兼顾吞吐与每包可靠性)"
        pool = results
    elif mode == "crclts":
        floor = max(3.0, 0.25 * max_crc)
        pool = [r for r in results if r["crc"] >= floor] or results
        keyf = lambda r: (r["crc_lts_pct"], r["crc"], r["hdr"])
        note = (f"按 crc/lts(每包成功率) 降序, 仅在 CRC>={floor:.0f} 的组合里挑 "
                f"(滤掉'只收到两三个包但都对'的高比率假象)")
    else:  # crc
        keyf = lambda r: (r["crc"], r["hdr"], r["crc_lts_pct"])
        note = "按累计 CRC 通过数降序 (越大=该组合每 dwell 收到的好包越多)"
        pool = results
    return sorted(pool, key=keyf, reverse=True), note


def summarize(args, results):
    if not results:
        print("\n[汇总] 没有任何结果.")
        return
    ranked, note = _rank_results(results, args.rank)
    print("\n" + "=" * 78)
    print(f"[排名 --rank {args.rank}] {note}")
    print("-" * 78)
    print(f"{'#':>2}  {'tx':>4} {'rx':>4} | {'cap':>4} {'det':>5} {'LTS':>5} "
          f"{'HDR':>4} {'CRC':>4}  {'crc/lts':>7}")
    for i, r in enumerate(ranked[:12], 1):
        tg = r["tx_gain"] if r["tx_gain"] is not None else float("nan")
        print(f"{i:>2}  {tg:>4g} {r['rx_gain']:>4g} | {r['cap']:>4d} "
              f"{r['det']:>5d} {r['lts']:>5d} {r['hdr']:>4d} {r['crc']:>4d}  "
              f"{r['crc_lts_pct']:>6.1f}%")

    best = ranked[0]
    if best["crc"] == 0:
        print("\n[结论] 所有组合 CRC=0 — 链路没打通. 排查方向:")
        print("  * 天线/馈线是否接在 U220202 的 RF B TX/RX 口 (--tx-chan 1)?")
        print("  * 收发 --freq / --samp-rate / --osf / --mod 是否一致?")
        print("  * 增益是否过低(信号弱)或过高(前端饱和)? 试着扩大 --tx-gains/--rx-gains.")
        print("  * det 高但 LTS=0 = 纯噪声; LTS>0 但 CRC=0 = 信号在但 BER 太高.")
        if args.csv:
            _write_csv(args, results)
            print(f"\n[已保存] 明细写入 {args.csv}")
        return

    tg = best["tx_gain"]
    print("\n[推荐] 最佳组合: "
          + (f"TX --gain {tg:g}  " if tg is not None else "")
          + f"RX --rx-gain {best['rx_gain']:g}   "
          f"(CRC={best['crc']}/{args.dwell:g}s, crc/lts={best['crc_lts_pct']:.1f}%)")

    if tg is not None:
        rx_chan = f" --rx-chan {args.rx_chan}" if args.rx_chan else ""
        tx_chan = f" --tx-chan {args.tx_chan}" if args.tx_chan else ""
        print("\n可直接复制运行 (分辨率/codec 按需改):")
        print(f"  RX: python usrp_ofdm_video.py --mode rxvideo "
              f"--rx-uri \"{args.rx_uri}\" --freq {args.freq:g} "
              f"--rx-gain {best['rx_gain']:g} --samp-rate {args.samp_rate:g} "
              f"--osf {args.osf:g} --mod {args.mod} --codec h265 --rx-bits 18 "
              f"--duration 60{rx_chan}")
        print(f"  TX: python usrp_ofdm_video.py --mode txvideo "
              f"--tx-uri \"{args.tx_uri}\" --freq {args.freq:g} "
              f"--gain {tg:g} --samp-rate {args.samp_rate:g} "
              f"--osf {args.osf:g} --mod {args.mod} --codec h265 --h264-crf 28 "
              f"--fwidth 1280 --fheight 640 --tx-fps 20 --preencode "
              f"--duration 90{tx_chan}")

    if args.csv:
        _write_csv(args, results)
        print(f"\n[已保存] 明细写入 {args.csv}")


def _write_csv(args, results):
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["freq_mhz", "samp_rate", "osf", "mod",
                    "tx_gain", "rx_gain", "captures", "det", "lts",
                    "hdr", "crc", "crc_over_lts_pct", "parsed"])
        for r in results:
            w.writerow([args.freq, args.samp_rate, args.osf, args.mod,
                        r["tx_gain"], r["rx_gain"], r["cap"], r["det"],
                        r["lts"], r["hdr"], r["crc"],
                        f"{r['crc_lts_pct']:.1f}", int(r["parsed"])])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="USRP B210 收发功率自动调试 (扫 tx/rx 增益网格, 按 CRC 排名)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--role", default="both", choices=["both", "tx", "rx"],
                   help="both=单机同时拉起收发并扫描(默认); "
                        "tx=只发(两机拓扑的发射端); rx=只收并解析(两机拓扑的接收端)")
    p.add_argument("--script", default=os.path.join(here, "usrp_ofdm_video.py"),
                   help="usrp_ofdm_video.py 路径 (默认同目录)")
    p.add_argument("--tx-uri", default="serial=U220202", help="TX 设备 (默认 U220202)")
    p.add_argument("--rx-uri", default="serial=U220213", help="RX 设备 (默认 U220213)")
    p.add_argument("--tx-chan", type=int, default=1, choices=[0, 1],
                   help="TX RF 前端通道. U220202 的 RF A 口损坏, 默认 1(RF B)")
    p.add_argument("--rx-chan", type=int, default=0, choices=[0, 1],
                   help="RX RF 前端通道, 默认 0(RF A)")
    p.add_argument("--freq", type=float, default=1000, help="中心频率 MHz (默认 1000)")
    p.add_argument("--samp-rate", type=float, default=20,
                   help="基带采样率 MHz, 收发须一致 (默认 20)")
    p.add_argument("--osf", type=float, default=1.0, choices=[1.0, 1.5],
                   help="过采样因子, 收发须一致 (默认 1.0)")
    p.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam", "64qam"],
                   help="调制, 收发须一致 (默认 qpsk)")
    p.add_argument("--tx-gains", default=None,
                   help="TX 增益列表: '55,60,65' 或 'start:stop:step'. "
                        "不给则按 --freq 自动选")
    p.add_argument("--rx-gains", default=None,
                   help="RX 增益列表, 同上. 不给则按 --freq 自动选")
    p.add_argument("--dwell", type=float, default=6.0,
                   help="每个组合的测量时长 s (默认 6)")
    p.add_argument("--warmup", type=float, default=2.5,
                   help="beacon 启动后等待暖机的秒数 (默认 2.5)")
    p.add_argument("--settle", type=float, default=1.0,
                   help="组合之间留给 USB 设备释放的间隔 s (默认 1.0)")
    p.add_argument("--rank", default="crc", choices=["crc", "crclts", "balanced"],
                   help="选最佳组合的指标: crc=好包数最多(默认,=吞吐); "
                        "crclts=每包成功率 crc/lts 最高(带 CRC 下限, 防小样本假象); "
                        "balanced=CRC×(crc/lts) 兼顾两者")
    p.add_argument("--csv", default="power_tune_results.csv",
                   help="明细 CSV 输出路径 (默认 power_tune_results.csv; 设为 '' 关闭)")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要执行的子进程命令, 不真正跑")
    args = p.parse_args()

    dtx, drx = _default_gains(args.freq)
    tx_gains = _parse_gain_list(args.tx_gains) if args.tx_gains else dtx
    rx_gains = _parse_gain_list(args.rx_gains) if args.rx_gains else drx

    if not os.path.isfile(args.script):
        print(f"[错误] 找不到 worker 脚本: {args.script}")
        sys.exit(1)

    if args.dry_run:
        print("[dry-run] role =", args.role)
        if args.role in ("both", "tx"):
            print("[dry-run] beacon 命令样例 (tx_gain=%g):" % tx_gains[0])
            print("    " + " ".join(_beacon_cmd(args, tx_gains[0], args.dwell + 5)))
        if args.role in ("both", "rx"):
            print("[dry-run] rxprobe 命令样例 (rx_gain=%g):" % rx_gains[0])
            print("    " + " ".join(_probe_cmd(args, rx_gains[0], args.dwell)))
        if args.role == "both":
            n = len(tx_gains) * len(rx_gains)
        elif args.role == "tx":
            n = len(tx_gains)
        else:
            n = len(rx_gains)
        print(f"[dry-run] tx_gains={tx_gains}  rx_gains={rx_gains}  组合数={n}")
        return

    t0 = time.time()
    if args.role == "both":
        results = sweep_both(args, tx_gains, rx_gains)
        summarize(args, results)
    elif args.role == "tx":
        run_role_tx(args, tx_gains)
    else:  # rx
        results = run_role_rx(args, rx_gains)
        summarize(args, results)
    print(f"\n[完成] 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
