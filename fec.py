"""Reed-Solomon erasure FEC over GF(256), numpy 向量化, 无外部依赖.

用于包级前向纠错: 把一帧切成 K 个等长数据块, 编码出 N=K+R 个块发送;
接收端收到任意 K 个 (CRC 通过的) 块即可恢复整帧 -> 在 10-15% 丢包下稳定出帧.

设计:
  - GF(256), 本原多项式 0x11d.
  - 生成矩阵 G = V @ inv(V[:K,:K]), 其中 V 为 N×K Vandermonde (a_i=exp(i) 互不相同),
    Vandermonde 为 MDS -> 任意 K 行可逆 -> 任意 K 个块都能解.
  - 系统码: 前 K 个编码块 == 原数据块 (不丢时零解码开销).
"""

import numpy as np

_PRIM = 0x11d
_GF_EXP = np.zeros(512, dtype=np.uint8)
_GF_LOG = np.zeros(256, dtype=np.int32)


def _gf_init():
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _PRIM
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_gf_init()


def _gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF inverse of 0")
    return int(_GF_EXP[255 - _GF_LOG[a]])


def _gf_mul_vec(c: int, vec: np.ndarray) -> np.ndarray:
    """标量 c (GF) 乘以 uint8 向量 vec (逐元素 GF 乘), 向量化."""
    if c == 0:
        return np.zeros_like(vec)
    out = _GF_EXP[_GF_LOG[c] + _GF_LOG[vec]].astype(np.uint8)
    out[vec == 0] = 0
    return out


def _gf_matmul(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """GF 矩阵乘: M (m,k) · X (k,L) -> (m,L), 全 uint8."""
    m, k = M.shape
    L = X.shape[1]
    out = np.zeros((m, L), dtype=np.uint8)
    for r in range(m):
        acc = np.zeros(L, dtype=np.uint8)
        Mr = M[r]
        for j in range(k):
            cj = int(Mr[j])
            if cj:
                acc ^= _gf_mul_vec(cj, X[j])
        out[r] = acc
    return out


def _gf_mat_inv(A: np.ndarray) -> np.ndarray:
    """GF(256) 方阵求逆 (高斯-约当)."""
    n = A.shape[0]
    M = np.concatenate([A.astype(np.uint8).copy(),
                        np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        piv = col
        while piv < n and M[piv, col] == 0:
            piv += 1
        if piv == n:
            raise ValueError("singular matrix in GF inverse")
        if piv != col:
            tmp = M[col].copy()
            M[col] = M[piv]
            M[piv] = tmp
        M[col] = _gf_mul_vec(_gf_inv(int(M[col, col])), M[col])
        for r in range(n):
            if r != col and M[r, col] != 0:
                M[r] ^= _gf_mul_vec(int(M[r, col]), M[col])
    return M[:, n:]


_GEN_CACHE = {}


def _generator(k: int, n: int) -> np.ndarray:
    """N×K 系统生成矩阵 (前 K 行为单位阵). 带缓存."""
    key = (k, n)
    if key in _GEN_CACHE:
        return _GEN_CACHE[key]
    if not (0 < k <= n <= 255):
        raise ValueError("require 0 < k <= n <= 255")
    V = np.zeros((n, k), dtype=np.uint8)
    for i in range(n):
        for j in range(k):
            V[i, j] = _GF_EXP[(i * j) % 255]
    top_inv = _gf_mat_inv(V[:k, :k])
    G = _gf_matmul(V, top_inv)   # (n,k), 前 k 行 = 单位阵
    _GEN_CACHE[key] = G
    return G


def rs_encode(data_blocks: np.ndarray, k: int, n: int) -> np.ndarray:
    """data_blocks: (k, L) uint8 -> 编码块 (n, L) uint8. 前 k 个 == 原数据."""
    G = _generator(k, n)
    return _gf_matmul(G, data_blocks)


def rs_decode(blocks: np.ndarray, indices, k: int, n: int) -> np.ndarray:
    """blocks: (k, L) 收到的任意 k 个编码块; indices: 它们在 0..n-1 中的下标;
    返回恢复的 k 个数据块 (k, L)."""
    idx = list(indices)
    if len(idx) < k:
        raise ValueError("need at least k blocks to decode")
    idx = idx[:k]
    G = _generator(k, n)
    Gsub = G[idx, :]                 # (k,k)
    Ginv = _gf_mat_inv(Gsub)
    return _gf_matmul(Ginv, blocks[:k])


# ============================================================================
# 帧级封装: 整帧字节 <-> N 个等长编码块 (供 TX 切块 / RX 收齐恢复)
# ============================================================================

def fec_pack_frame(data: bytes, block_size: int = 1100, overhead: float = 0.4):
    """整帧字节 -> (k, n, frame_len, enc(n,L)).
    k=数据块数(按 block_size 切), n=k+冗余, enc 前 k 个即原数据(系统码)."""
    frame_len = len(data)
    L = int(block_size)
    k = max(1, (frame_len + L - 1) // L)
    r = max(1, int(round(k * overhead)))
    n = min(k + r, 255)
    if k > 255:
        raise ValueError("frame too large for given block_size (k>255)")
    padded = data + b"\x00" * (k * L - frame_len)
    blocks = np.frombuffer(padded, dtype=np.uint8).reshape(k, L)
    enc = rs_encode(blocks, k, n)
    return k, n, frame_len, enc


def fec_unpack_frame(block_dict: dict, k: int, n: int, frame_len: int) -> bytes:
    """block_dict: {block_idx: uint8 数组(长 L)}; 需 >= k 个. 恢复整帧字节."""
    idxs = sorted(block_dict.keys())
    if len(idxs) < k:
        raise ValueError("not enough blocks")
    idxs = idxs[:k]
    L = int(block_dict[idxs[0]].shape[0])
    mat = np.zeros((k, L), dtype=np.uint8)
    for row, i in enumerate(idxs):
        mat[row] = block_dict[i]
    rec = rs_decode(mat, idxs, k, n)
    return rec.reshape(-1)[:frame_len].tobytes()


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    for (k, n) in [(10, 14), (12, 16), (8, 12), (16, 20)]:
        L = 1500
        data = rng.randint(0, 256, (k, L)).astype(np.uint8)
        enc = rs_encode(data, k, n)
        assert np.array_equal(enc[:k], data), "systematic check failed"
        keep = sorted(rng.choice(n, k, replace=False).tolist())
        rec = rs_decode(enc[keep], keep, k, n)
        ok = np.array_equal(rec, data)
        print("k=%2d n=%2d  drop=%d  recover=%s" % (k, n, n - k, "OK" if ok else "FAIL"))

    # 帧级测试: 模拟整帧 -> 切块编码 -> 随机丢块 -> 收齐 k 个 -> 恢复
    for flen in (5000, 23000, 47000):
        frame = bytes(rng.randint(0, 256, flen, dtype=np.uint8).tolist())
        k, n, fl, enc = fec_pack_frame(frame, block_size=1100, overhead=0.4)
        # 随机只保留 k 个 (丢掉 n-k 个)
        keep = sorted(rng.choice(n, k, replace=False).tolist())
        bd = {i: enc[i] for i in keep}
        rec = fec_unpack_frame(bd, k, n, fl)
        print("frame=%d k=%d n=%d drop=%d recover=%s"
              % (flen, k, n, n - k, "OK" if rec == frame else "FAIL"))
    print("FEC self-test done")
