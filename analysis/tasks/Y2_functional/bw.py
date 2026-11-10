"""Minimal read-only bigWig reader (pure Python + numpy): UCSC bbi format, no zoom levels used."""

from __future__ import annotations

import mmap
import struct
import zlib
from pathlib import Path

import numpy as np


class BigWig:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        h = struct.unpack("<IHHQQQHHQQIQ", self.m[:64])
        if h[0] != 0x888FFC26:
            raise ValueError("not a little-endian bigWig")
        self.chrom_tree, self.data_off, self.index_off, self.buf = h[3], h[4], h[5], h[10]
        self.chroms = self._read_chroms()

    def _read_chroms(self) -> dict[str, tuple[int, int]]:
        o = self.chrom_tree
        magic, block, ksize, vsize, n, _ = struct.unpack("<IIIIQQ", self.m[o:o + 32])
        assert magic == 0x78CA8C91 and vsize == 8
        out = {}

        def node(off):
            leaf, _r, cnt = struct.unpack("<BBH", self.m[off:off + 4])
            p = off + 4
            for _ in range(cnt):
                key = self.m[p:p + ksize].rstrip(b"\0").decode()
                p += ksize
                if leaf:
                    cid, size = struct.unpack("<II", self.m[p:p + 8])
                    out[key] = (cid, size)
                    p += 8
                else:
                    child = struct.unpack("<Q", self.m[p:p + 8])[0]
                    p += 8
                    node(child)

        node(o + 32)
        return out

    def leaves(self, chrom: str) -> np.ndarray:
        """Sorted (start, end, offset, size) of the data blocks of a chromosome."""
        cid = self.chroms[chrom][0]
        o = self.index_off
        assert struct.unpack("<I", self.m[o:o + 4])[0] == 0x2468ACE0
        rows = []
        stack = [o + 48]
        while stack:
            off = stack.pop()
            leaf, _r, cnt = struct.unpack("<BBH", self.m[off:off + 4])
            p = off + 4
            for _ in range(cnt):
                if leaf:
                    sc, sb, ec, eb, d, s = struct.unpack("<IIIIQQ", self.m[p:p + 32])
                    p += 32
                    if sc <= cid <= ec:
                        rows.append((sb if sc == cid else 0, eb if ec == cid else 2 ** 31, d, s))
                else:
                    sc, sb, ec, eb, d = struct.unpack("<IIIIQ", self.m[p:p + 24])
                    p += 24
                    if sc <= cid <= ec:
                        stack.append(d)
        a = np.array(rows, dtype=np.int64).reshape(-1, 4)
        return a[np.argsort(a[:, 0], kind="stable")]

    def block(self, off: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Starts, ends and values of the items in a data block."""
        raw = bytes(self.m[off:off + size])
        if self.buf:
            raw = zlib.decompress(raw)
        cid, cs, ce, step, span, typ, _r, n = struct.unpack("<IIIIIBBH", raw[:24])
        body = np.frombuffer(raw, offset=24, dtype=np.uint8)
        if typ == 1:
            a = np.frombuffer(body, dtype=np.dtype([("s", "<u4"), ("e", "<u4"), ("v", "<f4")]), count=n)
            return a["s"].astype(np.int64), a["e"].astype(np.int64), a["v"].astype(np.float32)
        if typ == 2:
            a = np.frombuffer(body, dtype=np.dtype([("s", "<u4"), ("v", "<f4")]), count=n)
            s = a["s"].astype(np.int64)
            return s, s + span, a["v"].astype(np.float32)
        if typ == 3:
            v = np.frombuffer(body, dtype="<f4", count=n)
            s = cs + step * np.arange(n, dtype=np.int64)
            return s, s + span, v.astype(np.float32)
        raise ValueError(f"unknown section type {typ}")

    def fill(self, chrom: str, starts: np.ndarray, ends: np.ndarray) -> dict[int, np.ndarray]:
        """For half-open intervals (sorted by start or not): value array per interval, NaN where no data."""
        L = self.leaves(chrom)
        order = np.argsort(starts)
        out = {int(i): np.full(int(ends[i] - starts[i]), np.nan, dtype=np.float32) for i in order}
        if len(L) == 0:
            return out
        maxend = np.maximum.accumulate(L[:, 1])
        cache: dict[int, tuple] = {}
        for i in order:
            s0, e0 = int(starts[i]), int(ends[i])
            lo = int(np.searchsorted(maxend, s0, side="right"))
            hi = int(np.searchsorted(L[:, 0], e0, side="left"))
            arr = out[int(i)]
            for b in range(lo, hi):
                if L[b, 1] <= s0:
                    continue
                if b not in cache:
                    if len(cache) > 64:
                        cache.pop(next(iter(cache)))
                    cache[b] = self.block(int(L[b, 2]), int(L[b, 3]))
                s, e, v = cache[b]
                j0 = np.searchsorted(e, s0, side="right")
                j1 = np.searchsorted(s, e0, side="left")
                for j in range(j0, j1):
                    a, c = max(int(s[j]), s0), min(int(e[j]), e0)
                    if c > a:
                        arr[a - s0:c - s0] = v[j]
        return out
