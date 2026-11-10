"""Convert every sheet of a legacy BIFF8 .xls file to TSV with the standard library only.

Usage: python -I xls_to_tsv.py <file.xls> <outdir>
Writes <outdir>/<stem>__<sheet>.tsv, one per sheet. Handles the OLE2 container and the BIFF8 cell
records LABELSST, LABEL, NUMBER, RK, MULRK, BOOLERR and FORMULA (cached numeric/string result).
Numbers are written with repr(); no date formatting is applied.
"""
import re
import struct
import sys
from pathlib import Path

ENDOFCHAIN, FREESECT = 0xFFFFFFFE, 0xFFFFFFFF


def read_ole_stream(data, names=("Workbook", "Book")):
    assert data[:8] == bytes.fromhex("d0cf11e0a1b11ae1"), "not an OLE2 file"
    ssz = 1 << struct.unpack_from("<H", data, 30)[0]
    mssz = 1 << struct.unpack_from("<H", data, 32)[0]
    n_fat, dir_start = struct.unpack_from("<I", data, 44)[0], struct.unpack_from("<I", data, 48)[0]
    cutoff = struct.unpack_from("<I", data, 56)[0]
    mfat_start, n_mfat = struct.unpack_from("<II", data, 60)
    difat_start, n_difat = struct.unpack_from("<II", data, 68)
    sect = lambda i: data[512 + i * ssz: 512 + (i + 1) * ssz]
    difat = list(struct.unpack_from("<109I", data, 76))
    s = difat_start
    for _ in range(n_difat):
        vals = struct.unpack("<%dI" % (ssz // 4), sect(s))
        difat.extend(vals[:-1])
        s = vals[-1]
    fat = []
    for i in difat[:n_fat]:
        fat.extend(struct.unpack("<%dI" % (ssz // 4), sect(i)))

    def chain(start, table):
        out, s = [], start
        while s not in (ENDOFCHAIN, FREESECT) and len(out) <= len(table):
            out.append(s)
            s = table[s]
        return out

    directory = b"".join(sect(i) for i in chain(dir_start, fat))
    entries = []
    for k in range(len(directory) // 128):
        e = directory[k * 128:(k + 1) * 128]
        nl = struct.unpack_from("<H", e, 64)[0]
        name = e[:max(nl - 2, 0)].decode("utf-16-le", "ignore")
        start, size = struct.unpack_from("<I", e, 116)[0], struct.unpack_from("<I", e, 120)[0]
        entries.append((name, e[66], start, size))
    root = entries[0]
    for name, typ, start, size in entries:
        if name in names and typ == 2:
            if size >= cutoff:
                return b"".join(sect(i) for i in chain(start, fat))[:size]
            mfat = []
            for i in chain(mfat_start, fat):
                mfat.extend(struct.unpack("<%dI" % (ssz // 4), sect(i)))
            ministream = b"".join(sect(i) for i in chain(root[2], fat))
            return b"".join(ministream[i * mssz:(i + 1) * mssz] for i in chain(start, mfat))[:size]
    raise ValueError("no Workbook stream")


def records(wb):
    pos = 0
    while pos + 4 <= len(wb):
        rid, ln = struct.unpack_from("<HH", wb, pos)
        yield pos, rid, wb[pos + 4: pos + 4 + ln]
        pos += 4 + ln


def rk_value(rk):
    if rk & 2:
        v = float(rk >> 2 if not (rk & 0x80000000) else (rk >> 2) - (1 << 30))
    else:
        v = struct.unpack("<d", struct.pack("<Q", (rk & 0xFFFFFFFC) << 32))[0]
    return v / 100 if rk & 1 else v


def fmt(v):
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v) if isinstance(v, float) else str(v)


class Pieces:
    """Iterator over the bytes of a record plus its CONTINUE records (for SST)."""

    def __init__(self, chunks):
        self.chunks, self.ci, self.off = chunks, 0, 0

    def take(self, n):
        out = b""
        while n:
            c = self.chunks[self.ci]
            if self.off >= len(c):
                self.ci, self.off = self.ci + 1, 0
                continue
            k = min(n, len(c) - self.off)
            out += c[self.off:self.off + k]
            self.off += k
            n -= k
        return out

    def at_chunk_end(self):
        return self.off >= len(self.chunks[self.ci])

    def read_string(self):
        nch = struct.unpack("<H", self.take(2))[0]
        flags = self.take(1)[0]
        rt = struct.unpack("<H", self.take(2))[0] if flags & 8 else 0
        ext = struct.unpack("<I", self.take(4))[0] if flags & 4 else 0
        wide, chars = flags & 1, []
        remaining = nch
        while remaining:
            if self.at_chunk_end():
                self.ci, self.off = self.ci + 1, 0
                wide = self.take(1)[0] & 1  # new option byte at start of CONTINUE
            avail = len(self.chunks[self.ci]) - self.off
            k = min(remaining, avail // (2 if wide else 1))
            if k == 0:
                self.ci, self.off = self.ci + 1, 0
                wide = self.take(1)[0] & 1
                continue
            raw = self.take(k * (2 if wide else 1))
            chars.append(raw.decode("utf-16-le" if wide else "latin-1"))
            remaining -= k
        self.take(4 * rt)
        self.take(ext)
        return "".join(chars)


def short_string(b, off, lenbytes=2):
    n = struct.unpack_from("<H" if lenbytes == 2 else "<B", b, off)[0]
    off += lenbytes
    flags = b[off]
    off += 1
    if flags & 1:
        return b[off:off + 2 * n].decode("utf-16-le")
    return b[off:off + n].decode("latin-1")


def main(path, outdir):
    path, outdir = Path(path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wb = read_ole_stream(path.read_bytes())
    recs = list(records(wb))
    sheets, sst = [], []
    for i, (pos, rid, body) in enumerate(recs):
        if rid == 0x85:  # BOUNDSHEET
            off = struct.unpack_from("<I", body, 0)[0]
            sheets.append((off, short_string(body, 6, 1)))
        elif rid == 0xFC:  # SST
            chunks = [body[8:]]
            j = i + 1
            while recs[j][1] == 0x3C:
                chunks.append(recs[j][2])
                j += 1
            n_unique = struct.unpack_from("<I", body, 4)[0]
            p = Pieces(chunks)
            sst = [p.read_string() for _ in range(n_unique)]
    by_pos = {pos: k for k, (pos, _, _) in enumerate(recs)}
    for off, name in sheets:
        k = by_pos[off]
        cells, pending = {}, None
        while True:
            k += 1
            pos, rid, b = recs[k]
            if rid == 0x0A:  # EOF
                break
            if rid == 0xFD:
                r, c, _, idx = struct.unpack_from("<HHHI", b)
                cells[(r, c)] = sst[idx]
            elif rid == 0x204:
                r, c, _ = struct.unpack_from("<HHH", b)
                cells[(r, c)] = short_string(b, 6)
            elif rid == 0x203:
                r, c, _, v = struct.unpack_from("<HHHd", b)
                cells[(r, c)] = fmt(v)
            elif rid == 0x27E:
                r, c, _, rk = struct.unpack_from("<HHHI", b)
                cells[(r, c)] = fmt(rk_value(rk))
            elif rid == 0xBD:
                r, c0 = struct.unpack_from("<HH", b)
                n = (len(b) - 6) // 6
                for j in range(n):
                    rk = struct.unpack_from("<I", b, 4 + j * 6 + 2)[0]
                    cells[(r, c0 + j)] = fmt(rk_value(rk))
            elif rid == 0x205:
                r, c, _, v, err = struct.unpack_from("<HHHBB", b)
                cells[(r, c)] = ("TRUE" if v else "FALSE") if not err else "#ERR"
            elif rid == 0x06:
                r, c = struct.unpack_from("<HH", b)
                res = b[6:14]
                if res[6:8] == b"\xff\xff":
                    if res[0] == 0:
                        pending = (r, c)
                    elif res[0] == 1:
                        cells[(r, c)] = "TRUE" if res[2] else "FALSE"
                    else:
                        cells[(r, c)] = ""
                else:
                    cells[(r, c)] = fmt(struct.unpack("<d", res)[0])
            elif rid == 0x207 and pending:
                cells[pending] = short_string(b, 0)
                pending = None
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        out = outdir / f"{path.stem}__{stem}.tsv"
        if cells:
            nr = max(r for r, _ in cells) + 1
            nc = max(c for _, c in cells) + 1
            with open(out, "w") as fh:
                for r in range(nr):
                    vals = [cells.get((r, c), "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
                            for c in range(nc)]
                    fh.write("\t".join(vals).rstrip("\t") + "\n")
        else:
            out.write_text("")
        print(out, len(cells), "cells")


if __name__ == "__main__":
    main(*sys.argv[1:])
