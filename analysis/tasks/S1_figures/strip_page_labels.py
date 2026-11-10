"""Remove the bold panel labels "(a)".. (font TT36, size 10) from one page of figures_planning_cropped.pdf.

The page is first split out with poppler's pdfseparate; the text blocks of the labels are deleted from the page's
content stream (nothing else changes) and the PDF is written again with a fresh xref. Used so that composed
figures carry only their new letters, not the old ones hidden under a crop.

    pdfseparate -f 31 -l 31 figures_planning_cropped.pdf p31.pdf
    python strip_page_labels.py p31.pdf fig_atlas_p31_unlabelled.pdf 11    # 538 is found from the page object
    python strip_page_labels.py p23.pdf fig_gin_p23_unlabelled.pdf 2
"""
import re
import sys
import zlib

LABEL = rb'(?s)BT 10 0 0 10 0 0\s+Tm\s*/TT36\s+1 Tf.*?ET'
src, dst, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = open(src, 'rb').read()
objs = {int(m.group(1)): m.group(2) for m in re.finditer(rb'(?s)(\d+) 0 obj(.*?)endobj', d)}
page = [b for b in objs.values() if re.search(rb'/Type /Page\s', b)][0]
cn = int(re.search(rb'/Contents (\d+) 0 R', page).group(1))
o = objs[cn]
c = zlib.decompress(o[o.index(b'stream') + 6:o.rindex(b'endstream')].lstrip(b'\r\n'))
found = re.findall(LABEL, c)
assert len(found) == expect and all(re.search(rb'\\\([a-z]', f) for f in found), found
z = zlib.compress(re.sub(LABEL, b'', c), 9)
objs[cn] = b'\n<</Filter /FlateDecode /Length %d >>\nstream\n' % len(z) + z + b'\nendstream\n'
tr = d[d.rfind(b'trailer'):]
root = re.search(rb'/Root (\d+) 0 R', tr).group(1)
info = re.search(rb'/Info (\d+) 0 R', tr).group(1)
size = max(objs) + 1
out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
off = {}
for n in sorted(objs):
    off[n] = len(out)
    out += b'%d 0 obj' % n + objs[n] + b'endobj\n'
x = len(out)
out += b'xref\n0 %d\n' % size + b'0000000000 65535 f \n'
for n in range(1, size):
    out += (b'%010d 00000 n \n' % off[n]) if n in off else b'0000000000 65535 f \n'
out += b'trailer\n<< /Size %d /Root %s 0 R /Info %s 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (size, root, info, x)
open(dst, 'wb').write(out)
print(f'{dst}: removed {len(found)} labels')
