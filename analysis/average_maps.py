"""Average the probability maps of two runs (same frames) into a new zip.
Usage: python analysis/average_maps.py <a.zip> <b.zip> <out.zip> [weight_a]
"""
import sys
import zipfile

import cv2
import numpy as np


def main():
    a, b, out = sys.argv[1:4]
    wa = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zo:
        names = sorted(n for n in za.namelist() if n.endswith(".png"))
        assert names == sorted(n for n in zb.namelist() if n.endswith(".png")), "frame lists differ"
        for n in names:
            pa = cv2.imdecode(np.frombuffer(za.read(n), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            pb = cv2.imdecode(np.frombuffer(zb.read(n), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m = np.clip(np.rint(wa * pa + (1 - wa) * pb), 0, 255).astype(np.uint8)
            zo.writestr(n, cv2.imencode(".png", m)[1].tobytes())
    print("averaged", len(names), "frames ->", out)


if __name__ == "__main__":
    main()
