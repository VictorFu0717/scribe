#!/usr/bin/env python3
"""把 docs/src/*.html 渲染成 docs/*.jpg。

用法: python docs/src/render.py docs/src/scribe-01-architecture.html 1790 1540

渲染大一點再依內容自動裁切 —— Chrome 的 --window-size 會把剛好等高的內容
切掉底部(舊版 scribe-01 就少了 SQLite 框的最後一行,肉眼很難發現)。
"""
import subprocess, sys, os
import numpy as np
from PIL import Image

src, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = os.path.join("docs", os.path.basename(src).replace(".html", ".jpg"))
png = "/tmp/_render.png"
subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
                "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={w+80},{h+300}", f"--screenshot={png}",
                f"file://{os.path.abspath(src)}"], capture_output=True)
im = Image.open(png).convert("RGB")
ink = np.argwhere(np.array(im.convert("L")) < 245)
if ink.size == 0:
    sys.exit("渲染結果是空白的")
y0, x0 = ink.min(0); y1, x1 = ink.max(0)
M = 56
im.crop((max(0, x0-M), max(0, y0-M),
         min(im.width, x1+M+1), min(im.height, y1+M+1))).save(out, quality=92)
print(f"{out}  {Image.open(out).size}")
