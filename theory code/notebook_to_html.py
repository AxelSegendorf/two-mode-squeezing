"""notebook_to_html.py — .ipynb → .html"""
import sys
from pathlib import Path

from nbconvert import HTMLExporter
import nbformat

src = Path(sys.argv[1] if len(sys.argv) > 1 else input("notebook: "))
html, _ = HTMLExporter().from_notebook_node(nbformat.read(src, as_version=4))
dst = src.with_suffix(".html")
dst.write_text(html, encoding="utf-8")
print(dst)
