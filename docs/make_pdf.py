"""TECHNICAL.md as a PDF, for the submission form.

    python docs/make_pdf.py [OUT.pdf]      # default: Blundell.pdf in the repository root

Needs the `markdown` package (pip install markdown) and Google Chrome, which prints the page.
Edit TECHNICAL.md, then run this again: the PDF is never edited by hand.
"""
import os
import subprocess
import sys
import tempfile

import markdown

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #111; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } p, li { margin: 4pt 0; }
hr { border: 0; border-top: 1px solid #ddd; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt; line-height: 1.35; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f1f1ef; } tr, img, pre { break-inside: avoid; } h2, h3 { break-after: avoid; }
code { font: 8.8pt Menlo, monospace; background: #f4f4f2; padding: 0 2pt; }
pre { background: #f4f4f2; padding: 6pt 8pt; overflow: hidden; }
pre code { padding: 0; font-size: 8pt; white-space: pre-wrap; } img { max-width: 100%; }
a { color: #1c5cab; }
"""


def main(out):
    with open(os.path.join(ROOT, "TECHNICAL.md"), encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables", "fenced_code", "sane_lists"])
    page = ("<!doctype html><html><head><meta charset='utf-8'><base href='file://{}/'>"
            "<style>{}</style></head><body>{}</body></html>").format(ROOT, CSS, body)
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "technical.html")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(page)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--user-data-dir=" + os.path.join(folder, "profile"),
                        "--allow-file-access-from-files", "--print-to-pdf=" + out,
                        "file://" + source], check=True, capture_output=True, timeout=180)
    print("-> {} ({} bytes)".format(out, os.path.getsize(out)))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "Blundell.pdf"))
