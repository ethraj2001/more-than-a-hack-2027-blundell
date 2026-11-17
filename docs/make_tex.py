"""TECHNICAL.md as LaTeX, and the PDF typeset from it.

    python docs/make_tex.py [OUT.pdf]      # writes TECHNICAL.tex, then Blundell.pdf with xelatex

TECHNICAL.md stays the source: edit it, then run this again. Needs the `markdown` package and a
TeX installation with xelatex. docs/make_pdf.py makes a PDF without TeX, through a browser.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

import markdown

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{array}
\usepackage{booktabs}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
\emergencystretch=2em
\begin{document}
"""
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "\u2212": r"\ensuremath{-}", "\u00b1": r"\ensuremath{\pm}", "\u00d7": r"\ensuremath{\times}",
           "\u00bd": r"\ensuremath{\tfrac{1}{2}}", "\u2192": r"\ensuremath{\to}",
           "\u2265": r"\ensuremath{\geq}", "<": r"\textless{}", ">": r"\textgreater{}"}
INLINE = {"strong": r"\textbf{", "b": r"\textbf{", "em": r"\emph{", "i": r"\emph{",
          "code": r"\texttt{"}
HEADINGS = {"h2": r"\section*{", "h3": r"\subsection*{", "h4": r"\subsubsection*{"}


def escape(text):
    return "".join(SPECIAL.get(char, char) for char in text)


class Converter(HTMLParser):
    """The HTML python-markdown makes of TECHNICAL.md, as LaTeX."""

    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.out = []          # the document so far
        self.stack = []        # where text is being collected: the document, or a table cell
        self.table = None      # rows of cells while inside a table
        self.pre = False
        self.link = None

    def emit(self, text):
        (self.stack[-1] if self.stack else self.out).append(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1":
            self.emit("{\\LARGE\\bfseries ")
        elif tag in HEADINGS:
            self.emit("\n" + HEADINGS[tag])
        elif tag == "p":
            self.emit("\n")
        elif tag in ("ul", "ol"):
            name = "itemize" if tag == "ul" else "enumerate"
            start = ("\\setcounter{enumi}{%d}\n" % (int(attrs["start"]) - 1)
                     if attrs.get("start") else "")
            self.emit("\n\\begin{%s}\\setlength{\\itemsep}{2pt}\n%s" % (name, start))
        elif tag == "li":
            self.emit("\\item ")
        elif tag in INLINE and not self.pre:
            self.emit(INLINE[tag])
        elif tag == "pre":
            self.pre = True
            self.emit("\n\\begin{Verbatim}[fontsize=\\footnotesize,frame=single,framesep=4pt]\n")
        elif tag == "a":
            self.link = attrs.get("href", "")
            self.emit("\\href{%s}{" % self.link.replace("%", "\\%").replace("#", "\\#"))
        elif tag == "img":
            self.emit("\\includegraphics[width=\\linewidth]{%s}" % attrs["src"])
        elif tag == "hr":
            self.emit("\n\\medskip\\hrule\\medskip\n")
        elif tag == "blockquote":
            self.emit("\n\\begin{quote}")
        elif tag == "table":
            self.table = []
        elif tag == "tr":
            self.table.append([])
        elif tag in ("td", "th"):
            self.stack.append([])
        elif tag == "br":
            self.emit("\\\\\n")

    def handle_endtag(self, tag):
        if tag == "h1":
            self.emit("}\\par\\medskip\n")
        elif tag in HEADINGS or (tag in INLINE and not self.pre) or tag == "a":
            self.emit("}")
            if tag in HEADINGS:
                self.emit("\n")
        elif tag == "p":
            self.emit("\n")
        elif tag in ("ul", "ol"):
            self.emit("\\end{%s}\n" % ("itemize" if tag == "ul" else "enumerate"))
        elif tag == "li":
            self.emit("\n")
        elif tag == "pre":
            self.pre = False
            self.emit("\\end{Verbatim}\n")
        elif tag == "blockquote":
            self.emit("\\end{quote}\n")
        elif tag in ("td", "th"):
            self.table[-1].append("".join(self.stack.pop()).strip())
        elif tag == "table":
            rows, self.table = self.table, None
            self.emit(table(rows))

    def handle_data(self, data):
        if self.pre:
            self.emit(data)
        elif self.table is not None and not self.stack:
            return  # whitespace between cells
        else:
            self.emit(re.sub(r'"([^"]*)"', r"``\1''", escape(data)))


def table(rows):
    """A longtable whose columns share the line in proportion to how much text they hold."""
    columns = len(rows[0])
    plain = [[re.sub(r"\\[a-zA-Z]+\*?|[{}]", "", cell) for cell in row] for row in rows]
    weight = [min(max(sum(len(row[c]) for row in plain) / float(len(plain)), 8.0), 60.0)
              for c in range(columns)]
    share = [w / sum(weight) for w in weight]
    spec = "".join(">{\\raggedright\\arraybackslash}p{\\dimexpr %.3f\\linewidth-2\\tabcolsep\\relax}"
                   % s for s in share)
    line = lambda row: " & ".join("{}" + cell for cell in row) + " \\\\\n"  # noqa: E731
    head = line(["\\textbf{%s}" % cell for cell in rows[0]])
    return ("\n{\\footnotesize\n\\begin{longtable}{%s}\n\\toprule\n%s\\midrule\n\\endhead\n%s"
            "\\bottomrule\n\\end{longtable}\n}\n" % (spec, head, "\\addlinespace[3pt]\n".join(line(r) for r in rows[1:])))


def figures(tex):
    """A paragraph that is only a picture becomes a centred figure of its own."""
    pattern = re.compile(r"\n((?:\\href\{[^}]*\}\{)?\\includegraphics\[width=\\linewidth\]\{[^}]*\}\}?)\n")
    return pattern.sub(lambda m: "\n\\begin{center}\n%s\n\\end{center}\n" % m.group(1), tex)


def main(out):
    with open(os.path.join(ROOT, "TECHNICAL.md"), encoding="utf-8") as handle:
        html = markdown.markdown(handle.read(), extensions=["tables", "fenced_code", "sane_lists"])
    converter = Converter()
    converter.feed(html)
    tex = PREAMBLE + figures("".join(converter.out)) + "\n\\end{document}\n"
    source = os.path.join(ROOT, "TECHNICAL.tex")
    with open(source, "w", encoding="utf-8") as handle:
        handle.write(tex)
    with tempfile.TemporaryDirectory() as folder:
        for _ in range(2):  # twice, so longtable settles its column widths
            run = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                                  "-output-directory", folder, source], cwd=ROOT,
                                 capture_output=True, text=True, timeout=300)
            if run.returncode != 0:
                sys.exit(run.stdout[-3000:])
        shutil.copy(os.path.join(folder, "TECHNICAL.pdf"), out)
    print("-> TECHNICAL.tex, {} ({} bytes)".format(out, os.path.getsize(out)))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "Blundell.pdf"))
