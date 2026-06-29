#!/usr/bin/env python3
"""Regenerate the self-extracting `bean_bundle.py` from this folder.

Run it whenever the portable/ code changes:

    python3 rebuild_bundle.py

It tars + gzips + base64-encodes every file under this directory (except the
wrappers below), and writes a fresh `bean_bundle.py`. The build is deterministic
(file metadata zeroed), so an unchanged tree reproduces an identical bundle.
"""
import base64
import gzip
import io
import os
import tarfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
# wrappers / noise that must NOT go inside the archive
EXCLUDE_NAMES = {"bean_bundle.py", "rebuild_bundle.py", "PASTE_GUIDE.md"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache"}


def _collect():
    files = []
    for root, dirs, names in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        for n in sorted(names):
            if n in EXCLUDE_NAMES or n.startswith("._") or n.endswith(".pyc"):
                continue
            full = os.path.join(root, n)
            rel = os.path.relpath(full, HERE)
            files.append((full, rel))
    return files


def _deterministic(ti):
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def build():
    files = _collect()
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:          # plain tar first
        for full, rel in files:
            tar.add(full, arcname=rel, filter=_deterministic)
    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0) as gz:  # reproducible gzip
        gz.write(tar_buf.getvalue())
    b64 = base64.b64encode(gz_buf.getvalue()).decode()
    data = "\n".join(textwrap.wrap(b64, 76))

    script = TEMPLATE % data
    out = os.path.join(HERE, "bean_bundle.py")
    with open(out, "w") as fh:
        fh.write(script)
    print(f"rebuilt bean_bundle.py  ({len(files)} files embedded, "
          f"{len(script):,} chars)")
    for _, rel in files:
        print("  ", rel)


TEMPLATE = '''#!/usr/bin/env python3
"""Self-extracting Bean bundle.

Copy/paste this ONE file onto the target machine, then run:

    python3 bean_bundle.py

It writes the full project (arp/, featgap/, pipeline.py, synth.py, tests/, docs)
into ./bean/ in the current directory. No network, no git, no pip needed to unpack.
Then:  cd bean && pip install -r requirements.txt && python -m pytest tests/ -q
"""
import base64, io, tarfile, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "bean"
DATA = """
%s
"""

def main():
    raw = base64.b64decode("".join(DATA.split()))
    os.makedirs(OUT, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(OUT)
    n = sum(len(f) for _, _, f in os.walk(OUT))
    print(f"Extracted {n} files into ./{OUT}/")
    print(f"Next:  cd {OUT} && pip install -r requirements.txt && python -m pytest tests/ -q")

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    build()
