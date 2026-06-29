# Copy/paste guide

Two ways to recreate this project on a machine where you can't `git clone`.

## Option A — one file (easiest)

Copy/paste **`bean_bundle.py`** (a single self-extracting file) onto the target
machine and run it:

```bash
python3 bean_bundle.py            # writes everything into ./bean/
cd bean
pip install -r requirements.txt   # numpy, scikit-learn, lightgbm, joblib
python -m pytest tests/ -q        # verify
```

`bean_bundle.py` contains the whole project as a base64 archive, so the paste is
exact — no chance of mangling indentation or special characters.

## Option B — file by file

Recreate this directory tree and paste each file's contents. **Create folders
first**, then the files (order within a folder doesn't matter):

```
bean/
├── pipeline.py          # entry point: mine_rules(...) + CLI
├── synth.py             # synthetic data generator
├── benchmark.py         # benchmark runner
├── conftest.py          # makes the folder importable under pytest
├── requirements.txt
├── README.md
├── SKILL.md
├── BACKLOG.md
├── arp/
│   ├── __init__.py
│   ├── encode.py
│   ├── fast.py
│   ├── scoring.py
│   ├── targeted.py
│   ├── mixed.py
│   ├── constraints.py
│   └── progress.py
├── featgap/
│   ├── __init__.py
│   ├── deep.py
│   ├── gap.py
│   ├── screen.py
│   └── synthesize.py
└── tests/
    └── test_basic.py
```

Then, from the `bean/` folder:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                                   # verify
python pipeline.py --synthetic --n 50000 --features 60 --patterns 15 --jobs 4
```

## Sanity check

`python -m pytest tests/ -q` should report **4 passed**. If imports fail, you are
probably running from the wrong directory — run from inside `bean/` (the folder
that contains `pipeline.py`), or add it to `PYTHONPATH`.

## Rebuilding the bundle

`bean_bundle.py` is generated from this folder. After editing any file here,
regenerate it with:

```bash
python3 rebuild_bundle.py        # rewrites bean_bundle.py from the current files
```

The build is deterministic (file metadata zeroed), so an unchanged tree always
produces an identical bundle. `bean_bundle.py`, `rebuild_bundle.py`, and this guide
are not embedded inside the archive (they are wrappers/tooling, not runtime code).

## Notes

- This is the **portable runtime subset**: the `mine_rules` pipeline plus the deep
  miner, feature engineering, constraints, and the categorical-native miner. The
  original repo's reference miner / extra demos are not included here (not needed
  to run the pipeline).
- Python 3.9+; only the four packages in `requirements.txt` are required
  (`pandas` is optional, used only for the CSV CLI path).
