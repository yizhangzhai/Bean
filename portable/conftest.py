"""Make the bundle root importable so `import arp` / `import featgap` work under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
