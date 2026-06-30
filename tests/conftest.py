import os
import sys
from pathlib import Path

# force CPU + quiet before any torch import that tests may trigger
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
