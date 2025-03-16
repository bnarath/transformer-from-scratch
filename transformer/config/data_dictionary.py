"""
This is the dictionary containing parameter info for the package run
"""

from enum import Enum
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


class HuggingFaceData(Enum):
    dataset = "ai4bharat/samanantar"
    name = "ml"
    split = "train"
    remove_feature = ["idx"]
    save_location = "data/english_malayalam_data.arrow"
    src_column = "src"
    tgt_column = "tgt"


class Training:
    framework = "torch"
