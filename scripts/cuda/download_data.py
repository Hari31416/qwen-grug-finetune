#!/usr/bin/env python3
import os
import sys

# Forward execution to scripts/download_hf_data.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.download_hf_data import main

if __name__ == "__main__":
    main()
