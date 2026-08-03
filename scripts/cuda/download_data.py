#!/usr/bin/env python3
import os
import sys

# Forward execution to scripts/download_hf_data.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.download_hf_data import download_from_hf, DEFAULT_HF_REPO, main


def download_hf_data(repo_id: str = DEFAULT_HF_REPO, output_dir: str = "data") -> bool:
    """Downloads SFT datasets (train.jsonl and valid.jsonl) from Hugging Face Hub."""
    return download_from_hf(repo_id=repo_id, output_dir=output_dir)


if __name__ == "__main__":
    main()
