#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import shutil
from typing import Optional

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_hf_data")

DEFAULT_HF_REPO = "hari31416/qwen-grug-finetune"


def download_from_hf(
    repo_id: str,
    output_dir: str,
    iteration: str = "iteration-2-regularized",
    token: Optional[str] = None,
) -> bool:
    """Downloads dataset files from Hugging Face Hub into local directory."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logger.error(
            "huggingface_hub library is not installed. Please run: pip install huggingface_hub"
        )
        return False

    logger.info("Connecting to Hugging Face Repo: %s", repo_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        logger.info("Found %d files in repo '%s'", len(repo_files), repo_id)
    except Exception as e:
        logger.error("Failed to list files in HF repo '%s': %s", repo_id, e)
        return False

    # Target filenames to look for
    target_files = ["train.jsonl", "valid.jsonl"]
    downloaded_count = 0

    for fname in target_files:
        # Search candidate paths in the repo
        candidate_paths = [
            f"{iteration}/data/{fname}",
            f"data/{fname}",
            f"it-2-3/{fname}",
            f"it-1/{fname}",
            fname,
        ]

        found_path = None
        for cand in candidate_paths:
            if cand in repo_files:
                found_path = cand
                break

        if found_path:
            logger.info("Downloading '%s' from HF path: '%s'...", fname, found_path)
            try:
                downloaded_file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=found_path,
                    repo_type="model",
                    token=token,
                )
                dest_file_path = os.path.join(output_dir, fname)
                shutil.copy2(downloaded_file_path, dest_file_path)
                logger.info("Successfully saved: %s", dest_file_path)
                downloaded_count += 1
            except Exception as dl_err:
                logger.error("Failed to download '%s': %s", found_path, dl_err)
        else:
            logger.warning("Could not find candidate file for '%s' in repo.", fname)

    if downloaded_count > 0:
        logger.info(
            "Successfully downloaded %d dataset file(s) to '%s'.",
            downloaded_count,
            output_dir,
        )
        return True
    else:
        logger.error("No dataset files were downloaded.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SFT datasets (train.jsonl & valid.jsonl) from Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repository ID (default: {DEFAULT_HF_REPO})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.data_dir,
        help=f"Directory to save train.jsonl and valid.jsonl (default: {config.data_dir})",
    )
    parser.add_argument(
        "--iteration",
        type=str,
        default="iteration-2-regularized",
        help="Subfolder iteration in HF repo (e.g. iteration-2-regularized, iteration-1)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face User Access Token",
    )

    args = parser.parse_args()

    success = download_from_hf(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        iteration=args.iteration,
        token=args.token,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
