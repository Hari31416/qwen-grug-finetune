import os
import sys
import json
import asyncio
import logging
import argparse
from typing import Dict, Any, List, Set
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compress_traces")

# Load environment variables
load_dotenv()

STYLE_GUIDE_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "style_guide.md",
)


def load_compression_system_prompt() -> str:
    """Build the compressor system prompt from style_guide.md."""
    if not os.path.exists(STYLE_GUIDE_PATH):
        logger.error("Style guide not found at: %s", STYLE_GUIDE_PATH)
        sys.exit(1)

    with open(STYLE_GUIDE_PATH, "r", encoding="utf-8") as f:
        style_guide = f.read().strip()
        if "## Core Objective" in style_guide:
            style_guide = style_guide.split("## Core Objective")[1]

    return (
        "You are an expert assistant specializing in token-efficient chain-of-thought compression.\n"
        "Follow the style guide below exactly.\n"
        "Output ONLY the compressed reasoning text — no wrappers, labels, markdown fences, or commentary.\n\n"
        f"{style_guide}"
    )


def build_compressed_record(
    record: Dict[str, Any], compressed_thinking: str
) -> Dict[str, Any]:
    """Build a compressed trace record from a raw record and compressed thinking text."""
    return {
        "id": record["id"],
        "source": record["source"],
        "prompt": record["prompt"],
        "choices": record.get("choices"),
        "ground_truth": record["ground_truth"],
        "raw_thinking": record["raw_thinking"],
        "compressed_thinking": compressed_thinking,
        "raw_answer": record["raw_answer"],
        "raw_answer_correct": record["raw_answer_correct"],
    }


def write_compressed_batch(out_f, records: List[Dict[str, Any]]) -> None:
    """Append a batch of compressed records to the output JSONL file."""
    for compressed_record in records:
        out_f.write(json.dumps(compressed_record) + "\n")
    out_f.flush()


async def compress_trace(
    client: AsyncOpenAI,
    model: str,
    raw_thinking: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Compresses a single reasoning trace using the LLM API under semaphore concurrency limits."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_thinking},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            msg = response.choices[0].message
            if msg.content:
                return msg.content.strip()
            return ""
        except Exception as e:
            logger.error("API request failed for trace: %s", e)
            return ""


async def compress_record(
    client: AsyncOpenAI,
    model: str,
    record: Dict[str, Any],
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any] | None:
    """Compress one raw record and return the compressed record, or None on failure."""
    compressed_thinking = await compress_trace(
        client, model, record["raw_thinking"], system_prompt, semaphore
    )
    if not compressed_thinking:
        logger.warning(
            "Skipping ID=%s due to empty or failed compression output.", record["id"]
        )
        return None
    return build_compressed_record(record, compressed_thinking)


async def main_async(
    limit: int = None, concurrency: int = 5, batch_size: int = 10
) -> None:
    # Set up directories
    config.setup_directories()

    raw_traces_file = os.path.join(config.raw_traces, "traces.jsonl")
    compressed_traces_file = os.path.join(config.compressed_traces, "traces.jsonl")

    if not os.path.exists(raw_traces_file):
        logger.error(
            "Raw traces file not found at: %s. Please run generate_traces.py first.",
            raw_traces_file,
        )
        sys.exit(1)

    # Load raw traces
    raw_records: List[Dict[str, Any]] = []
    with open(raw_traces_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                if record.get("raw_answer_correct") is True:
                    raw_records.append(record)

    logger.info("Loaded %d raw traces from %s.", len(raw_records), raw_traces_file)

    if limit is not None:
        raw_records = raw_records[:limit]
        logger.info("Limiting compression to the first %d traces.", limit)

    # Read existing compressed traces to enable resume
    existing_ids: Set[str] = set()
    if os.path.exists(compressed_traces_file):
        with open(compressed_traces_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        existing_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
        logger.info(
            "Found %d already compressed traces. Resuming...", len(existing_ids)
        )

    # Filter remaining records
    records_to_compress = [r for r in raw_records if r["id"] not in existing_ids]
    if not records_to_compress:
        logger.info("All records already compressed! Nothing to do.")
        return

    logger.info(
        "Compressing %d traces using API concurrency limit of %d...",
        len(records_to_compress),
        concurrency,
    )

    system_prompt = load_compression_system_prompt()
    logger.info("Loaded compression system prompt from %s.", STYLE_GUIDE_PATH)

    # Configure OpenAI-compatible API client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    api_base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_API_BASE")
    )
    api_model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")

    if not api_key or not api_base or not api_model:
        logger.error(
            "API credentials or model not set. Set OPENAI_API_KEY, OPENAI_API_BASE, and OPENAI_MODEL in .env."
        )
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    semaphore = asyncio.Semaphore(concurrency)

    os.makedirs(os.path.dirname(compressed_traces_file), exist_ok=True)
    tasks = [
        asyncio.create_task(
            compress_record(client, api_model, record, system_prompt, semaphore)
        )
        for record in records_to_compress
    ]

    saved_count = 0
    pending_batch: List[Dict[str, Any]] = []
    total_to_compress = len(records_to_compress)

    from tqdm import tqdm

    with open(compressed_traces_file, "a", encoding="utf-8") as out_f:
        for completed in tqdm(
            asyncio.as_completed(tasks),
            total=total_to_compress,
            desc="Compressing traces",
        ):
            compressed_record = await completed
            if compressed_record is None:
                continue

            pending_batch.append(compressed_record)
            if len(pending_batch) >= batch_size:
                write_compressed_batch(out_f, pending_batch)
                saved_count += len(pending_batch)
                tqdm.write(
                    f"Saved {saved_count}/{total_to_compress} compressed traces."
                )
                pending_batch.clear()

        if pending_batch:
            write_compressed_batch(out_f, pending_batch)
            saved_count += len(pending_batch)
            tqdm.write(f"Saved {saved_count}/{total_to_compress} compressed traces.")

    logger.info(
        "Trace compression complete! Saved %d traces to %s.",
        saved_count,
        compressed_traces_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress raw CoT traces into Grug-style telegraphic thinking blocks."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of traces to compress"
    )
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Number of concurrent API requests"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of compressed traces to buffer before writing to disk",
    )
    args = parser.parse_args()

    asyncio.run(
        main_async(
            limit=args.limit, concurrency=args.concurrency, batch_size=args.batch_size
        )
    )


if __name__ == "__main__":
    main()
