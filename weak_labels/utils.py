"""Utility functions for I/O, HuggingFace datasets, and text processing."""

import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional
from datasets import load_dataset, Dataset
from loguru import logger


# ============================================================
# JSONL I/O
# ============================================================

def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """
    Read JSONL file line by line.
    
    Args:
        path: Path to JSONL file
    
    Yields:
        Parsed JSON objects
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return
    
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]], mode: str = "a"):
    """
    Write records to JSONL file.
    
    Args:
        path: Output path
        records: Iterable of dictionaries to write
        mode: File mode ("a" for append, "w" for overwrite)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: str | Path) -> int:
    """Count lines in JSONL file."""
    path = Path(path)
    if not path.exists():
        return 0
    
    with path.open("r") as f:
        return sum(1 for line in f if line.strip())


# ============================================================
# HuggingFace Dataset Loading
# ============================================================

def load_hf_dataset(
    repo_id: str,
    split: str = "train",
    cache_dir: Optional[str | Path] = None,
    streaming: bool = False
) -> Dataset:
    """
    Load HuggingFace dataset with caching.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split to load
        cache_dir: Optional cache directory for faster reloads
        streaming: Whether to stream dataset (for very large datasets)
    
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset: {repo_id} (split={split})")
    
    kwargs = {"split": split}
    
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    
    if streaming:
        kwargs["streaming"] = True
    
    try:
        dataset = load_dataset(repo_id, **kwargs)
        
        if not streaming:
            logger.info(f"✓ Loaded {len(dataset)} examples")
        else:
            logger.info(f"✓ Dataset loaded in streaming mode")
        
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {repo_id}: {e}")
        raise


# ============================================================
# File System Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: str | Path) -> bool:
    """Check if file exists and is not empty."""
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def get_file_size_mb(path: str | Path) -> float:
    """Get file size in MB."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


# ============================================================
# Text Processing
# ============================================================

def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.split()


def count_tokens(text: str) -> int:
    """Count tokens using simple whitespace split."""
    return len(simple_tokenize(text))


def truncate_text(text: str, max_tokens: int = 512) -> str:
    """Truncate text to max_tokens."""
    tokens = simple_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
