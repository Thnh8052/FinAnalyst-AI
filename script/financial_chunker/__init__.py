"""
Financial Chunker Package for FinAnalyst-AI.
Supports Multi-Granularity Financial Document Chunking:
- Method 1: Naive Fixed-Size Chunking (Baseline 1)
- Method 2: Deterministic Structure-Aware Chunking (Baseline 2)
- Method 2.5: Heading Pre-Split + Batch LLM Merge (Intermediate)
- Method 3: Comprehensive LLM-Assisted Structure-Aware Chunking (Proposed Method)
"""

from __future__ import annotations

__version__ = "0.1.0"
