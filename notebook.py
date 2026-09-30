"""Kaggle notebook entry point — thin wrapper around arc2.submission.

This file is what the Kaggle notebook cells execute.
Keep it minimal — all logic lives in the arc2 package.

Usage in Kaggle notebook:
    !pip install numpy  # should be pre-installed
    %run notebook.py
"""

from arc2.submission import main

main(
    task_dir="/kaggle/input/arc-prize-2026-arc-agi-2/challenges",
    output_path="submission.json",
    time_budget=36000.0,  # 10h, 2h margin from 12h limit
)
