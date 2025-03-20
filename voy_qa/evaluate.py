#!/usr/bin/env python3
"""
VoyQA Evaluation Script

Usage:
    python evaluate.py --create-eval-set --num-samples 50  # Create evaluation set
    python evaluate.py --run-eval                          # Run evaluation
    python evaluate.py --analyze                           # Analyze results
    python evaluate.py --content-eval                      # Evaluate content quality
    python evaluate.py --all                               # Run all steps
"""

import sys
import os
from pathlib import Path

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the evaluation module
from src.evaluation import main

if __name__ == "__main__":
    # Create evaluation directory if it doesn't exist
    Path("evaluation").mkdir(exist_ok=True)
    
    # Run the main function from the evaluation module
    main() 