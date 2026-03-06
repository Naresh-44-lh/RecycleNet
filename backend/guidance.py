"""
guidance.py — Waste disposal guidance data.

This module re-exports the GUIDANCE dictionary from ml.predict so that
backend code can import guidance data independently of the ML model,
for example to serve static info endpoints without loading TensorFlow.
"""
from ml.predict import GUIDANCE

__all__ = ["GUIDANCE"]
