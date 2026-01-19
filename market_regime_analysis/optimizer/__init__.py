"""
Parameter optimization for market regime analysis.

This module provides tools for optimizing trading strategy parameters
to maximize out-of-sample performance while preventing overfitting.
"""

from .grid_search import GridSearchOptimizer
from .objective import ObjectiveFunction, OptimizationMetrics
from .parameter_space import ParameterSpace

__all__ = [
    "GridSearchOptimizer",
    "ObjectiveFunction",
    "OptimizationMetrics",
    "ParameterSpace",
]
