"""Stellar Polytropic Deviation Analyzer."""

from stellar_analyzer.core.pipeline import analyze_mesa_job, analyze_profile, analyze_star, batch_analyze

__all__ = ["analyze_star", "analyze_profile", "analyze_mesa_job", "batch_analyze"]
__version__ = "0.2.0"
