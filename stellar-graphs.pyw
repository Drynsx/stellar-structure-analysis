"""Launch the Stellar Analyzer graph workspace without a console window."""

from stellar_analyzer.cli import DEFAULT_JOB
from stellar_analyzer.core.pipeline import analyze_mesa_job
from stellar_analyzer.visualization import show_plot_window


if __name__ == "__main__":
    show_plot_window(analyze_mesa_job(DEFAULT_JOB, None))
