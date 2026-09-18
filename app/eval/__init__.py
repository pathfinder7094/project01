"""Evaluation package.

The benchmark module is imported lazily to keep `python -m app.eval.benchmark`
free of runpy's "found in sys.modules" warning.
"""


def run_benchmark(*args, **kwargs):
    from .benchmark import run_benchmark as _run_benchmark
    return _run_benchmark(*args, **kwargs)


__all__ = ["run_benchmark"]
