"""
edupro.data
===========
Data layer: schemas, loading, cleaning, validation, integration, and pipeline.
"""

from edupro.data.pipeline import Dataset, load_all

__all__ = ["Dataset", "load_all"]
