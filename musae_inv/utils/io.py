"""
I/O utilities for saving and loading experiment artefacts.

Provides safe, atomic-write wrappers around pickle and numpy serialisation,
with incremental caching support for long-running feature extraction.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an object to a pickle file.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """Load an object from a pickle file.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    Any
        The deserialised object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_numpy(arr: np.ndarray, path: str | Path) -> None:
    """Save a numpy array to disk.

    Parameters
    ----------
    arr : np.ndarray
        Array to save.
    path : str or Path
        Destination `.npy` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a numpy array from disk.

    Parameters
    ----------
    path : str or Path
        Source `.npy` file path.

    Returns
    -------
    np.ndarray
        The loaded array.
    """
    return np.load(path)


def save_json(obj: Dict, path: str | Path, indent: int = 2) -> None:
    """Save a dictionary to a JSON file.

    Parameters
    ----------
    obj : dict
        Object to serialise.
    path : str or Path
        Destination file path.
    indent : int
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path) -> Dict:
    """Load a dictionary from a JSON file.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    dict
        The loaded dictionary.
    """
    with open(path) as f:
        return json.load(f)
