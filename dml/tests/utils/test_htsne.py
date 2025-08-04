"""Tests for heterogenous SNE functionality"""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import pickle
from dml.stats import *
from dml.utils.htsne import htsne
import numpy as np

DATA_DIR = "pysp/tests/data"
ANSWER_DIR = "pysp/tests/answerkeys"

def test_htsne() -> None:
    """Test if HTSNE behaves as expected with data only input."""
    with open(os.path.join(DATA_DIR, "testInput_htsne.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    answer = np.load(os.path.join(ANSWER_DIR, "testOutput_htsne.npy"))
    rv = htsne(data, seed=10)

    assert np.all(answer == rv)