"""Tests for heterogenous SNE functionality"""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import pickle
from pysp.stats import *
from pysp.utils.htsne import htsne
import numpy as np

def test_htsne() -> None:
    """Test if HTSNE behaves as expected with data only input."""
    with open('pysp/tests/data/testInput_htsne.pkl', 'rb') as f:
        data = pickle.load(f)
    
    answer = np.load('pysp/tests/answerkeys/testOutput_htsne.npy')
    rv = htsne(data, seed=10)

    assert np.all(answer == rv)