"""Tests for utils/automatic.py"""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import pytest
import pickle
import numpy as np
from pysp.utils.automatic import get_dpm_mixture, get_estimator
from pysp.bstats import *


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture(case_id: int) -> None:
    """Tests if pipeline for creating estimator and estiamting a DPM works."""
    with open(f"pysp/tests/data/testInput_automatic{case_id}.pkl", 'rb') as f:
        data = pickle.load(f)
    
    model = get_dpm_mixture(data, rng=np.random.RandomState(1))

    with open(f"pysp/tests/answerkeys/testOutput_automatic{case_id}.txt", 'r') as f:
        answer = f.read()
    
    assert answer == str(model)

