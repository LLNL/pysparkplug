"""Tests for automatic.py with mpi4py support using 4 cores."""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import pytest
import pickle
from mpi4py import MPI
import numpy as np
from pysp.mpi4py.utils.automatic import get_dpm_mixture_mpi
from pysp.bstats import *


@pytest.mark.parametrize("case_id", [0, 1])
def test_get_dpm_mixture_mpi(case_id: int) -> None:
    """Tests if pipeline for creating estimator and estiamting a DPM works."""
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    with open(f"pysp/tests/data/testInput_automatic{case_id}.pkl", 'rb') as f:
        data = pickle.load(f)
    
    model = get_dpm_mixture_mpi(data, rng=np.random.RandomState(1))

    with open(f"pysp/tests/answerkeys/testOutput_automatic_get_dpm_mixture_mpi_n4_case{case_id}.txt", 'r') as f:
        answer = f.read()
    
    assert answer == str(model)

