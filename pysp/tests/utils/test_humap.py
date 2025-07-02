"""Test integration with UMAP."""
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
import pickle
from pysp.stats import *
from pysp.utils.humap import humap
import numpy as np


def test_humap() -> None:
    """Test for humap using automatic fitting."""

    with open('pysp/tests/data/testInput_htsne.pkl', 'rb') as f:
        data = pickle.load(f)

    umap_kwargs = {
    'n_neighbors': 15,
    'min_dist': 0.2,
    'random_state': 42  # Set your desired seed here
    }

    embeddings, mix_model, fit, posteriors = humap(data, seed=10, umap_kwargs=umap_kwargs)

    with open("pysp/tests/answerkeys/testOutput_humap.pkl", "rb") as f:
        answer_dict = pickle.load(f)

    assert np.all(answer_dict['embeddings'] == embeddings)
    assert str(mix_model) == answer_dict['mix_model']
    assert str(fit) == answer_dict['fit']
    assert np.all(posteriors == answer_dict['posteriors'])