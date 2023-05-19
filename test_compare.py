from hypothesis import given, settings
from hypothesis.strategies import text, lists
from sentence_transformers import SentenceTransformer
import numpy as np
from run_onnx import DefaultEmbeddingModel


def _run_and_compare(texts):
    model = DefaultEmbeddingModel()
    result_onnx = model(texts)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    result_st = st_model.encode(texts)
    assert np.allclose(result_onnx, result_st, atol=1e-6)


@given(text())
@settings(deadline=1500)
def test_compare_single(text):
    _run_and_compare([text])


@given(lists(text(), min_size=1))
@settings(deadline=1500)
def test_compare_lists(texts):
    _run_and_compare(texts)


@given(lists(text(), min_size=50))
@settings(deadline=5000)
def test_compare_large_lists(texts):
    _run_and_compare(texts)


@given(text(min_size=100))
@settings(deadline=5000)
def test_compare_large_text(text):
    _run_and_compare([text])