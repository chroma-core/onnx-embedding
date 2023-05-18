from hypothesis import given, settings
from hypothesis.strategies import text, lists
from sentence_transformers import SentenceTransformer
import numpy as np
from run_onnx import DefaultEmbeddingModel


@given(text())
@settings(deadline=1500)
def test_compare_single(text):
    model = DefaultEmbeddingModel()
    result_onnx = model([text])

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    result_st = st_model.encode([text])
    assert np.allclose(result_onnx, result_st, atol=1e-6)

@given(lists(text(), min_size=1))
@settings(deadline=1500)
def test_compare_lists(texts):
    model = DefaultEmbeddingModel()
    result_onnx = model(texts)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    result_st = st_model.encode(texts)
    assert np.allclose(result_onnx, result_st, atol=1e-6)