# Compare with using the sentence-transformers library
from datasets import load_dataset
from evaluate import load
from sentence_transformers import SentenceTransformer
import numpy as np
from run_onnx import DefaultEmbeddingModel

# Benchmark dataset to see if the embeddings are the same
# https://huggingface.co/datasets/glue
eval_dataset = load_dataset("glue", "stsb", split="validation")
metric = load("glue", "stsb")
# eval_dataset = eval_dataset.select(range(100))

st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
onnx_model = DefaultEmbeddingModel()


def evaluate_stsb(example):
    st_s1 = st_model.encode([example["sentence1"]])
    st_s2 = st_model.encode([example["sentence2"]])
    st_result = np.dot(st_s1[0], st_s2[0])
    onnx_s1 = onnx_model([example["sentence1"]])
    onnx_s2 = onnx_model([example["sentence2"]])
    onnx_result = np.dot(onnx_s1[0], onnx_s2[0])
    return {
        "reference": (example["label"] - 1) / (5 - 1),  # rescale to [0,1]
        "sentence_transformers": float(st_result),
        "onnx": float(onnx_result),
    }


results = eval_dataset.map(evaluate_stsb)
onnx_results = metric.compute(
    predictions=results["onnx"], references=results["reference"]
)
st_results = metric.compute(
    predictions=results["sentence_transformers"], references=results["reference"]
)

print(f"ONNX results: {onnx_results['pearson']}")
print(f"ST results: {st_results['pearson']}")

# Compare by dot product of embeddings
def compare_embeddings(example):
    onnx_s1 = onnx_model([example["sentence1"]])
    st_s1 = onnx_model([example["sentence1"]])
    onnx_result = np.dot(onnx_s1[0], st_s1[0])
    return {
        "onnx": float(onnx_result),
    }


results = eval_dataset.map(compare_embeddings)
# Make sure the embeddings are the same
EPS = 1e-6
similar_count = 0
for result in results:
    if abs(result["onnx"] - 1) < EPS:
        similar_count += 1
print(f"Similar embeddings: {similar_count/len(results)*100}%")
