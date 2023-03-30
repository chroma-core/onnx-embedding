from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

SAVE_PATH = Path("onnx")

model_id = "sentence-transformers/all-MiniLM-L6-v2"

# load vanilla transformers and convert to onnx
model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
