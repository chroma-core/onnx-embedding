# A dependency-light way to run the onnx model

from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np

# Use pytorches default epsilon for division by zero 
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = 1e-12
    return v / norm

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = ort.InferenceSession("onnx/model.onnx")

# TODO: why is tokenizer capping at 128?
tokenizer = Tokenizer.from_pretrained(model_id)
tokenizer.enable_truncation(max_length=256)
# max_seq_length = 256
# https://github.com/UKPLab/sentence-transformers/blob/3e1929fddef16df94f8bc6e3b10598a98f46e62d/docs/_static/html/models_en_sentence_embeddings.html#LL480
# tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

sample_text = "This is a sample text"
encoded = tokenizer.encode(sample_text)
input_ids = np.array(encoded.ids)
attention_mask = np.array(encoded.attention_mask)
onnx_input = {
    "input_ids": np.array([input_ids], dtype=np.int64),
    "attention_mask": np.array([attention_mask], dtype=np.int64),
    "token_type_ids": np.array([np.zeros(len(input_ids), dtype=np.int64)], dtype=np.int64),
}
onnx_output = model.run(None, onnx_input)
# To see index of last hidden state
# print(model.get_outputs()[0].name)
last_hidden_state = onnx_output[0]
# Perform mean pooling with attention weighting
input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
# Normalize embeddings
embeddings = normalize(embeddings).astype(np.float32)