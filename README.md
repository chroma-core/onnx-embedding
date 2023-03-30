# ONNX Embedding Model

This repo houses chromadbs code to generate the "sentence-transformers/all-MiniLM-L6-v2" model into onnx as well as reference code for how to run it. 

The model is stored on S3 and chromadb will fetch/cache it from there. 

We do this because sentence-transformers introduces a lot of transitive dependencies that we don't want to have to install in the chromadb and some of those also don't work on newer python versions.

NOTE: We do not plan to support more than one default model in the near future in this way. If you want to use more models you should use chromadbs other embedding functions which depend on libraries like sentence-transformers.

## Running the example model

```
pip install -r requirements.txt
```

and then

```
python run_onnx.py
```

The requirements in requirements.txt are the minimum requirements to run the model.

## Generating the model

```
pip install -r requirements-dev.txt
```

and then

```
python generate_onnx.py
```

## Validating the model implementation is correct

```
pip install -r requirements-dev.txt
```

and then

```
python compare_onnx.py
```

This will compare the output of the onnx model to the output of the sentence-transformers model by evaluating the glue stsb benchmark as well as looking at the cosine similarity of the embeddings for the dataset.
