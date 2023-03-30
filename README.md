# ONNX Embedding Model

This repo houses our code to generate the "sentence-transformers/all-MiniLM-L6-v2" model into onnx as well as reference code for how to run it.

We will store the model on S3 and chromadb will fetch/cache it from there. 

We do this because sentence-transformers introduces a lot of transitive dependencies that we don't want to have to install in the chromadd and some of those also don't work on newer python versions.