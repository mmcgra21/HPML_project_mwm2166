#!/bin/bash

# Install Fairseq
pip install fairseq==0.10.2 sacremoses
# Install LightSeq
pip install lightseq
# Uninstall cublas because causes errors
pip uninstall nvidia_cublas_cu11 -y
