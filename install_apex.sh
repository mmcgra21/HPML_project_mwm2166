#!/bin/bash

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
# # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# # otherwise
# # pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
# Need to comment out lines in setup.py or just change condition to False
sed -i '39,47s/^/#/' setup.py
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..