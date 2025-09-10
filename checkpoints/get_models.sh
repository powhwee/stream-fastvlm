#!/usr/bin/env bash

wget https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip 

# Extract models
unzip -qq llava-fastvithd_0.5b_stage3.zip

# Clean up
rm llava-fastvithd_0.5b_stage3.zip

