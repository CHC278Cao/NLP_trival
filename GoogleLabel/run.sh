#!/usr/bin/env bash

export TRAIN_FILE=data/train.csv
export TEST_FILE=data/test.csv
export SUBMISSION=data/sample_submission.csv
export MODE=$1
export MODEL_PATH=$2

python -m model.train