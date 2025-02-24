#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

BASE_DIR=$1
DATASET=$2

python run_inference.py \
    --prompt_prefix_path ${BASE_DIR}/prompts/retrieval_128k/retrieval_${DATASET}_128k.txt \
    --data_dir ${BASE_DIR}/data/retrieval/${DATASET}/128k \
    --split dev \
    --context_length 128k \
    --output_path ${BASE_DIR}/outputs/retrieval/${DATASET}/128k/predictions.jsonl \
    --project_id ${PROJECT_ID}

python run_evaluation.py \
    --answer_file_path ${BASE_DIR}/data/retrieval/${DATASET}/128k/dev_queries.jsonl \
    --pred_file_path ${BASE_DIR}/outputs/retrieval/${DATASET}/128k/predictions.jsonl \
    --task_type retrieval
