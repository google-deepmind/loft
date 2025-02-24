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
TASK_TYPE=retrieval
LENGTH=128k
SPLIT=dev
mkdir -p ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}

python run_inference.py \
    --prompt_prefix_path ${BASE_DIR}/prompts/${TASK_TYPE}_${LENGTH}/${TASK_TYPE}_${DATASET}_${LENGTH}.txt \
    --data_dir ${BASE_DIR}/data/${TASK_TYPE}/${DATASET}/${LENGTH} \
    --split ${SPLIT} \
    --context_length ${LENGTH} \
    --output_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --project_id ${PROJECT_ID}

python run_evaluation.py \
    --answer_file_path ${BASE_DIR}/data/${TASK_TYPE}/${DATASET}/${LENGTH}/dev_queries.jsonl \
    --pred_file_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --task_type ${TASK_TYPE}
