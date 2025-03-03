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
LENGTH="128k"
TASK_TYPE="retrieval"
SPLIT="dev"
if [[ ${TASK_TYPE} == "icl" ]]; then
  PROMPT_TYPE="many_shot"  # Use "many_shot" for ICL.
else
  PROMPT_TYPE="few_shot_with_cot"
fi
PROMPT="${TASK_TYPE}_${DATASET}_${LENGTH}_${SPLIT}:${PROMPT_TYPE}"
echo "Prompt: ${PROMPT}"

mkdir -p ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}

answer_file_extension="jsonl"
if [[ ${TASK_TYPE} == "icl" ]]; then
  answer_file_extension="json"
fi

python run_inference.py \
    --prompt_name ${PROMPT} \
    --task_type ${TASK_TYPE} \
    --base_dir ${BASE_DIR} \
    --data_dir ${TASK_TYPE}/${DATASET}/${LENGTH} \
    --split ${SPLIT} \
    --context_length ${LENGTH} \
    --output_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --project_id ${PROJECT_ID} \
    --overwrite

python run_evaluation.py \
    --answer_file_path ${BASE_DIR}/data/${TASK_TYPE}/${DATASET}/${LENGTH}/dev_queries.${answer_file_extension} \
    --pred_file_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --task_type ${TASK_TYPE}
