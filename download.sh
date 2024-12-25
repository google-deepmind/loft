#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
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
ORIGINAL_DIR=$(pwd)
mkdir -p ${BASE_DIR}

# Text retrieval datasets.
cd ${BASE_DIR}
mkdir -p data/retrieval/
cd data/retrieval
DATASETS=("arguana" "fever" "fiqa" "msmarco" "nq" "quora" "scifact" "webis_touche2020" "topiocqa" "hotpotqa" "musique" "qampari" "quest")
for DATASET in "${DATASETS[@]}"; do
  wget https://storage.googleapis.com/loft-bench/retrieval/${DATASET}.zip
  unzip ${DATASET}.zip
  rm ${DATASET}.zip
done
cd ${ORIGINAL_DIR}

# Text RAG datasets.
cd ${BASE_DIR}
mkdir -p data/rag/
cd data/rag
DATASETS=("nq" "hotpotqa" "musique" "qampari" "quest")
for DATASET in "${DATASETS[@]}"; do
  wget https://storage.googleapis.com/loft-bench/rag/${DATASET}.zip
  unzip ${DATASET}.zip
  rm ${DATASET}.zip
done
cd ${ORIGINAL_DIR}

# SQL datasets.
cd ${BASE_DIR}
mkdir -p data/sql/
cd data/sql
DATASETS=("spider" "sparc")
for DATASET in "${DATASETS[@]}"; do
  wget https://storage.googleapis.com/loft-bench/sql/${DATASET}.zip
  unzip ${DATASET}.zip
  rm ${DATASET}.zip
done
cd ${ORIGINAL_DIR}

# ICL datasets.
cd ${BASE_DIR}
mkdir -p data/icl/
cd data/icl
DATASETS=("date_understanding" "salient_translation_error_detection" "tracking_shuffled_objects_seven_objects" "web_of_lies")
for DATASET in "${DATASETS[@]}"; do
  wget https://storage.googleapis.com/loft-bench/icl/${DATASET}.zip
  unzip ${DATASET}.zip
  rm ${DATASET}.zip
done

# Preprocess and fill in required fields.
cd ${ORIGINAL_DIR}
DATASETS=("fiqa" "msmarco" "quora" "webis_touche2020")
for DATASET in "${DATASETS[@]}"; do
  python preprocess.py \
    --input_dir ${BASE_DIR}/data/retrieval/${DATASET} \
    --dataset ${DATASET}
done

# Sample retrieval 128k prompts.
cd ${BASE_DIR}
mkdir -p prompts/
cd prompts
wget https://storage.googleapis.com/loft-bench/prompts/retrieval_128k.zip
unzip retrieval_128k.zip
rm retrieval_128k.zip
