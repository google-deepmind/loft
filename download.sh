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
cd ${BASE_DIR}
mkdir -p retrieval/
cd retrieval

# Text retrieval datasets.
DATASETS=("arguana" "fever" "fiqa" "msmarco" "nq" "quora" "scifact" "webis_touche2020" "topiocqa" "hotpotqa" "musique" "qampari" "quest")
for DATASET in "${DATASETS[@]}"; do
  wget https://storage.googleapis.com/loft-bench/retrieval/${DATASET}.zip
  unzip ${DATASET}.zip
  rm ${DATASET}.zip
done
