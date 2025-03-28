# LOFT: A 1 Million+ Token Long-Context Benchmark

This repository houses the resources for LOFT, the Long Context Frontiers benchmark, introduced in the research paper [Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?](https://arxiv.org/abs/2406.13121).
LOFT consists of 6 long-context task categories spanning retrieval, multi-hop
compositional reasoning, and more, totaling 35 datasets and 4 modalities.

## Installation
```bash
$ git clone git@github.com:google-deepmind/loft.git
$ cd loft/
$ pip install -r requirements.txt
```

## Download Datasets and Prompts
The script below downloads all the LOFT datasets under `BASE_DIR`.

```bash
$ BASE_DIR=your-choice-of-directory
$ sh download.sh $BASE_DIR
```

Each dataset is also available from the links in the [Datasets](#datasets) table.
For a small subset, `download.sh` will additionally run `preprocess.py`, which
infills the missing fields in the queries and corpus files.
Once the download is completed, you will see the file structure as below:

```
$BASE_DIR
└── data
     ├── retrieval
     │   ├── arguana
     │   │   ├── 128k
     │   │   │   ├── corpus.jsonl
     │   │   │   ├── dev_queries.jsonl
     │   │   │   ├── few_shot_queries.jsonl
     │   │   │   └── test_queries.jsonl
     │   │   ├── 1m
     │   │   └── 32k
     │   ├── fever
     │   │   ├── ...
     │   ├── ...
     ├── rag
     ├── sql
     ├── icl
     └── mm
```

The `data` folder contains the LOFT datasets and the `prompts` folder contains
samples of prompts used in LOFT.
We also provide an example prompt in `PROMPT_EXAMPLE.txt` showing how
Corpus-in-Context (CiC) prompting can be done for the text retrieval task.

## Inference and Evaluation
We currently support using `gemini-1.5-flash-002` from VertexAI for inference.
Please prepare your `PROJECT_ID` from [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#expandable-1).
To run the inference with `gemini-1.5-flash-002` and evaluate predictions:

```bash
BASE_DIR=$1
DATASET=$2
LENGTH="128k"
TASK_TYPE="retrieval"
SPLIT="dev"
PROMPT_TYPE="few_shot_with_cot"
PROMPT="${TASK_TYPE}_${DATASET}_${LENGTH}_${SPLIT}:${PROMPT_TYPE}"
echo "Prompt: ${PROMPT}"

mkdir -p ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}
answer_file_extension="jsonl"

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
```

The same script can be found from `infer_eval.sh`.
We provide example queries and predictions files in [evaluation/example_predictions/](evaluation/example_predictions/).
Each `task_type` outputs many different metric scores.
To understand which `task_type` to use for each dataset and also to see the primary evaluation metric reported in the paper for each dataset, see the [Datasets](#datasets) table.

## Datasets

| Task | Dataset | Description | Task Type | Primary Metric | Infilling Needed? | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Text Retrieval | [ArguAna](https://github.com/beir-cellar/beir) | Argument Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/arguana.zip) |
| Text Retrieval | [FEVER](https://github.com/beir-cellar/beir) | Fact Checking | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/fever.zip) |
| Text Retrieval | [FIQA](https://github.com/beir-cellar/beir) | Question Answering | `retrieval` | `recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/fiqa.zip) |
| Text Retrieval | [MS MARCO](https://github.com/beir-cellar/beir) | Web Search | `retrieval` |`recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/msmarco.zip) |
| Text Retrieval | [NQ](https://github.com/beir-cellar/beir) | Question Answering | `retrieval` |`recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/nq.zip) |
| Text Retrieval | [Quora](https://github.com/beir-cellar/beir) | Duplication Detection | `retrieval` |`recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/quora.zip) |
| Text Retrieval | [SciFact](https://github.com/beir-cellar/beir) | Citation Prediction | `retrieval` |`recall@1`  | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/scifact.zip) |
| Text Retrieval | [Touché-2020](https://github.com/beir-cellar/beir) | Argument Retrieval | `retrieval` | `recall@1`  | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/webis_touche2020.zip) |
| Text Retrieval | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | `retrieval` |`recall@1`  | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/topiocqa.zip) |
| Text Retrieval | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA | `retrieval` | `mrecall@2` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/hotpotqa.zip) |
| Text Retrieval | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | `retrieval` | `mrecall@5` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/musique.zip) |
| Text Retrieval | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | `retrieval` |  `mrecall@5` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/qampari.zip) |
| Text Retrieval | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | `retrieval` | `mrecall@3` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/quest.zip) |
| Visual Retrieval | [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) | Image Retrieval | `retrieval` | `recall@1` |✅ | Coming Soon |
| Visual Retrieval | [MS COCO](https://cocodataset.org) | Image Retrieval | `retrieval` | `recall@1` |✅ | Coming Soon |
| Visual Retrieval | [OVEN](https://github.com/open-vision-language/oven) | Image-text Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/mm/oven.zip) |
| Visual Retrieval | [MSR-VTT](https://cove.thecvf.com/datasets/839) | Video Retrieval | `retrieval` | `recall@1`| ✅ | Coming Soon |
| Audio Retrieval | [FLEURS-en](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/mm/fleurs_en_tts.zip) |
| Audio Retrieval | [FLEURS-es](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/mm/fleurs_es_tts.zip) |
| Audio Retrieval | [FLEURS-fr](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1`| - | [Link](https://storage.googleapis.com/loft-bench/mm/fleurs_fr_tts.zip) |
| Audio Retrieval | [FLEURS-hi](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/mm/fleurs_hi_tts.zip) |
| Audio Retrieval | [FLEURS-zh](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/mm/fleurs_zh_tts.zip) |
| RAG | [NQ](https://github.com/beir-cellar/beir) | Question Answering | `rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/nq.zip) |
| RAG | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | `rag` |  `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/topiocqa.zip) |
| RAG | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA | `rag` |  `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/hotpotqa.zip) |
| RAG | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | `rag` |  `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/musique.zip) |
| RAG | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | `multi_value_rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/qampari.zip) |
| RAG | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | `multi_value_rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/quest.zip) |
| SQL | [Spider](https://yale-lily.github.io/spider) | Single-turn SQL | `sql` | `exec_acc` | - | [Link](https://storage.googleapis.com/loft-bench/sql/spider.zip) |
| SQL | [SParC](https://yale-lily.github.io/sparc) | Multi-turn SQL | `sql` | `exec_acc` | - | [Link](https://storage.googleapis.com/loft-bench/sql/sparc.zip) |
| Many-Shot ICL | [BBH-date](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/date_understanding.zip) |
| Many-Shot ICL |[BBH-salient](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/salient_translation_error_detection.zip) |
| Many-Shot ICL |[BBH-tracking7](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/tracking_shuffled_objects_seven_objects.zip) |
| Many-Shot ICL |[BBH-web](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/web_of_lies.zip) |
| Many-Shot ICL |[LIB-dialogue](https://github.com/TIGER-AI-Lab/LongICLBench) | Classification | - | - | ✅ | Coming Soon |

## LOFT-Hard Subset
From the experiments in our [paper](https://arxiv.org/abs/2406.13121), we
learned that Gemini 1.5 was already performing well on many LOFT datasets, but
also it showed some headroom on other datasets.
Hence, we recommend iterating on the following four datasets:

* **MuSiQue, QAMPARI, QUEST, Spider**

Full datasets and inference are supported from the current OSS.

## Past & Upcoming Releases

* [ ] Remaining multi-modal data and inference.
* [x] Prompt conversion code (data => prompt).
* [x] Inference code and prompts for retrieval (10/25/24).
* [x] Evaluation code for ICL and some ICL and visual retrieval datasets (8/30/24).
* [x] Evaluation code for text tasks and code to regenerate some of the LOFT datasets (6/29/24).
* [x] Initial release with links to download many of the LOFT text datasets (6/20/24).

## Citing this work

```
@article{Lee2024LongContext,
  title={Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?},
  author={Jinhyuk Lee and Anthony Chen and Zhuyun Dai and Dheeru Dua and Devendra Singh Sachan and Michael Boratko and Yi Luan and Sébastien M. R. Arnold and Vincent Perot and Siddharth Dalmia and Hexiang Hu and Xudong Lin and Panupong Pasupat and Aida Amini and Jeremy R. Cole and Sebastian Riedel and Iftekhar Naim and Ming-Wei Chang and Kelvin Guu},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.13121},
  url={https://arxiv.org/abs/2406.13121}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Individual tasks may be subject to copyright and licensing from their respective
owners - please see individual download files for details.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
