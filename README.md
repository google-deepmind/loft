# LOFT: A 1 Million+ Token Long-Context Benchmark

This repository houses the resources for LOFT, the Long Context Frontiers benchmark, introduced in the research paper [Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?]().
LOFT consists of 6 long-context task categories spanning retrieval, multi-hop compositional reasoning, and more, totaling 30+ datasets and 4 modalities.

In our initial release, we provide links to download many of the text datasets in LOFT.
We also provide an example prompt in `PROMPT_EXAMPLE.txt` showing how Corpus-in-Context (CiC) prompting can be done for the text retrieval task.

**Future Releases**

We're actively working on expanding this repository.
Upcoming releases will include:

* Code for Dataset Regeneration: Enabling a user to recreate the datasets in LOFT that we do not release.
* Task-Specific Evaluation Code: Providing code to measure LCLMs' performance on each LOFT task.

**Releases**

* [6/20/24]: Initial release with links to download many of the LOFT text datasets.

## Installation

None for now.

## Datasets

| Task | Dataset | Description | Public? | Download |
| :---: | :---: | :---: | :---: | :---: |
| Text Retrieval | [ArguAna](https://github.com/beir-cellar/beir) | Argument Retrieval | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/arguana.zip) |
| Text Retrieval | [FEVER](https://github.com/beir-cellar/beir) | Fact Checking | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/fever.zip) |
| Text Retrieval | [FIQA](https://github.com/beir-cellar/beir) | Question Answering | ❌ | No |
| Text Retrieval | [MS MARCO](https://github.com/beir-cellar/beir) | Web Search | ❌ | No | 
| Text Retrieval | [NQ](https://github.com/beir-cellar/beir) | Question Answering | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/nq.zip) | 
| Text Retrieval | [Quora](https://github.com/beir-cellar/beir) | Duplication Detection | ❌ | No | 
| Text Retrieval | [SciFact](https://github.com/beir-cellar/beir) | Citation Prediction | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/scifact.zip) | 
| Text Retrieval | [Touché-2020](https://github.com/beir-cellar/beir) | Argument Retrieval | ❌ | No | 
| Text Retrieval | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | ❌ | No |
| Text Retrieval | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA| ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/hotpotqa.zip) |
| Text Retrieval | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/musique.zip) |
| Text Retrieval | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/qampari.zip) |
| Text Retrieval | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/quest.zip) |
| Visual Retrieval | [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) | Image Retrieval | ❌ | No |
| Visual Retrieval | [MS COCO](https://cocodataset.org) | Image Retrieval | ❌ | No |
| Visual Retrieval | [OVEN](https://github.com/open-vision-language/oven) | Image-text Retrieval | ✅ | Coming Soon |
| Visual Retrieval | [MSR-VTT](https://cove.thecvf.com/datasets/839) | Video Retrieval | ❌ | No |
| Audio Retrieval | [FLEURS-en](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | ✅ | Coming Soon |
| Audio Retrieval | [FLEURS-es](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | ✅ | Coming Soon |
| Audio Retrieval | [FLEURS-fr](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | ✅ | Coming Soon |
| Audio Retrieval | [FLEURS-hi](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | ✅ | Coming Soon |
| Audio Retrieval | [FLEURS-zh](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | ✅ | Coming Soon |
| RAG | [NQ](https://github.com/beir-cellar/beir) | Question Answering | ✅ | [Link](https://storage.googleapis.com/loft-bench/rag/nq.zip) |
| RAG | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | ❌ | No |
| RAG | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA| ✅ | [Link](https://storage.googleapis.com/loft-bench/rag/hotpotqa.zip) |
| RAG | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/rag/musique.zip) |
| RAG | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/rag/qampari.zip) |
| RAG | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | ✅ | [Link](https://storage.googleapis.com/loft-bench/rag/quest.zip) |
| SQL | [Spider](https://yale-lily.github.io/spider) | Single-turn SQL | ✅ | [Link](https://storage.googleapis.com/loft-bench/sql/spider.zip) |
| SQL | [SParC](https://yale-lily.github.io/sparc) | Multi-turn SQL | ✅ | [Link](https://storage.googleapis.com/loft-bench/sql/sparc.zip) |
| Many-Shot ICL | [BBH-date](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | ✅ | Coming Soon |
| Many-Shot ICL | [BBH-salient](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | ✅ | Coming Soon |
| Many-Shot ICL | [BBH-tracking7](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | ✅ | Coming Soon |
| Many-Shot ICL | [BBH-web](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | ✅ | Coming Soon |
| Many-Shot ICL | [LIB-dialogue](https://github.com/TIGER-AI-Lab/LongICLBench) | Classification | ❌ | No |

## Citing this work

```latex
@article{Lee2024LongContext,
  title={Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?},
  author={Jinhyuk Lee and Anthony Chen and Zhuyun Dai and Dheeru Dua and Devendra Singh Sachan and Michael Boratko and Yi Luan and Sébastien M. R. Arnold and Vincent Perot and Siddharth Dalmia and Hexiang Hu and Xudong Lin and Panupong Pasupat and Aida Amini and Jeremy R. Cole and Sebastian Riedel and Iftekhar Naim and Ming-Wei Chang and Kelvin Guu},
  journal={ArXiv},
  year={2024},
  volume={},
  url={}
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

Individual tasks may be subject to copyright and licensing from their respective owners - please see individual download files for details.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
