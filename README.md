# LOFT: A 1 Million+ Token Long-Context Benchmark

This repository houses the resources for LOFT, the Long Context Frontiers benchmark, introduced in the research paper [Can Long-Context Language Models Subsume Retrieval, SQL, and More?]().
LOFT consists of 6 long-context task categories spanning retrieval, multi-hop compositional reasoning, and more, totaling 30+ datasets and 4 modalities.

**Initial Release**
We provide access to a subset of the LOFT data (NQ and QUEST) to provide a hands-on experience and a feel for the types of tasks involved.
We also provide an example prompt in `PROMPT_EXAMPLE.txt` showing an example of how prompting can be done for the text retrieval task.

**Future Releases**
We're actively working on expanding this repository.
Upcoming releases will include:

* Code for Dataset Regeneration: Enabling a user to recreate the full range of LOFT datasets.
* Task-Specific Evaluation Code: Providing code to measure LCLMs' performance on each LOFT task.

**Updates**

* [6/18/24]: Initial release with links to a portion of LOFT data subsets.

## Installation

None for now.

## Usage

Run `wget https://storage.googleapis.com/loft-bench/loft_oss.zip` to pull the available LOFT data.

## Citing this work

```latex
@article{Lee2024LongContext,
  title={Can Long-Context Language Models Subsume Retrieval, SQL, and More?},
  author={Jinhyuk Lee and Anthony Chen and Zhuyun Dai and Dheeru Dua and Devendra Singh Sachan and Michael Boratko and Yi Luan and Séb Arnold and Vincent Perot and Siddharth Dalmia and Hexiang Hu and Xudong Lin and Panupong Pasupat and Aida Amini and Jeremy R. Cole and Sebastian Riedel and Iftekhar Naim and Ming-Wei Chang and Kelvin Guu},
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

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
