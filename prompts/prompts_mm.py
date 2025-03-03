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

"""Register prompts for multimodal retrieval."""

import functools
import os

from prompts import prompt_registry
from prompts import utils as prompt_utils
from prompts.constants import common as common_constants
from prompts.constants import mm as task_constants
import utils


PromptRegistry = prompt_registry.PromptRegistry
TASK = 'mm'

MULTIMODAL_RETRIEVAL_DATASETS = ('oven', 'msrvtt', 'flickr30k', 'mscoco')
RESOURCE_DIR = 'resources/'
LENGTHS = ('32k', '128k', '1m')
SPLITS = ('dev', 'test')

# Register few-shot prompts first.
# NOTE: This prompt will be used inside the other prompts, and not designed for
# direct use.
for length in LENGTHS:
  for dataset in MULTIMODAL_RETRIEVAL_DATASETS:
    # If there exists a mapper for the prompt, use it.
    prompt_name = task_constants.PROMPT_MAPPER.get(dataset, dataset)
    PromptRegistry.add(
        name=f'{TASK}_{dataset}_{length}:few_shot_examples',
        data_dir=f'{TASK}/{dataset}/{length}',
        cacheable_corpus=True,
        split='few_shot',
        data_loader=functools.partial(
            utils.load_data_from_file,
            resource_dir=os.path.join(RESOURCE_DIR, dataset),
        ),
        context_processors=[],  # No shared context is used.
        query_turn_processors=[
            functools.partial(
                prompt_utils.add_multimodal_query_turns,
                query_prefix_format=task_constants.FEW_SHOT_QUERY_FORMAT_PREFIX[
                    prompt_name
                ],
                query_suffix_format=task_constants.FEW_SHOT_QUERY_FORMAT_SUFFIX[
                    prompt_name
                ],
                use_example_id=True,
            ),
            functools.partial(
                prompt_utils.append_reasoning_to_query_turns,
                reasoning_format=task_constants.FEW_SHOT_EXAMPLE_ANSWER_FORMAT[
                    dataset
                ],
                qid2reasoning=None,
            ),
            functools.partial(
                prompt_utils.append_gold_answers_to_query_turns,
                answer_format=common_constants.FINAL_ANSWER_FORMAT,
            ),
        ],
        gold_answer_processors=[
            prompt_utils.convert_pids_into_gold_answers,
        ],
    )

for length in LENGTHS:
  for dataset in MULTIMODAL_RETRIEVAL_DATASETS:
    for split in SPLITS:
      prompt_name = task_constants.PROMPT_MAPPER.get(dataset, dataset)

      name = f'{TASK}_{dataset}_{length}_{split}:few_shot_with_cot'
      corpus_instruction = task_constants.CORPUS_INSTRUCTION[prompt_name]
      PromptRegistry.add(
          name=name,
          data_dir=f'{TASK}/{dataset}/{length}',
          split=split,
          cacheable_corpus=False,
          data_loader=functools.partial(
              utils.load_data_from_file,
              resource_dir=os.path.join(RESOURCE_DIR, dataset),
          ),
          context_processors=[
              functools.partial(
                  prompt_utils.add_text_chunks,
                  texts=[
                      corpus_instruction,
                  ],
              ),
              # Adds both corpus chunks and few-shot.
              functools.partial(
                  prompt_utils.add_corpus_chunks_and_query_turns_from_few_shot_examples,
                  corpus_format=task_constants.CORPUS_FORMAT[prompt_name],
                  shuffle_seed=None,
                  add_image_chunk=True,
                  few_shot_prompt_name=(
                      f'{TASK}_{length}_{dataset}:few_shot_examples'
                  ),
              ),
          ],
          query_turn_processors=[
              functools.partial(
                  prompt_utils.add_multimodal_query_turns,
                  use_example_id=False,
                  query_prefix_format=task_constants.QUERY_FORMAT_PREFIX[
                      prompt_name
                  ],
                  query_suffix_format=task_constants.QUERY_FORMAT_SUFFIX[
                      prompt_name
                  ],
              ),
          ],
          gold_answer_processors=[
              prompt_utils.convert_pids_into_gold_answers,
          ],
      )
