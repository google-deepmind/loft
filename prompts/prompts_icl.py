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

"""Register prompts for Many-shot ICL."""

import functools

from prompts import prompt_registry
from prompts import utils as prompt_utils
from prompts.constants import icl as task_constants
import utils


PromptRegistry = prompt_registry.PromptRegistry
TASK = 'icl'

BBH_TASK_LENGTH = {
    'date_understanding': 23578,
    'salient_translation_error_detection': 28299,
    'tracking_shuffled_objects_seven_objects': 28360,
    'web_of_lies': 10380,
}

LENGTHS = ('32k', '128k', '1m')
SPLITS = ('dev', 'test')
LENGTH_TO_MAX_TOKENS = {
    '2k': 2_000,
    '4k': 4_000,
    '8k': 8_000,
    '16k': 16_000,
    '32k': 32_000,
    '128k': 128_000,
    '200k': 200_000,
    '1m': 1_000_000,
}

dataset = 'bbh'
for length in ['32k', '16k', '8k', '4k', '2k']:
  for subtask_name, subtask_length in BBH_TASK_LENGTH.items():
    data_dir = f'{TASK}/{subtask_name}/{length}'
    max_tokens = LENGTH_TO_MAX_TOKENS[length]
    dataset_name = f'{subtask_name}'
    for split in SPLITS:
      PromptRegistry.add(
          name=f'{TASK}_{dataset_name}_{length}_{split}:many_shot',
          data_dir=data_dir,
          split=split,
          cacheable_corpus=True,
          data_loader=utils.load_bbh_data_from_file,
          context_processors=[
              functools.partial(
                  prompt_utils.add_text_chunks,
                  texts=[
                      task_constants.CORPUS_INSTRUCTION[dataset],
                  ],
              ),
              functools.partial(
                  prompt_utils.add_many_shot_chunks,
                  chunk_format_fn=prompt_utils.get_bbh_example_chunk,
                  corpus_format=task_constants.CORPUS_FORMAT[dataset],
              ),
          ],
          query_turn_processors=[
              functools.partial(
                  prompt_utils.add_query_turns_for_many_shot,
                  chunk_format_fn=prompt_utils.get_bbh_example_chunk,
                  corpus_format=task_constants.CORPUS_FORMAT[dataset],
              ),
          ],
          gold_answer_processors=[],
      )

dataset = 'long_icl_bench_dialogue_re'
for length in LENGTHS:
  data_dir = f'{TASK}/{dataset}/{length}'
  dataset_name = dataset
  for split in SPLITS:
    PromptRegistry.add(
        name=f'{TASK}_{dataset_name}_{length}_{split}:many_shot',
        data_dir=data_dir,
        split=split,
        cacheable_corpus=True,
        data_loader=functools.partial(
            utils.load_long_icl_bench_dialogue_re_data_from_file,
        ),
        context_processors=[
            functools.partial(
                prompt_utils.add_text_chunks,
                texts=[
                    task_constants.CORPUS_INSTRUCTION[dataset],
                ],
            ),
            functools.partial(
                prompt_utils.add_many_shot_chunks,
                chunk_format_fn=prompt_utils.get_long_icl_bench_dialogue_re_example_chunk,
                corpus_format=task_constants.CORPUS_FORMAT[dataset],
                target_proposition=task_constants.TARGET_PROPOSITION,
                ending_notation=task_constants.ENDING_NOTATION,
            ),
        ],
        query_turn_processors=[
            functools.partial(
                prompt_utils.add_query_turns_for_many_shot,
                chunk_format_fn=prompt_utils.get_long_icl_bench_dialogue_re_example_chunk,
                corpus_format=task_constants.CORPUS_FORMAT[dataset],
                target_proposition=task_constants.TARGET_PROPOSITION,
                ending_notation=task_constants.ENDING_NOTATION,
            ),
        ],
        gold_answer_processors=[],
    )
