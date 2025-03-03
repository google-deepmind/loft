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

"""Register prompts for SQL."""

import functools
from prompts import prompt_registry
from prompts import utils as prompt_utils
from prompts.constants import sql as task_constants


PromptRegistry = prompt_registry.PromptRegistry
TASK = 'sql'

SQL_DATASETS = (
    'spider',
    'sparc',
)
LENGTHS = ('32k', '128k', '1m')
SPLITS = ('dev', 'test')

# Few-shot examples are directly provided as a list of strings in SQL.
for length in LENGTHS:
  for dataset in SQL_DATASETS:
    for split in SPLITS:
      prompt_name = task_constants.PROMPT_MAPPER.get(dataset, dataset)
      PromptRegistry.add(
          name=f'{TASK}_{dataset}_{length}_{split}:few_shot_with_cot',
          data_dir=f'{TASK}/{dataset}/{length}',
          split=split,
          cacheable_corpus=True,
          context_processors=[
              functools.partial(
                  prompt_utils.add_text_chunks,
                  texts=[
                      task_constants.CORPUS_INSTRUCTION[prompt_name],
                  ],
              ),
              # Add few-shot examples.
              functools.partial(
                  prompt_utils.add_text_chunks,
                  texts=task_constants.FEW_SHOT_EXAMPLES_V1[prompt_name],
              ),
          ],
          query_turn_processors=[
              functools.partial(
                  prompt_utils.add_query_turns_with_corpus,
                  query_format=task_constants.QUERY_FORMAT_0[prompt_name],
                  corpus_format=task_constants.CORPUS_FORMAT[prompt_name],
                  follow_up_query_format=task_constants.FOLLOW_UP_QUERY_FORMAT_0[
                      prompt_name
                  ],
              ),
          ],
          gold_answer_processors=[],
      )
