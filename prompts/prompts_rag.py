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

"""Register prompts for RAG."""

import functools
from prompts import prompt_registry
from prompts import utils as prompt_utils
from prompts.constants import common as common_constants
from prompts.constants import rag as task_constants


PromptRegistry = prompt_registry.PromptRegistry
TASK = 'rag'

RAG_DATASETS = (
    'nq',
    'hotpotqa',
    'musique',
    'qampari',
    'quest',
    'topiocqa',
    'romqa',  # cannot be used for training
)
LENGTHS = ('32k', '128k', '1m')
SPLITS = ('dev', 'test')


# get_few_shot_reasoning_format among rag and retrieval.
def get_query_format(dataset_name: str, few_shot_type_name: str) -> str:
  """Get query format str for the given dataset and few-shot type."""

  p = task_constants.PROMPT_MAPPER.get(dataset_name, dataset_name)
  if few_shot_type_name == 'few_shot_with_cot':
    return task_constants.QUERY_FORMAT_SIMPLE_REVERSE[p]
  else:
    raise ValueError(f'Unsupported few-shot type: {few_shot_type_name}')


def get_few_shot_reasoning_format(
    dataset_name: str, few_shot_type_name: str
) -> str | None:
  """Get few-shot reasoning format str for the given dataset and few-shot type."""
  p = task_constants.PROMPT_MAPPER.get(dataset_name, dataset_name)
  if few_shot_type_name == 'few_shot_with_cot':
    return task_constants.FEW_SHOT_EXAMPLE_COT_FORMAT_SIMPLE_REVERSE[p]
  else:
    raise ValueError(f'Unsupported few-shot type: {few_shot_type_name}')


# Register few-shot prompts first.
# NOTE: This prompt will be used inside the other prompts, and not designed for
# direct use.
for length in LENGTHS:
  for dataset in RAG_DATASETS:
    # Processors for the few-shot examples.
    query_turn_processors = [
        # Format few-shot example's query.
        functools.partial(
            prompt_utils.add_query_turns,
            query_format=common_constants.FEW_SHOT_SEPARATOR
            + get_query_format(
                dataset_name=dataset, few_shot_type_name='few_shot_with_cot'
            ),
            use_example_id=True,
        ),
    ]
    query_turn_processors.append(
        functools.partial(
            prompt_utils.append_reasoning_to_query_turns,
            reasoning_format=get_few_shot_reasoning_format(
                dataset_name=dataset, few_shot_type_name='few_shot_with_cot'
            ),
            qid2reasoning=None,
        ),
    )
    query_turn_processors.append(
        # Format few-shot example's answer.
        functools.partial(
            prompt_utils.append_gold_answers_to_query_turns,
            answer_format=common_constants.FINAL_ANSWER_FORMAT,
        ),
    )

    PromptRegistry.add(
        name=f'{TASK}_{dataset}_{length}:few_shot_examples',
        data_dir=f'{TASK}/{dataset}/{length}',
        cacheable_corpus=True,
        split='few_shot',
        is_multi_turn=dataset == 'topiocqa',
        context_processors=[],  # No shared context is used.
        query_turn_processors=query_turn_processors,
        gold_answer_processors=[],
    )

# Adding few-shot prompts.
for length in LENGTHS:
  for dataset in RAG_DATASETS:
    for split in SPLITS:
      prompt_name = task_constants.PROMPT_MAPPER.get(dataset, dataset)
      corpus_instruction = task_constants.CORPUS_INSTRUCTION[
          prompt_name
      ]
      corpus_format = task_constants.CORPUS_FORMAT_ECHO[prompt_name]
      name = f'{TASK}_{dataset}_{length}_{split}:few_shot_with_cot'

      test_query_processors = [
          functools.partial(
              prompt_utils.add_query_turns,
              query_format=common_constants.TEST_QUERY_SEPARATOR
              + get_query_format(
                  dataset_name=dataset,
                  few_shot_type_name='few_shot_with_cot',
              ),
          )
      ]
      PromptRegistry.add(
          name=name,
          data_dir=f'{TASK}/{dataset}/{length}',
          split=split,
          cacheable_corpus=False,
          is_multi_turn=dataset == 'topiocqa',
          context_processors=[
              functools.partial(
                  prompt_utils.add_text_chunks,
                  texts=[
                      corpus_instruction,
                      task_constants.FORMATTING_INSTRUCTION,
                  ],
              ),
              # Adds both corpus chunks and few-shot.
              functools.partial(
                  prompt_utils.add_corpus_chunks_and_query_turns_from_few_shot_examples,
                  corpus_format=corpus_format,
                  shuffle_seed=None,
                  few_shot_prompt_name=(
                      f'{TASK}_{dataset}_{length}:few_shot_examples'
                  ),
              ),
          ],
          query_turn_processors=test_query_processors,
          gold_answer_processors=[],
      )
