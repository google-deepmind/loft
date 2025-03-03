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

"""Define PromptRegistry class to store prompts.

PromptRegistry is a class that stores prompts based on the LOFT data.
"""

from collections.abc import Callable, Sequence
import copy
import dataclasses
import random
from typing import Any, Optional
import utils

ContentChunk = utils.ContentChunk
QueryTurn = utils.QueryTurn
Example = utils.Example


@dataclasses.dataclass(frozen=True)
class LOFTPrompt:
  """Prompt for LOFT."""

  data_dir: str
  split: str
  data_loader: Callable[..., Any] = utils.load_data_from_file
  context_processors: Optional[Sequence[Callable[..., Any]]] = None
  query_turn_processors: Optional[Sequence[Callable[..., Any]]] = None
  gold_answer_processors: Optional[Sequence[Callable[..., Any]]] = None
  share_context: bool = True
  use_context_first: bool = True
  append_gold_answers_to_query_turns: bool = False
  is_multi_turn: bool = False
  # Whether the corpus is cacheable for that particular prompt.
  # Defaults to False since safer (always correct).
  cacheable_corpus: bool = False


class PromptRegistry:
  """Class to store prompts based on the LOFT data."""

  prompts = {}
  base_dir = ""

  @classmethod
  def add(
      cls,
      name: str,
      data_dir: str,
      split: str,
      data_loader: Callable[..., Any] = utils.load_data_from_file,
      context_processors: Optional[Sequence[Callable[..., Any]]] = None,
      query_turn_processors: Optional[Sequence[Callable[..., Any]]] = None,
      gold_answer_processors: Optional[Sequence[Callable[..., Any]]] = None,
      share_context: bool = True,
      use_context_first: bool = True,
      append_gold_answers_to_query_turns: bool = False,
      is_multi_turn: bool = False,
      cacheable_corpus: bool = False,
  ):
    """Adds a prompt to the registry."""
    if name in cls.prompts:
      raise ValueError(f"Prompt {name} already exists in the prompt registry.")

    cls.prompts[name] = LOFTPrompt(
        data_dir=data_dir,
        split=split,
        data_loader=data_loader,
        context_processors=context_processors,
        query_turn_processors=query_turn_processors,
        gold_answer_processors=gold_answer_processors,
        share_context=share_context,
        use_context_first=use_context_first,
        append_gold_answers_to_query_turns=append_gold_answers_to_query_turns,
        is_multi_turn=is_multi_turn,
        cacheable_corpus=cacheable_corpus,
    )

  @classmethod
  def get_examples(
      cls,
      name: str,
      base_dir: Optional[str] = None,
      max_examples: Optional[int] = None,
      loft_data: Optional[utils.LOFTData] = None,
      **kwargs,
  ) -> list[Example]:
    """Returns the examples for a given prompt name."""
    shuffle_queries = kwargs.get("shuffle_queries", False)

    if name not in cls.prompts:
      task_name = name.split("_")[0]
      registry_str = "\n".join(
          filter(
              lambda x: x.startswith(task_name),
              list(cls.prompts.keys()),
          )
      )
      raise ValueError(
          f"Prompt {name} not found in registry.\nAvailable"
          f" prompts:\n{registry_str}"
      )
    loft_prompt = cls.prompts[name]
    examples = []

    if loft_data is None:
      if not base_dir:
        base_dir = cls.base_dir
      else:
        cls.base_dir = base_dir
      # 1. Load the LOFT data if not provided.
      loft_data = loft_prompt.data_loader(
          data_dir=loft_prompt.data_dir,
          base_dir=base_dir,
          split=loft_prompt.split,
      )

    # 2. For each query, create an example.
    # NOTE: Each chunk (query or context) is added in-place to the list.
    context_chunks: list[ContentChunk] = []
    corpus_document_boundaries: list[tuple[int, int]] = []
    # Locates where a particular passage with a given pid is in the context.
    # Will be updated if fixed_pid_mapper is given for context processors.
    # NOTE: fixed_pid_mapper given to add_corpus_chunks can update pid_mapper.
    queries = list(loft_data.queries.keys())

    if shuffle_queries:
      random.shuffle(queries)

    pid_mapper: dict[str, str] = {
        pid: str(p_idx) for p_idx, pid in enumerate(loft_data.corpus)
    }
    loft_data.metadata["pid_mapper"] = pid_mapper

    for example_idx, qid in enumerate(queries):
      # Each query turn can be a list of query chunks (e.g. image + text).
      query_turns: list[list[ContentChunk]] = []
      gold_pids: list[Any] = []
      gold_answers: list[Any] = copy.deepcopy(loft_data.answers[qid])
      if not loft_prompt.is_multi_turn:
        if loft_data.metadata.values() and "qrels" in next(
            iter(loft_data.metadata.values())
        ):
          gold_pids = copy.deepcopy(
              [pid for pid, _ in loft_data.metadata[qid]["qrels"]]
          )
        else:
          if (
              isinstance(loft_data.answers[qid], list)
              and all([len(ans) == 2 for ans in loft_data.answers[qid]])
          ):
            gold_pids = copy.deepcopy(
                [pid for pid, _ in loft_data.answers[qid]]
            )
        # Make these as a single turn.
        gold_pids = [gold_pids]
        gold_answers = [gold_answers]
      else:
        if loft_data.metadata.values() and "qrels" in next(
            iter(loft_data.metadata.values())
        ):
          gold_pids = copy.deepcopy([
              [pid for pid, _ in qrels]
              for qrels in loft_data.metadata[qid]["qrels"]
          ])
          gold_answers = copy.deepcopy(
              [[gold_answer] for gold_answer in gold_answers]
          )

      # 2-1. Process the context chunks.
      if loft_prompt.context_processors:
        if (
            not loft_prompt.cacheable_corpus
            or not context_chunks
            or not loft_prompt.share_context
        ):
          context_chunks: list[ContentChunk] = []  # Reset context chunks.
          corpus_document_boundaries: list[tuple[int, int]] = (
              []
          )  # Reset corpus document boundaries.
          for context_processor in cls.prompts[name].context_processors:
            context_processor(
                chunks=context_chunks,
                pid_mapper=pid_mapper,
                gold_pids=gold_pids,
                loft_data=loft_data,
                qid=qid,
                corpus_document_boundaries=corpus_document_boundaries,
                **kwargs,
            )

      # 2-2. Process the gold answers.
      # NOTE: We need to process the gold answers first for the cases where we
      # need to append gold answers to query turns.
      if loft_prompt.gold_answer_processors:
        for gold_answer_processor in cls.prompts[name].gold_answer_processors:
          gold_answer_processor(
              query_turns=query_turns,
              gold_answers=gold_answers,
              loft_data=loft_data,
              qid=qid,
              pid_mapper=pid_mapper,
          )

      # 2-3. Process the query chunks.
      if loft_prompt.query_turn_processors:
        for query_turn_processor in cls.prompts[name].query_turn_processors:
          query_turn_processor(
              query_turns=query_turns,
              # Needed for the reasoning chain.
              gold_pids=gold_pids,
              gold_answers=gold_answers,
              loft_data=loft_data,
              qid=qid,
              example_id=str(example_idx + 1),
              pid_mapper=pid_mapper,
          )

      # 2-4. Create the example.
      examples.append(
          Example(
              qid=qid,
              num_turns=len(query_turns),
              context_chunks=context_chunks,
              # Lazily convert to QueryTurn objects.
              query_turns=[QueryTurn(chunks=chunks) for chunks in query_turns],
              gold_answers=gold_answers,
              gold_pids=gold_pids,
              corpus_document_boundaries=corpus_document_boundaries,
          )
      )
      if max_examples and len(examples) >= max_examples:
        break

    return examples
