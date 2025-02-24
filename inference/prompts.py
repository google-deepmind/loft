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

import copy
import dataclasses
from typing import Any, Callable, Optional

from inference import utils


ContentChunk = utils.ContentChunk
QueryTurn = utils.QueryTurn
Example = utils.Example
MimeType = utils.MimeType


@dataclasses.dataclass(frozen=True)
class CiCPrompt:
  """Prompt for LOFT."""

  data_dir: str
  split: str
  prefix: str
  data_loader: Callable[..., Any] = utils.load_data_from_file
  is_multi_turn: bool = False


class PromptRegistry:
  """Class to store prompts based on the LOFT data."""

  prompts = {}

  @classmethod
  def add(
      cls,
      name: str,
      data_dir: str,
      split: str,
      prefix: str,
      data_loader: Callable[..., Any] = utils.load_data_from_file,
      is_multi_turn: bool = False,
  ):
    """Adds a prompt to the registry."""
    if name in cls.prompts:
      raise ValueError(f"Prompt {name} already exists in the prompt registry.")

    cls.prompts[name] = CiCPrompt(
        data_dir=data_dir,
        split=split,
        prefix=prefix,
        data_loader=data_loader,
        is_multi_turn=is_multi_turn,
    )

  @classmethod
  def get_examples(
      cls, name: str, max_examples: Optional[int] = None
  ) -> list[Example]:
    """Returns the examples for a given prompt name."""
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
    cic_prompt = cls.prompts[name]
    examples = []

    # 1. Load the LOFT data.
    loft_data = cic_prompt.data_loader(
        data_dir=cic_prompt.data_dir, split=cic_prompt.split
    )

    # 2. For each query, create an example.
    for _, (qid, queries) in enumerate(loft_data.queries.items()):
      # Each query turn can be a list of query chunks (e.g. image + text).
      query_turns: list[list[ContentChunk]] = []
      if isinstance(queries, str):
        assert (
            not cic_prompt.is_multi_turn
        ), "Queries must be a list of strings for multi-turn."
        queries = [queries]
      gold_pids: list[Any] = []
      gold_answers: list[Any] = copy.deepcopy(loft_data.answers[qid])
      if not cic_prompt.is_multi_turn:
        if loft_data.metadata.values() and "qrels" in next(
            iter(loft_data.metadata.values())
        ):
          gold_pids = copy.deepcopy(
              [pid for pid, _ in loft_data.metadata[qid]["qrels"]]
          )
        else:
          if isinstance(loft_data.answers[qid], list) and all(
              [len(ans) == 2 for ans in loft_data.answers[qid]]
          ):
            gold_pids = copy.deepcopy(
                [pid for pid, _ in loft_data.answers[qid]]
            )
        # Make these as a single turn.
        gold_pids = [gold_pids]
        gold_answers = [gold_answers]
        query_turns.append(
            [
                ContentChunk(
                    _text=queries[0],
                    mime_type=MimeType.TEXT,
                )
            ]
        )
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
        for single_query in queries:
          query_turns.append(
              [
                  ContentChunk(
                      _text=single_query,
                      mime_type=MimeType.TEXT,
                  )
              ]
          )

      # 2-1. Process the context chunks.
      context_chunks: list[ContentChunk] = []  # Reset context chunks.
      for chunk in cic_prompt.prefix.split("\n"):
        chunk = chunk.strip()
        if not chunk.startswith("@@@@@ CHUNK") and chunk:
          context_chunks.append(
              ContentChunk(
                  _text=chunk,
                  mime_type=MimeType.TEXT,
              )
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
          )
      )
      if max_examples and len(examples) >= max_examples:
        break

    return examples
