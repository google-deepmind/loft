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

"""Evaluation utilities."""

import collections
import re
import string
from typing import Any
import unicodedata


def extract_gold_passage_ids(gold_answer: list[str | int]) -> str:
  """Extracts passage IDs from the gold answers field.

  The gold answer in the query file for retrieval looks like this ["doc35916",
  1].
  We extract the document ID for the gold answer to use for evaluation purposes.

  Args:
    gold_answer: The gold answer from the query file.

  Returns:
    The document ID for the gold answer.
  """
  if not isinstance(gold_answer[0], str) or not isinstance(gold_answer[1], int):
    raise ValueError(
        "Gold answer must be a list consisting of a str and an int."
    )
  return gold_answer[0]


def normalize_passage_id(passage_id: Any) -> str:
  return str(passage_id).strip()


def normalize_passage_ids(passage_ids: list[Any]) -> list[str]:
  return [normalize_passage_id(passage_id) for passage_id in passage_ids]


def normalize_answer(s: str) -> str:
  """Taken from SQuAD evaluation."""

  s = unicodedata.normalize("NFD", s)

  def remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text: str) -> str:
    return " ".join(text.split())

  def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text: str) -> str:
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answers(answers: list[str]) -> list[str]:
  return [normalize_answer(answer) for answer in answers]


def convert_to_str(texts: list[Any]) -> list[str]:
  return [str(text) for text in texts]


def get_tokens(s: str) -> list[str]:
  """Taken from SQuAD evaluation."""
  if not s:
    return []
  return normalize_answer(s).split()


def compute_em(gold_answers: list[str], pred_answer: str) -> float:
  """Calculates exact match score. Taken from SQuAD evaluation."""
  return max([float(ga == pred_answer) for ga in gold_answers])


def compute_subspan_em(gold_answers: list[str], pred_answer: str) -> float:
  """Calculates subspan match score."""
  return max([1.0 if ga in pred_answer else 0.0 for ga in gold_answers])


def compute_f1(gold_answers: list[str], pred_answer: str) -> float:
  """Calculates F1 score. Taken from SQuAD evaluation."""
  pred_toks = get_tokens(pred_answer)

  f1_scores = []
  for ga in gold_answers:
    gold_toks = get_tokens(ga)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
      f1_scores.append(0.0)
      continue

    if not gold_toks or not pred_toks:
      # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
      f1 = float(gold_toks == pred_toks)
    else:
      precision = 1.0 * num_same / len(pred_toks)
      recall = 1.0 * num_same / len(gold_toks)
      f1 = (2 * precision * recall) / (precision + recall)
    f1_scores.append(f1)

  return max(f1_scores)
