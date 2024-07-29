# Copyright 2024 DeepMind Technologies Limited
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

"""Evaluation functions for SQL.

The primary metric for SQL is execution accuracy. This compares the gold answer
from executing the gold SQL query to the predicted answer. Gold and predicted
answers must be lists of lists.

We do not enforce order on the predicted answer. For instance, if the question
is "What are top 3 largest cities in the United States?" and the gold answer is
[["New York"], ["Los Angeles"], ["Chicago"]], we consider the predicted answer
[["Los Angeles"], ["Chicago"], ["New York"]] as correct. There are some
questions that require sorting the answer. We filter questions that require
sorted answer when creating the SQL data for LOFT.
"""

import collections
from typing import Any

from evaluation import loft_evaluation as evaluation


def compute_execution_accuracy(
    gold_answer: list[list[str]],
    pred_answer: list[list[str]],
) -> float:
  """Calculates the execution accuracy."""
  if len(gold_answer) != len(pred_answer):
    return 0.0

  # Convert the list of lists into a list of Sets to allow for different
  # ordering (very relaxed).
  gold_answer = [set(ga) for ga in gold_answer]
  pred_answer = [set(pa) for pa in pred_answer]
  # Check that the gold answer perfect matches the predicted answer.
  for pa in pred_answer:
    if pa not in gold_answer:
      return 0.0
  return 1.0


def normalize_sql_answer(answers: list[list[Any]]) -> list[list[str]]:
  """Normalizes all answers.

  Takes a list of list of answers and converts all elements in the list
  of lists to a string while applying some form of normalization.

  Args:
      answers: A list of lists of answers.

  Returns:
      normalized_answers: A list of list of answers as strings.
  """
  normalized_answers = []
  for subanswer in answers:
    normalized_answers.append([])
    for item in subanswer:
      # Try to convert all numbers in string form into a number for rounding.
      # Round all numbers to 2 decimals to handle various answer precision
      try:
        item = f"{float(item):.2f}"
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      item = str(item).strip().lower()
      if item:
        normalized_answers[-1].append(item)
  return normalized_answers


class SqlEvaluation(evaluation.LOFTEvalution):
  """Evaluation for SQL model outputs."""

  def __init__(
      self,
      config: evaluation.EvaluationConfig,
  ):
    super().__init__()
    self.config = config
    self.metrics = collections.defaultdict(list)

  def evaluate(
      self, instance: evaluation.EvaluationInstance
  ) -> list[dict[str, Any]]:
    """Evaluates a SQL prediction."""
    multi_turn_metrics = []
    for turn_number in range(instance.num_turns):
      if instance.num_turns > 1:
        answers = normalize_sql_answer(instance.gold_answers[turn_number])
      else:
        answers = normalize_sql_answer(instance.gold_answers)
      if not isinstance(answers, list) or not isinstance(answers[0], list):
        raise ValueError(
            f"Gold answers must be a list of lists but got {answers}"
        )

      # We want our output to be nested lists.
      pred_answers = instance.model_output[turn_number]
      if pred_answers and not isinstance(pred_answers[0], list):
        pred_answers = [pred_answers]
      pred_answers = normalize_sql_answer(pred_answers)

      instance_metrics = {
          "qid": instance.qid,
          "exec_acc": compute_execution_accuracy(answers, pred_answers),
          "metadata": {"turn_number": turn_number},
      }
      # Make sure to call below to aggregate all the metrics.
      self.add_instance_metrics(instance_metrics)
      multi_turn_metrics.append(instance_metrics)

    return multi_turn_metrics
