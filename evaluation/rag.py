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

"""Evaluation functions for RAG. EM/F1 are from SQuAD and slightly modified."""

import collections
from typing import Any

from evaluation import loft_evaluation as evaluation
from evaluation import utils
import numpy as np
import scipy.optimize


def compute_em_multi_value(
    gold_answers: list[str], pred_answers: list[str]
) -> float:
  """Calculates exact match score. Taken from SQuAD evaluation."""
  return float(set(gold_answers) == set(pred_answers))


def compute_coverage(gold_answers: list[str], pred_answers: list[str]) -> float:
  """Calculates coverage of gold_answers in pred_answers."""
  return len(set(pred_answers).intersection(set(gold_answers))) / float(
      len(gold_answers)
  )


def compute_multi_value_subspan_em(
    gold_answers: list[str], pred_answers: list[str]
) -> float:
  """Calculates subspan match score. Adopted from DROP evaluation."""
  scores = np.zeros([len(gold_answers), len(pred_answers)])
  for gold_index, gold_item in enumerate(gold_answers):
    for pred_index, pred_item in enumerate(pred_answers):
      if gold_item in pred_item or pred_item in gold_item:
        scores[gold_index, pred_index] = 1
  row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores)
  aligned_scores = np.zeros(len(gold_answers))
  for r, c in zip(row_ind, col_ind):
    aligned_scores[r] = scores[r, c]
  return float(all(aligned_scores))


class MultiValueRagEvaluation(evaluation.LOFTEvalution):
  """Evaluation for RAG model outputs for single-turn Set based datasets."""

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
    """Evaluates a RAG prediction."""

    gold_answers = self.process_goldens(instance.gold_answers)
    pred_answers = self.process_prediction(instance.model_output[0])

    if not pred_answers:
      instance_metrics = {
          'qid': instance.qid,
          'em': 0.0,
          'subspan_em': 0.0,
          'f1': 0.0,
      }
      self.add_instance_metrics(instance_metrics)
      return [instance_metrics]

    instance_metrics = {}
    instance_metrics['qid'] = instance.qid
    instance_metrics['em'] = compute_em_multi_value(gold_answers, pred_answers)
    instance_metrics['coverage'] = compute_coverage(gold_answers, pred_answers)
    instance_metrics['subspan_em'] = compute_multi_value_subspan_em(
        gold_answers, pred_answers
    )
    # Make sure to call below to aggregate all the metrics.
    self.add_instance_metrics(instance_metrics)

    return [instance_metrics]


class RagEvaluation(evaluation.LOFTEvalution):
  """Evaluation for multi-turn RAG model outputs."""

  def __init__(self, config: evaluation.EvaluationConfig):
    super().__init__()
    self.config = config
    self.metrics = collections.defaultdict(list)

  def evaluate(
      self, instance: evaluation.EvaluationInstance
  ) -> list[dict[str, Any]]:
    """Evaluates a RAG prediction."""

    multi_turn_metrics = []
    for turn_number in range(instance.num_turns):
      if instance.num_turns == 1:
        gold_answers = self.process_goldens(instance.gold_answers)
      else:
        gold_answers = self.process_goldens(instance.gold_answers[turn_number])
      pred_answers = self.process_prediction(instance.model_output[turn_number])

      if not pred_answers:
        instance_metrics = {
            'qid': instance.qid,
            'turn_id': str(turn_number),
            'em': 0.0,
            'subspan_em': 0.0,
            'f1': 0.0,
        }

        multi_turn_metrics.append(instance_metrics)
        self.add_instance_metrics(instance_metrics)
        continue

      instance_metrics = {}
      # Single prediction is allowed and matched against. Ill-formed Model
      # outputs may provide multiple answers but are ignored.
      if len(pred_answers) > 1:
        print(
            'Warning: Multiple answers found in prediction for single value'
            f' retrieval: {pred_answers}.'
        )

      pred_answer = pred_answers[0]
      instance_metrics['qid'] = instance.qid
      instance_metrics['turn_id'] = str(turn_number)
      instance_metrics['em'] = utils.compute_em(gold_answers, pred_answer)
      instance_metrics['subspan_em'] = utils.compute_subspan_em(
          gold_answers, pred_answer
      )
      instance_metrics['f1'] = utils.compute_f1(gold_answers, pred_answer)

      # Make sure to call below to aggregate all the metrics.
      self.add_instance_metrics(instance_metrics)

      multi_turn_metrics.append(instance_metrics)

    return multi_turn_metrics
