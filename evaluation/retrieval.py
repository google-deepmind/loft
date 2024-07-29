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

"""Evaluation functions for retrieval."""

import collections
from typing import Any
from evaluation import loft_evaluation as evaluation

EvaluationInstance = evaluation.EvaluationInstance


def compute_recall_at_k(
    gold_ids: list[str],
    pred_ids: list[str],
    top_k: int,
    capped: bool = False,
) -> float:
  """Calculates the recall at k."""
  assert top_k > 0
  if not pred_ids:
    return 0
  pred_ids = set(pred_ids[:top_k])
  relevant_in_top_k = float(len(pred_ids.intersection(gold_ids)))

  # Capped recall@k is triggered when # of gold docs is > top_k
  if capped and len(gold_ids) > top_k:
    recall = relevant_in_top_k / top_k
  else:
    recall = relevant_in_top_k / len(gold_ids)
  return recall


def compute_mrecall_at_k(
    gold_ids: list[str],
    pred_ids: list[str],
    top_k: int,
) -> float:
  """Calculates the mRecall at k.

  This metric was introduced in Min et al., 2021:
  https://aclanthology.org/2021.emnlp-main.560.pdf

  Args:
    gold_ids: A list of gold IDs.
    pred_ids: A list of prediction IDs.
    top_k: The number of predictions to consider.

  Returns:
    mRecall@k metric.
  """
  assert top_k > 0
  if not pred_ids:
    return 0
  pred_ids = set(pred_ids[:top_k])
  relevant_in_top_k = float(len(pred_ids.intersection(gold_ids)))

  # This computes the completeness of the answers.
  return float(relevant_in_top_k == min(top_k, len(gold_ids)))


class RetrievalEvaluation(evaluation.LOFTEvalution):
  """Evaluation for Multi-turn retrieval datasets."""

  def __init__(
      self,
      config: evaluation.EvaluationConfig,
  ):
    super().__init__()
    self.config = config
    self.metrics = collections.defaultdict(list)

  def evaluate(self, instance: EvaluationInstance) -> list[dict[str, Any]]:
    """Evaluates a retrieval prediction."""

    metrics = []
    for turn_number in range(instance.num_turns):
      instance_metrics = {}
      if instance.num_turns == 1:
        gold_ids = self.process_goldens(instance.gold_answers)
      else:
        gold_ids = self.process_goldens([instance.gold_answers[turn_number]])
      pred_ids = self.process_prediction(instance.model_output[turn_number])
      instance_metrics["qid"] = instance.qid
      instance_metrics["turn_id"] = str(turn_number)
      instance_metrics["recall@1"] = compute_recall_at_k(
          gold_ids, pred_ids, 1, False
      )
      instance_metrics["recall@2"] = compute_recall_at_k(
          gold_ids, pred_ids, 2, False
      )
      instance_metrics["recall@3"] = compute_recall_at_k(
          gold_ids, pred_ids, 3, False
      )
      instance_metrics["recall@5"] = compute_recall_at_k(
          gold_ids, pred_ids, 5, False
      )
      instance_metrics["mrecall@1"] = compute_mrecall_at_k(
          gold_ids, pred_ids, 1
      )
      instance_metrics["mrecall@2"] = compute_mrecall_at_k(
          gold_ids, pred_ids, 2
      )
      instance_metrics["mrecall@3"] = compute_mrecall_at_k(
          gold_ids, pred_ids, 3
      )
      instance_metrics["mrecall@5"] = compute_mrecall_at_k(
          gold_ids, pred_ids, 5
      )
      instance_metrics["capped_recall@1"] = compute_recall_at_k(
          gold_ids, pred_ids, 1, True
      )
      metrics.append(instance_metrics)

      # Make sure to call below to aggregate all the metrics.
      self.add_instance_metrics(instance_metrics)

    return metrics
