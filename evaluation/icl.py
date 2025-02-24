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

"""Evaluation functions for Many-Shot ICL."""

import collections
from typing import Any

from evaluation import loft_evaluation as evaluation
from evaluation import utils


class IclEvaluation(evaluation.LOFTEvalution):
  """Evaluation for ICL model outputs."""

  def __init__(self, config: evaluation.EvaluationConfig):
    super().__init__()
    self.config = config
    self.metrics = collections.defaultdict(list)

  def evaluate(
      self, instance: evaluation.EvaluationInstance
  ) -> list[dict[str, Any]]:
    """Evaluates an ICL prediction."""

    multi_turn_metrics = []
    for turn_number in range(instance.num_turns):
      gold_answers = self.process_goldens(instance.gold_answers[turn_number])
      pred_answers = self.process_prediction(instance.model_output[turn_number])
      instance_metrics = {'qid': instance.qid, 'turn_id': str(turn_number)}

      if not pred_answers:
        instance_metrics['em'] = 0.0
      else:
        # Single prediction is allowed and matched against. Ill-formed Model
        # outputs may provide multiple answers but are ignored.
        if len(pred_answers) > 1:
          print(
              'Warning: Multiple answers found in prediction for single value'
              f' answers: {pred_answers}.'
          )
        instance_metrics['em'] = utils.compute_em(gold_answers, pred_answers[0])

      # Make sure to call below to aggregate all the metrics.
      self.add_instance_metrics(instance_metrics)
      multi_turn_metrics.append(instance_metrics)

    return multi_turn_metrics
