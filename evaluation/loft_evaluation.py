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

"""LOFT evaluation protocol."""

import abc
import dataclasses
from typing import Any, Callable, Protocol
import numpy as np


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
  """Evaluation configuration."""

  # Sequence of processing functions for model response.
  process_model_response_fns: list[Callable[..., Any]]

  # Sequence of processing functions for gold answer.
  process_gold_answer_fns: list[Callable[..., Any]]


@dataclasses.dataclass(frozen=True)
class EvaluationInstance:
  """The protocol for classes that perform evaluation on LOFT EvaluationInstance."""

  # Unique instance identifier.
  qid: str
  # Multiple gold references in an instance.
  gold_answers: list[Any]
  # Single model response (greedy) in an instance.
  model_output: Any | list[Any]
  # Number of converstaional turns in the instance
  num_turns: int
  # Any additional metadata about the instance.
  metadata: dict[str, Any] | None = None


class LOFTEvalution(Protocol):
  """The protocol for classes that perform evaluation on LOFT EvaluationInstance."""

  config: EvaluationConfig
  metrics: dict[str, Any]

  def process_goldens(self, goldens: list[Any]) -> list[Any]:
    """Processes goldens by executing functions defined in EvaluationConfig."""
    assert self.config is not None
    assert self.config.process_gold_answer_fns

    processed_goldens = []
    for gold in goldens:
      for fn in self.config.process_gold_answer_fns:
        gold = fn(gold)
      processed_goldens.append(gold)

    return processed_goldens

  def process_prediction(self, prediction: str) -> Any:
    """Processes model response by executing functions defined in EvaluationConfig."""
    assert self.config is not None
    assert self.config.process_model_response_fns

    for fn in self.config.process_model_response_fns:
      prediction = fn(prediction)

    return prediction

  def add_instance_metrics(self, instance_metrics: dict[str, Any]):
    """Add instance specific metrics to the global metrics field."""
    assert self.metrics is not None
    for metric_name, value in instance_metrics.items():
      self.metrics[metric_name].append(value)

  @abc.abstractmethod
  def evaluate(self, instance: EvaluationInstance) -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing evaluation metrics."""

  def aggregate_metrics(self) -> dict[str, Any]:
    assert self.metrics is not None
    aggregated_metrics = {}
    for metric_name, metric_values in self.metrics.items():
      if any([
          not (isinstance(value, float) or isinstance(value, int))
          for value in metric_values
      ]):
        continue
      aggregated_metrics[metric_name] = np.mean(metric_values)
    return aggregated_metrics
