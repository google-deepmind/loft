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

r"""Run evaluation on a set of predictions.

We provide a script to run evaluation on a set of predictions. The predictions
are expected to be in jsonl format, where each line is a json
dictionary containing the following fields:
* qid: The id of the query.
* model_outputs: The model predictions extracted from the model response.
* num_turns: The number of turns in the conversation.

We provide example predictions for each task under
evaluation/example_predictions. To run evaluation on the example predictions:
```
python run_evaluation.py \
  --answer_file_path evaluation/example_predictions/rag_nq/queries.jsonl \
  --pred_file_path evaluation/example_predictions/rag_nq/preds.jsonl \
  --task_type rag

python run_evaluation.py \
  --answer_file_path evaluation/example_predictions/rag_quest/queries.jsonl \
  --pred_file_path evaluation/example_predictions/rag_quest/preds.jsonl \
  --task_type multi_value_rag

python run_evaluation.py \
  --answer_file_path evaluation/example_predictions/retrieval_nq/queries.jsonl \
  --pred_file_path evaluation/example_predictions/retrieval_nq/preds.jsonl \
  --task_type retrieval

python run_evaluation.py \
  --answer_file_path evaluation/example_predictions/sql_spider/queries.jsonl \
  --pred_file_path evaluation/example_predictions/sql_spider/preds.jsonl \
  --task_type sql
```
where <task_type> is one of the keys in evaluation.EVALUATION_TASKS.

To understand which <task_type> to use for a given dataset, see the table under
the README.
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import evaluation
from evaluation import loft_evaluation


_ANSWER_FILE_PATH = flags.DEFINE_string(
    "answer_file_path",
    None,
    help="Path to gold answers",
    required=True,
)
_PRED_FILE_PATH = flags.DEFINE_string(
    "pred_file_path",
    None,
    help="Path to predictions to run evaluation on.",
    required=True,
)
_TASK_TYPE = flags.DEFINE_enum(
    "task_type",
    None,
    enum_values=evaluation.EVALUATION_TASKS.keys(),
    help="Task name to run evaluation on.",
    required=True,
)


def _load_predictions_from_jsonl(path: str) -> dict[str, Any]:
  """Loads predictions from a jsonl file."""
  predictions = {}
  for line in open(path):
    line = json.loads(line)
    predictions[line["qid"]] = line
  return predictions


def run_evaluation(
    answer_file_path: str,
    pred_file_path: str,
    task_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Evaluates predictions and returns metrics.

  Args:
    answer_file_path: Path to gold answers.
    pred_file_path: Path to predictions to run evaluation on.
    task_type: Task name to run evaluation on.

  Returns:
    metrics_per_line: List of metrics dictionaries per prediction.
    final_metrics: Metrics averaged over all predictions.
  """
  eval_task = evaluation.EVALUATION_TASKS[task_type]
  predictions = _load_predictions_from_jsonl(pred_file_path)

  metrics_per_line = []
  num_unanswered_queries = 0
  for line in open(answer_file_path):
    data_blob = json.loads(line)
    qid = data_blob["qid"]
    prediction = predictions.get(qid, None)
    if not prediction:
      num_unanswered_queries += 1
      # An EvaluationInstance is an object that contains the information
      # necessary to compute the metrics for a single prediction. Here we
      # create a dummy instance with an empty model output, which will cause
      # the evaluation to return 0 for all metrics because we do not have an
      # answer to compare to.
      eval_instance = loft_evaluation.EvaluationInstance(
          qid=qid,
          gold_answers=data_blob["answers"],
          model_output=[""],
          num_turns=1,
      )
      print(f"[Warning] Query {qid} was unanswered, marking it as incorrect.")
    else:
      eval_instance = loft_evaluation.EvaluationInstance(
          qid=qid,
          gold_answers=data_blob["answers"],
          model_output=prediction["model_outputs"],
          num_turns=prediction["num_turns"],
      )

    # Compute list of metrics dictionaries per instance
    instance_metric = eval_task.evaluate(eval_instance)
    metrics_per_line.extend(instance_metric)

  # Average all metrics over all predictions
  quality_metrics = eval_task.aggregate_metrics()
  final_metrics = {
      "quality": quality_metrics,
      "num_unanswered_queries": num_unanswered_queries,
  }

  return metrics_per_line, final_metrics


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  metrics_per_line, final_metrics = run_evaluation(
      answer_file_path=_ANSWER_FILE_PATH.value,
      pred_file_path=_PRED_FILE_PATH.value,
      task_type=_TASK_TYPE.value,
  )

  # Write two metrics file. E.g., if the preds file is /path/to/nq.jsonl, write:
  # * /path/to/nq_metrics_per_line.jsonl: Metrics per prediction.
  # * /path/to/nq_metrics.json: Metrics averaged over all predictions.
  output_dir = os.path.dirname(_PRED_FILE_PATH.value)
  file_basename = os.path.splitext(os.path.basename(_PRED_FILE_PATH.value))[0]

  mplpath = os.path.join(output_dir, f"{file_basename}_metrics_per_line.jsonl")
  with open(mplpath, "w") as f:
    for l in metrics_per_line:
      f.write(json.dumps(l) + "\n")

  metrics_path = os.path.join(output_dir, f"{file_basename}_metrics.json")
  with open(metrics_path, "w") as f:
    f.write(json.dumps(final_metrics, indent=4))

  print(json.dumps(final_metrics, indent=4))
  print(f"""Two files written to directory {output_dir}:
* {os.path.basename(mplpath)}: Metrics for each line in the prediction file.
* {os.path.basename(metrics_path)}: Metrics for all predictions.
  """.strip())


if __name__ == "__main__":
  app.run(main)
