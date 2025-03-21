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

r"""Run LCLM inference on LOFT.

We run LCLMs on LOFT by having LCLMs ingest the entire corpus along with each
query and output the answer in natural language. We use a few-shot prompt to
show what the CoT reasoning should look like.

Example run command:
# Retrieval
BASE_DIR=./data
LENGTH="32k"
TASK_TYPE="retrieval"
SPLIT="dev"
PROMPT_TYPE="few_shot_cot_simple_reverse:corpus_echo:None_max_shots"
PROMPT="${TASK_TYPE}_${DATASET}_${LENGTH}_${SPLIT}:${PROMPT_TYPE}"

mkdir -p ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}

python run_inference.py \
    --prompt_name ${PROMPT} \
    --base_dir ${BASE_DIR} \
    --data_dir ${TASK_TYPE}/${DATASET}/${LENGTH} \
    --split ${SPLIT} \
    --context_length ${LENGTH} \
    --output_path ${BASE_DIR}/outputs/${TASK_TYPE}/${DATASET}/${LENGTH}/${SPLIT}_predictions.jsonl \
    --project_id ${PROJECT_ID} \
    --overwrite
"""

import concurrent.futures
import functools
import json
import os
from typing import Any, Dict, Sequence

from absl import app
from absl import flags
from inference import models
import prompts  # pylint: disable=unused-import
from prompts import prompt_registry
import tqdm
import utils


CONTEXT_LENGTH_TO_NUM_TOKENS = {
    "32k": 32000,
    "128k": 128000,
    "1m": 1000000,
}

_PROMPT_NAME = flags.DEFINE_string(
    "prompt_name",
    None,
    "Name of the prompt to use.",
    required=True,
)
_TASK_TYPE = flags.DEFINE_string(
    "task_type",
    None,
    "Task type of the prompt to use.",
    required=True,
)
_BASE_DIR = flags.DEFINE_string(
    "base_dir",
    None,
    "Path to the base directory.",
    required=True,
)
_DATA_DIR = flags.DEFINE_string(
    "data_dir",
    None,
    "Relative path to the data directory given the base directory.",
    required=True,
)
_SPLIT = flags.DEFINE_string(
    "split",
    "dev",
    "Split of the data to use.",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Path to write prediction outputs as a JSONL file.",
    required=True,
)
_MODEL_URL_OR_NAME = flags.DEFINE_string(
    "model_url_or_name",
    "gemini-1.5-pro",
    "Evergreen model URL or API-based model name.",
)
_PROJECT_ID = flags.DEFINE_string(
    "project_id",
    None,
    "Project ID of Google Cloud Project.",
    required=True,
)
_CONTEXT_LENGTH = flags.DEFINE_enum(
    "context_length",
    "32k",
    CONTEXT_LENGTH_TO_NUM_TOKENS.keys(),
    "Context length of the prompt. Four pre-defined lengths are available.",
)
_OVERWRITE = flags.DEFINE_bool(
    "overwrite",
    False,
    "If True, regenerate the outputs. If False, reuse results from output file"
    "if it already exists.",
)
_CACHE_FIRST_INPUT = flags.DEFINE_bool(
    "cache_first_input",
    False,
    "If True, run and cache the first input to the model.",
)
_MAX_WORKERS = flags.DEFINE_integer(
    "max_workers",
    1,
    "Maximum number of workers to use for multi-thread inference. This should"
    "be 1x to 2x the number of model replicas available.",
)
_LOG_FAILING_PROMPTS = flags.DEFINE_bool(
    "log_failing_prompts",
    True,
    "If True, log the failing prompts. This is useful for debugging VertexAI.",
)

MimeType = utils.MimeType
ContentChunk = utils.ContentChunk
PromptRegistry = prompt_registry.PromptRegistry


def get_num_tokens(text_input: str) -> int:
  # Simple tokenization for the estimated number of tokens.
  return len(text_input.strip().split(" "))


def _run_one_example(
    example: utils.Example,
    model: models.Model,
    finished_lines: Dict[str, Any],
) -> Dict[str, Any] | None:
  """Runs one example and returns the output."""
  try:
    return utils.run_one_example(example, model, finished_lines)
  except Exception as exception:  # pylint: disable=broad-exception-caught
    print(exception)
    output_path = f"{_OUTPUT_PATH.value}.failed_prompt.{example.qid}"
    print(f"Logging failing prompt to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if _LOG_FAILING_PROMPTS.value:
      with open(output_path, "wb") as f:
        for chunk in example.all_chunks:
          f.write(chunk.data)
        f.flush()
    return None


def main(argv: Sequence[str]) -> None:
  del argv
  if _PROMPT_NAME.value not in PromptRegistry.prompts:
    task_name = _PROMPT_NAME.value.split("_")[0]
    print(PromptRegistry.prompts.keys())
    registry_str = "\n".join(
        filter(
            lambda x: x.startswith(task_name),
            list(PromptRegistry.prompts.keys()),
        )
    )
    raise ValueError(
        f"Prompt {_PROMPT_NAME.value} not found in registry.\nAvailable"
        f" prompts:\n{registry_str}"
    )

  pid_mapper = None
  if _TASK_TYPE.value in ["retrieval", "mm"]:
    pid_mapper = {
        str(idx): pid
        for idx, pid in enumerate(
            utils.load_data_from_file(
                data_dir=_DATA_DIR.value,
                base_dir=_BASE_DIR.value,
                split=_SPLIT.value,
            ).corpus
        )
    }
  answer_prefix = "final answer"
  if _TASK_TYPE.value == "icl":
    answer_prefix = "output"

  model = models.get_model(
      model_url_or_name=_MODEL_URL_OR_NAME.value,
      project_id=_PROJECT_ID.value,
      pid_mapper=pid_mapper,
      answer_prefix=answer_prefix,
  )

  finished_lines = {}
  if os.path.exists(_OUTPUT_PATH.value) and not _OVERWRITE.value:
    with open(_OUTPUT_PATH.value) as f:
      for l in f:
        l = json.loads(l)
        finished_lines[l["qid"]] = l
  else:
    os.makedirs(os.path.dirname(_OUTPUT_PATH.value), exist_ok=True)
  print(f"Found {len(finished_lines)} finished lines.")

  # Log the configuration that was used. This is nice for knowing what exact
  # command was to run when you have to look at the results months after an
  # experiment is run (e.g. during rebuttal).
  utils.save_run_metadata(
      flags.FLAGS.flags_by_module_dict(), output_path_prefix=_OUTPUT_PATH.value
  )

  # Load the lines for inference and the one-shot prompt, then runs inference.
  examples = PromptRegistry.get_examples(
      name=_PROMPT_NAME.value, base_dir=_BASE_DIR.value
  )
  qid2example = {ex.qid: ex for ex in examples}

  for ex in qid2example.values():
    if not all(chunk.mime_type == MimeType.TEXT for chunk in ex.context_chunks):
      continue
    num_tokens = get_num_tokens(
        "\n".join(chunk.data.decode("utf-8") for chunk in ex.all_chunks)
    )
    if num_tokens > CONTEXT_LENGTH_TO_NUM_TOKENS[_CONTEXT_LENGTH.value]:
      raise ValueError(
          f"qid={ex.qid} has {num_tokens} tokens in its prompt, which is more"
          f" than the context length of {_CONTEXT_LENGTH.value}"
      )

  # Caching and saving one prompt to disk.
  print("Starting saving one prompt to disk...")
  indexing_example = list(qid2example.values())[0]
  prompt_path = f"{_OUTPUT_PATH.value}.prompt_first_query_example"
  utils.save_content_chunks(indexing_example.all_chunks, prompt_path)
  print(f"Finished saving one prompt to disk in {prompt_path}.txt")

  if _CACHE_FIRST_INPUT.value:
    try:
      print("Starting caching.")
      # Do prefix cache by running the inference once.
      model_output = model.infer(
          list(indexing_example.all_chunks),
      )
      print("Finished caching. Model output:", model_output)
    except Exception as exception:  # pylint: disable=broad-exception-caught
      print(exception)
      print("Failed to cache; continuing inference without caching...")

  with open(_OUTPUT_PATH.value, "w", encoding="utf-8") as f:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_WORKERS.value
    ) as executor:
      eval_futures = executor.map(
          functools.partial(
              _run_one_example, model=model, finished_lines=finished_lines
          ),
          qid2example.values(),
      )
      for output in tqdm.tqdm(eval_futures, total=len(qid2example)):
        if output:
          f.write(json.dumps(output, ensure_ascii=False) + "\n")
          f.flush()

    print(f"Wrote results to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
  app.run(main)
