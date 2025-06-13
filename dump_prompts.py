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

r"""Dumps prompts for a LOFT dataset.

TASK="retrieval"
DATASET="quest"
LENGTH="128k"
SPLIT="test"
PROMPT_NAME="${TASK}_${DATASET}_${LENGTH}_${SPLIT}:few_shot_with_cot"
python3 dump_prompts.py \
    --prompt_name="${PROMPT_NAME}" \
    --base_dir="${HOME}" \
    --output_format=text \
    --output_dir="${HOME}/prompts/${PROMPT_NAME}" \
    --output_format=csv
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import pandas as pd
import prompts  # pylint: disable=unused-import
from prompts import prompt_registry
import utils


PromptRegistry = prompt_registry.PromptRegistry

_PROMPT_NAME = flags.DEFINE_string(
    "prompt_name",
    None,
    "Name of the prompt to use.",
    required=True,
)

_BASE_DIR = flags.DEFINE_string(
    "base_dir",
    None,
    "Base directory of the prompt to use.",
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Path to output directory.",
    required=True,
)

_OUTPUT_FORMAT = flags.DEFINE_enum(
    "output_format",
    "text",
    ["text", "csv"],
    "Export format, text or csv.",
    required=False,
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    None,
    "Maximum number of examples to dump.",
)


def main(argv: Sequence[str]) -> None:
  del argv

  examples = PromptRegistry.get_examples(
      name=_PROMPT_NAME.value, base_dir=_BASE_DIR.value
  )

  os.makedirs(_OUTPUT_DIR.value)

  max_examples = _MAX_EXAMPLES.value or len(examples)

  examples_dicts = []
  for example in examples[:max_examples]:

    if _OUTPUT_FORMAT.value == "text":
      # dump to text
      output_path_prefix = os.path.join(
          _OUTPUT_DIR.value, f"{example.qid}_chunks"
      )
      utils.save_content_chunks(
          content_chunks=example.all_chunks,
          output_path_prefix=output_path_prefix,
      )
    else:
      # track for subsequent dump
      if example.num_turns > 1:
        raise ValueError("Only 1 turn is supported for csv dumping.")
      examples_dicts.append({
          "prompt_id": example.qid,
          "prompt": utils.concatenate_chunks(example.all_chunks).strip(),
          "targets": example.gold_answers,
      })

  # dump to csv
  if _OUTPUT_FORMAT.value == "csv":
    csv_path = os.path.join(_OUTPUT_DIR.value, _PROMPT_NAME.value + ".csv")
    pd.DataFrame(examples_dicts).to_csv(csv_path)

  print(f"Dumped examples to {_OUTPUT_DIR.value}")


if __name__ == "__main__":
  app.run(main)
