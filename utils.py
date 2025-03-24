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

"""Common utility for running inference and evaluation."""

import ast
import base64
from collections.abc import Sequence
import dataclasses
import enum
import getpass
import io
import itertools
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import attrs
import numpy as np
from PIL import Image


_PROJECT_DIRECTORY = "learning/gemini/agents/projects/loft"
_DATA_FILE_PATH = "{data_dir}/{split}_queries.jsonl"
Passage = Dict[str, Any]


def _resize_image_bytes_to_512x512(input_image_bytes: bytes) -> bytes:
  # Open the image from bytes
  input_image = Image.open(io.BytesIO(input_image_bytes))

  # Resize the image to 512x512
  resized_image = input_image.resize((512, 512))

  # Save the resized image to a bytes buffer
  output_bytes_io = io.BytesIO()
  resized_image.save(output_bytes_io, format="JPEG")
  return output_bytes_io.getvalue()


@dataclasses.dataclass
class LOFTData:
  queries: Dict[str, str]
  corpus: Dict[str, Passage]
  answers: Dict[str, Any]
  metadata: Dict[str, Any]


class MimeType(enum.StrEnum):
  TEXT = "text/plain"
  IMAGE_JPEG = "image/jpeg"
  AUDIO_WAV = "audio/wav"


@dataclasses.dataclass
class ContentChunk:
  """A single content chunk."""

  mime_type: MimeType
  role: str = "user"
  _text: Optional[str] = None  # Will be directly encoded to bytes.
  _path: Optional[str] = None  # The file will be read as bytes.
  _embedding: Optional[np.ndarray] = None
  _image_bytes: Optional[bytes] = None

  @property
  def data(self) -> bytes:
    """Cached property for data."""
    if self._text is not None:
      if self.mime_type != MimeType.TEXT:
        raise ValueError(
            "Text content chunk must have mime type TEXT, but got %s"
            % self.mime_type
        )
      return self._text.encode("utf-8")

    if self._path is not None:
      # NOTE: Assuming that _path is a single-turn data.
      if isinstance(self._path, list):
        if len(self._path) > 1:
          raise ValueError(
              "Multi-turn data is not supported for path content chunk."
          )
        self._path = self._path[0]
      with open(self._path, "rb") as fp:
        if self.mime_type == MimeType.IMAGE_JPEG:
          image_bytes = fp.read()
          return _resize_image_bytes_to_512x512(image_bytes)
        elif self.mime_type == MimeType.AUDIO_WAV:
          return fp.read()

    if self._image_bytes is not None:
      return self._image_bytes

    raise ValueError("Either _text or _path must be set.")

  @property
  def path_string(self) -> str:
    if self._path is not None:
      if isinstance(self._path, list):
        return "\n".join(self._path) + "\n"
      return self._path + "\n"
    else:
      return "No path specified."

  def __post_init__(self):
    """Validate that either _text or _path is set."""
    if self._text is None and self._path is None and self._image_bytes is None:
      raise ValueError("Either _text or _path must be set.")


@attrs.define(frozen=True)
class QueryTurn:
  """A query turn in the LOFT Example."""

  # A query turn can have multiple chunks (e.g. a text and an image).
  chunks: tuple[ContentChunk, ...] = attrs.field(converter=tuple)


def _convert_to_tuple_recursively(
    obj: Any,
) -> tuple[Any, ...] | Any:
  if isinstance(obj, list):
    return tuple(_convert_to_tuple_recursively(v) for v in obj)
  else:
    return obj


@attrs.define(frozen=True)
class Example:
  """A test example in the LOFT dataset."""

  qid: str
  num_turns: int

  # Context of the conversation, e.g., instructions, corpus and fewshots
  context_chunks: tuple[ContentChunk, ...] = attrs.field(converter=tuple)

  # Query turns, each turn possibly containing multiple chunks.
  query_turns: tuple[QueryTurn, ...] = attrs.field(converter=tuple)

  # Answers for each turn
  gold_answers: tuple[tuple[Any, ...], ...] | None = attrs.field(
      converter=_convert_to_tuple_recursively, default=None
  )

  # PIDs for each turn
  # This can be used to know which passages were used to generate the
  # gold_answers, for instance for reordering a corpus of passages.
  gold_pids: tuple[tuple[Any, ...], ...] | None = attrs.field(
      converter=_convert_to_tuple_recursively, default=None
  )

  # Document boundaries within the corpus, in the format of:
  # (start_1, end_1), ..., (start_n, end_n), with start and end being index
  # within context_chunks. This is useful for blockwise, where document
  # boundaries need to be known.
  corpus_document_boundaries: tuple[tuple[int, int], ...] | None = attrs.field(
      converter=_convert_to_tuple_recursively, default=None
  )

  def __post_init__(self):
    if len(self.query_turns) != self.num_turns:
      raise ValueError("query_turns must have length num_turns.")

  @property
  def all_chunks(self) -> tuple[ContentChunk, ...]:
    return self.context_chunks + tuple(  # pytype: disable=bad-return-type
        itertools.chain.from_iterable(q.chunks for q in self.query_turns)
    )


def save_content_chunks(
    content_chunks: tuple[ContentChunk, ...],
    output_path_prefix: str,
):
  """Saves content chunks to a recordio file and optionally to a txt file."""

  with open(f"{output_path_prefix}.txt", "w") as f:
    for chunk in content_chunks:
      f.write(f"@@@@@ CHUNK ({chunk.role}, {chunk.mime_type}) @@@@@\n")
      if chunk.mime_type == MimeType.TEXT:
        f.write(chunk.data.decode("utf-8"))
      else:
        f.write(chunk.path_string)
      f.write("\n")


def concatenate_chunks(chunks: tuple[ContentChunk, ...]) -> str:
  """Returns a string representation of the chunks."""
  content = ""
  for chunk in chunks:
    if chunk.mime_type == MimeType.TEXT:
      content += chunk.data.decode("utf-8")
    else:
      raise ValueError("Only text chunks are supported for now.")
  return content


def save_run_metadata(
    flags_by_module: Dict[str, Any],
    output_path_prefix: str,
):
  """Saves the run metadata for debugging after the fact."""
  os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
  run_metadata = {
      "user": getpass.getuser(),
      "flags": {},
  }
  for module, module_flags in flags_by_module.items():
    # We're just interested in logging our own flags.
    if _PROJECT_DIRECTORY in module:
      for flag in module_flags:
        run_metadata["flags"][flag.name] = str(flag.value)
  with open(f"{output_path_prefix}.run_metadata", "w") as f:
    f.write(json.dumps(run_metadata, indent=2))


def load_data_from_file(
    data_dir: str,
    base_dir: str,
    split: str = "dev",
    resource_dir: str = "",
    strip_text: bool = True,
    blocklist_words: Sequence[str] | None = ("childhood abuse",),
) -> LOFTData:
  """Loads the queries, corpus and answers from the given data directory."""
  del resource_dir
  queries = {}
  corpus = {}
  answers = {}
  metadata = {}

  def _maybe_strip(s):
    return s.strip() if strip_text and isinstance(s, str) else s

  def _maybe_clean(s):
    if blocklist_words is None:
      return s
    for word in blocklist_words:
      if word in s:
        logging.info("Remove blocklisted word from %s", s)
        s = s.replace(word, "")
    return s

  resource_dir = os.path.join(base_dir, "data", data_dir, "resources")
  if not os.path.exists(resource_dir):
    resource_dir = ""

  with open(
      _DATA_FILE_PATH.format(
          data_dir=os.path.join(base_dir, "data", data_dir), split=split
      ),
      "rt",
  ) as f:
    for line in f:
      line = json.loads(line)
      queries[line["qid"]] = _maybe_strip(line["query_text"])
      answers[line["qid"]] = line["answers"]
      metadata[line["qid"]] = line["metadata"]
      if resource_dir:
        if "img_path" in line["metadata"]:
          metadata[line["qid"]]["img_path"] = [
              os.path.join(resource_dir, img_path)
              for img_path in line["metadata"]["img_path"]
          ]
        if "audio_path" in line["metadata"]:
          metadata[line["qid"]]["audio_path"] = [
              os.path.join(resource_dir, audio_path)
              for audio_path in line["metadata"]["audio_path"]
          ]

  with open(os.path.join(base_dir, "data", data_dir, "corpus.jsonl"), "r") as f:
    for line in f:

      line = json.loads(line)
      corpus[line["pid"]] = {
          "title": _maybe_clean(_maybe_strip(line["title_text"])),
          "passage": _maybe_clean(_maybe_strip(line["passage_text"])),
          "metadata": line["metadata"],
      }
      if resource_dir:
        if "img_path" in corpus[line["pid"]]["metadata"]:
          corpus[line["pid"]]["metadata"]["img_path"] = [
              os.path.join(resource_dir, img_path)
              for img_path in line["metadata"]["img_path"]
          ]
        if "audio_path" in corpus[line["pid"]]["metadata"]:
          corpus[line["pid"]]["metadata"]["audio_path"] = os.path.join(
              resource_dir, line["metadata"]["audio_path"]
          )

  return LOFTData(queries, corpus, answers, metadata)


def load_bbh_data_from_file(
    data_dir: str,
    base_dir: str,
    split: str = "dev",
    **kwargs,
) -> LOFTData:
  """Loads the queries, corpus and answers from the given ICL data directory."""
  del kwargs
  queries = {}
  corpus = {}
  answers = {}
  metadata = {}
  with open(os.path.join(base_dir, "data", data_dir, "corpus.json"), "r") as f:
    support_set = json.load(f)
  with open(
      os.path.join(base_dir, "data", data_dir, f"{split}_queries.json"), "r"
  ) as f:
    query_set = json.load(f)

  for query_idx, query in enumerate(query_set):
    queries[f"{str(query_idx)}"] = query
    answers[f"{str(query_idx)}"] = [query["target"].split(" ")]

  for support_idx, support in enumerate(support_set):
    corpus[f"support_{str(support_idx)}"] = support

  return LOFTData(queries, corpus, answers, metadata)


def load_long_icl_bench_dialogue_re_data_from_file(
    data_dir: str,
    base_dir: str,
    split: str = "dev",
    **kwargs,
) -> LOFTData:
  """Loads the queries, corpus and answers from the given ICL data directory."""
  del kwargs
  queries = {}
  corpus = {}
  answers = {}
  metadata = {}
  with open(os.path.join(base_dir, "data", data_dir, "corpus.json"), "r") as f:
    support_set = json.load(f)
  query_set = []
  with open(
      os.path.join(base_dir, "data", data_dir, f"{split}_queries.jsonl"), "r"
  ) as f:
    for line in f:
      query_set.append(json.loads(line))

  for query_idx, query in enumerate(query_set):
    queries[f"query_{split}_{str(query_idx)}"] = query
    answers[f"query_{split}_{str(query_idx)}"] = [query["answers"]]
    metadata[f"query_{split}_{str(query_idx)}"] = {
        "relations": query["relations"]
    }

  for support_idx, support in enumerate(support_set):
    corpus[f"support_{str(support_idx)}"] = support

  return LOFTData(queries, corpus, answers, metadata)


def encode_bytes_as_inline_image(image_bytes: bytes) -> str:
  return base64.b64encode(image_bytes).decode("utf-8")


def _trim_model_output(model_output: str) -> str:
  """Trims the output to the first period."""
  if "====== Now let's start!" in model_output:
    model_output = model_output.split("====== Now let's start!")[0]
    logging.info("Trimmed model output: %s", model_output)
  return model_output


def run_one_example(
    example: Example,
    model: Any,
    finished_lines: Dict[str, Any],
) -> Dict[str, Any]:
  """Runs inference on one LOFT Example.

  Args:
    example: the example to run inference on.
    model: the Model to use.
    finished_lines: a dict of qid to pre-computed result as cache.

  Returns:
    A dict of inference results that can be saved to json.

  Raises:
    ValueError: if the model inference fails.
  """

  if example.qid in finished_lines:
    return finished_lines[example.qid]

  conversation = list(example.context_chunks)
  responses: list[str] = []
  response_time_taken = []
  for query_turn in example.query_turns:
    conversation += list(query_turn.chunks)
    try:
      st = time.time()
      model_output = model.infer(conversation)
      if len(example.query_turns) > 1:
        # Trim model output for multi-turn conversations.
        model_output = _trim_model_output(model_output)
      responses.append(model_output)

      # Add the model output as an assistant chunk.
      conversation.append(
          ContentChunk(
              mime_type=MimeType.TEXT, _text=str(model_output), role="assistant"
          )
      )
      response_time_taken.append(time.time() - st)
    except Exception as exception:  # pylint: disable=broad-exception-caught
      raise ValueError(
          f"Failed to run inference for query turn {example.qid} with error"
          f" {exception}."
      ) from exception

  output = {
      "qid": example.qid,
      "num_turns": example.num_turns,
      "model_outputs": responses,
      "metadata": {
          "time_taken_seconds": response_time_taken,
          "new_gold_answers": example.gold_answers,
      },
  }

  return output


def extract_prediction(
    model_output: str, answer_prefix: str = "final answer"
) -> List[str]:
  """Extracts the prediction from the model output."""

  def _escape_single_quotes(s: str):
    # Converts patterns like "['child bride', 'the devil's sleep']" to
    # "['child bride', 'the devil\'s sleep']" to allow for proper parsing.

    pattern = r"([a-zA-Z0-9])'([a-zA-Z0-9])"
    replacement = r"\1\'\2"

    return re.sub(pattern, replacement, s)

  # Remove formatting.
  model_output = model_output.replace("*", "").replace("`", "")
  model_output = model_output.strip().split("\n")
  # Extract the predictions from the model output
  preds = []
  for l in model_output:
    # Turns the string "Final Answer: [1, ...]" into the list of ints [1, ...]
    if "[" in l and "]" in l:
      if answer_prefix not in l.lower():
        logging.warning("Answer prefix %s not found in %s", answer_prefix, l)
      pred_start_index = l.find("[")
      pred_end_index = l.rfind("]") + 1  # Finds the last "]"
      pred_as_str = l[pred_start_index:pred_end_index].strip()
      try:
        pred_as_str = _escape_single_quotes(pred_as_str)
        preds = ast.literal_eval(pred_as_str)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(l, e)
      break
  return preds
