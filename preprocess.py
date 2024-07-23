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

r"""Preprocess the LOFT data by filling in the missing fields.

Example usage:
python preprocess.py \
    --input_dir=data/loft/retrieval/fiqa \
    --dataset=fiqa
"""

from collections.abc import Sequence
import json
import os
import zipfile

from absl import app
from absl import flags
# If not installed, run: pip install wget
import wget


DATASET_DOWNLOAD_LINKS = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "msmarco": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "webis_touche2020": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
}

_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    default=None,
    help="The input directory to extract the LOFT data from.",
    required=True,
)
_DATASET = flags.DEFINE_enum(
    "dataset",
    default=None,
    enum_values=list(DATASET_DOWNLOAD_LINKS),
    help="Dataset to download and preprocess.",
    required=True,
)
_COMPRESSION_TYPE = flags.DEFINE_enum(
    "compression_type",
    default="zip",
    enum_values=["zip"],
    help="Compression type of the dataset.",
)

DATASET_LENGTHS = ["32k", "128k", "1m"]
QUERY_FILES = [
    "dev_queries.jsonl",
    "few_shot_queries.jsonl",
    "test_queries.jsonl",
]


def extract_dataset(
    dataset: str, input_dir: str, compression_type: str
) -> None:
  """Extracts the dataset from the compressed file."""
  os.rename(
      os.path.join(input_dir, dataset.replace("_", "-") + ".zip"),
      os.path.join(input_dir, dataset + ".zip"),
  )
  if compression_type == "zip":
    with zipfile.ZipFile(
        os.path.join(input_dir, dataset + ".zip"), "r"
    ) as zip_ref:
      zip_ref.extractall(input_dir)
  else:
    raise ValueError(f"Unsupported compression type: {compression_type}")
  os.rename(
      os.path.join(input_dir, dataset.replace("_", "-")),
      os.path.join(input_dir, dataset),
  )


def download_dataset(dataset: str, download_dir: str) -> None:
  """Downloads the dataset from the dataset download link."""

  os.makedirs(download_dir, exist_ok=True)
  wget.download(DATASET_DOWNLOAD_LINKS[dataset], out=download_dir)
  extract_dataset(dataset, download_dir, _COMPRESSION_TYPE.value)


def load_dataset(
    dataset: str, input_dir: str
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
  """Load the downloaded source dataset."""
  qid2text = {}
  pid2text = {}
  # Other datasets like Flickr will be added later.
  if dataset in ["fiqa", "msmarco", "quora", "webis_touche2020"]:
    # Fill in the missing fields in the query and corpus files.
    source_dir = os.path.join(input_dir, "source", dataset)
    with open(os.path.join(source_dir, "queries.jsonl"), "r") as f:
      for line in f:
        query = json.loads(line)
        qid2text[query["_id"]] = query["text"]
    with open(os.path.join(source_dir, "corpus.jsonl"), "r") as f:
      for line in f:
        passage = json.loads(line)
        pid2text[passage["_id"]] = {
            "title": passage["title"],
            "text": passage["text"],
        }
  else:
    raise ValueError(f"Dataset {dataset} not available.")

  return qid2text, pid2text


def update_loft_dataset(
    qid2text: dict[str, str],
    pid2text: dict[str, dict[str, str]],
    input_dir: str,
) -> None:
  """Update the LOFT dataset with the missing fields."""
  for length in DATASET_LENGTHS:
    for query_file in QUERY_FILES:
      target_query_file = os.path.join(input_dir, length, query_file)
      if not os.path.exists(target_query_file):
        print(f"Skipping {target_query_file} as it does not exist.")
        continue
      queries = []
      with open(target_query_file, "r") as f:
        for line in f:
          query = json.loads(line)
          if query["qid"] not in qid2text:
            raise ValueError(f"Query {query['qid']} not found in the queries.")
          query["query_text"] = qid2text[query["qid"]]
          queries.append(query)
      with open(target_query_file, "w") as f:
        for query in queries:
          json.dump(query, f)
          f.write("\n")
        print(f"Wrote to {target_query_file}.")

    target_corpus_file = os.path.join(input_dir, length, "corpus.jsonl")
    passages = []
    with open(target_corpus_file, "r") as f:
      for line in f:
        passage = json.loads(line)
        if passage["pid"] not in pid2text:
          raise ValueError(f"Passage {passage['pid']} not found in the corpus.")
        passage["title_text"] = pid2text[passage["pid"]]["title"]
        passage["passage_text"] = pid2text[passage["pid"]]["text"]
        passages.append(passage)
    with open(target_corpus_file, "w") as f:
      for passage in passages:
        json.dump(passage, f)
        f.write("\n")
      print(f"Wrote to {target_corpus_file}.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  download_dataset(_DATASET.value, os.path.join(_INPUT_DIR.value, "source"))
  qid2text, pid2text = load_dataset(_DATASET.value, _INPUT_DIR.value)
  update_loft_dataset(qid2text, pid2text, _INPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
