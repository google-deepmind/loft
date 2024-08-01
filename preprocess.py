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
import glob
import json
import os
import zipfile

from absl import app
from absl import flags
import cv2
import numpy as np
import tqdm
import wget


# pylint: disable=line-too-long
DATASET_DOWNLOAD_LINKS = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "msmarco": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "webis_touche2020": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
    "msrvtt": (
        "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
    ),
}
# pylint: enable=line-too-long

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

VIDEO_FILEPATTERN = "msrvtt/videos/all/{}.mp4"
DATASET_LENGTHS = ["32k", "128k", "1m"]
QUERY_FILES = [
    "dev_queries.jsonl",
    "few_shot_queries.jsonl",
    "test_queries.jsonl",
]


def extract_frames_from_video(video_path, output_pattern, num_frames=3):
  """Extract video frames from a input video at a given frame rate."""
  # Open the video file
  video_capture = cv2.VideoCapture(video_path)

  # Get the total number of frames in the video
  total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

  # Generate the frame indices to sample uniformly
  frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

  frame_names = []
  for frame_index in frame_indices:
    # Set the video capture to the specific frame index
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = video_capture.read()

    if ret:
      # Save the sampled frame to the output folder
      frame_name = f"{output_pattern}_{frame_index:08d}.jpg"
      cv2.imwrite(frame_name, frame)
      frame_names.append(os.path.basename(frame_name))

  video_capture.release()
  return frame_names


def extract_video_resource(
    download_dir: str, resource_dir: str
) -> dict[str, str]:
  """Extract video into image frames."""
  if not os.path.exists(resource_dir):
    os.makedirs(resource_dir)

  video2frames = dict()
  video_filepaths = glob.glob(
      os.path.join(download_dir, VIDEO_FILEPATTERN.format("*"))
  )
  for video_filepath in tqdm.tqdm(video_filepaths):
    video_id = os.path.basename(video_filepath).split(".")[0]
    video_frame_pattern = os.path.join(resource_dir, video_id)

    video2frames[video_id] = extract_frames_from_video(
        video_filepath, video_frame_pattern
    )
  return video2frames


def extract_dataset(
    dataset: str, input_dir: str, compression_type: str
) -> None:
  """Extracts the dataset from the compressed file."""
  if compression_type == "zip":
    with zipfile.ZipFile(
        os.path.join(input_dir, dataset + ".zip"), "r"
    ) as zip_ref:
      extracted_dir = zip_ref.namelist()[0]
      zip_ref.extractall(input_dir)
      # Rename the extracted directory to the dataset name. Needed for datasets
      # like msrvtt and webis_touche2020 where the extracted directory name is
      # different from the dataset name.
      os.rename(
          os.path.join(input_dir, extracted_dir),
          os.path.join(input_dir, dataset),
      )
  else:
    raise ValueError(f"Unsupported compression type: {compression_type}")


def download_dataset(dataset: str, download_dir: str) -> None:
  """Downloads the dataset from the dataset download link."""

  os.makedirs(download_dir, exist_ok=True)
  zipped_filepath = os.path.join(download_dir, dataset + ".zip")
  if not os.path.exists(zipped_filepath):
    wget.download(DATASET_DOWNLOAD_LINKS[dataset], out=zipped_filepath)
  else:
    print("Skipping downloading as the zip file already exists.")

  if not os.path.exists(os.path.join(download_dir, dataset)):
    extract_dataset(dataset, download_dir, _COMPRESSION_TYPE.value)
  else:
    print("Skipping extracting as the dataset already exists.")


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


def update_mm_loft_dataset(
    input_dir: str,
    resource_mapping: dict[str, str],
) -> None:
  """Update the LOFT dataset with the missing fields."""
  for length in DATASET_LENGTHS:
    # Loading the corpus file.
    target_corpus_file = os.path.join(input_dir, length, "corpus.jsonl")
    passages = []
    with open(target_corpus_file, "r") as f:
      for line in f:
        passage = json.loads(line)
        resource_id = passage["pid"]
        passage["metadata"]["img_paths"] = resource_mapping[resource_id]
        passages.append(passage)

    # Writing the corpus file.
    with open(target_corpus_file, "w") as f:
      for passage in passages:
        json.dump(passage, f)
        f.write("\n")
      print(f"Wrote to {target_corpus_file}.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  download_dataset(_DATASET.value, os.path.join(_INPUT_DIR.value, "source"))
  if _DATASET.value in ["msrvtt"]:
    resource_mapping = extract_video_resource(
        os.path.join(_INPUT_DIR.value, "source"),
        os.path.join(_INPUT_DIR.value, "resource"),
    )
    update_mm_loft_dataset(_INPUT_DIR.value, resource_mapping)
  elif _DATASET.value in ["fiqa", "msmarco", "quora", "webis_touche2020"]:
    qid2text, pid2text = load_dataset(_DATASET.value, _INPUT_DIR.value)
    update_loft_dataset(qid2text, pid2text, _INPUT_DIR.value)
  else:
    raise ValueError(
        f"Preprocessor for dataset {_DATASET.value} not available."
    )


if __name__ == "__main__":
  app.run(main)
