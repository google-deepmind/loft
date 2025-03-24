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

"""Utils for LOFT Prompts."""

from collections.abc import Callable, Sequence
import random
from typing import Any, Optional

from prompts import prompt_registry
from prompts.constants import common as common_constants
import utils as loft_utils


ContentChunk = loft_utils.ContentChunk
MimeType = loft_utils.MimeType
PromptRegistry = prompt_registry.PromptRegistry
Passage = dict[str, Any]

DEFAULT_TITLE = "None"


def get_text_chunk(text: str, is_train: bool = False) -> ContentChunk:
  """Get text chunk for the given text."""
  return ContentChunk(
      _text=f"{text}\n",
      mime_type=MimeType.TEXT,
      role="assistant" if is_train else "user",
  )


def add_text_chunks(
    chunks: list[ContentChunk], texts: Sequence[str], **kwargs
) -> None:
  """Adds text chunks to the list of chunks."""
  del kwargs
  for text in texts:
    chunks.append(get_text_chunk(text))


def move_documents_in_corpus(
    corpus: dict[str, Passage],
    pids: list[list[str]] | None,
    documents_position: float,
) -> dict[str, Passage]:
  """Moves pids documents to the specified location in the corpus.

  Args:
    corpus: Corpus. A dictionary of id (str) to Passage.
    pids: The PIDs of the documents to move.
    documents_position: where to insert the document within the corpus. Is [0,
      1], with 0.0 being the beginning and 1.0 the end of the corpus.

  Returns:
    The updated corpus with the pids documents at the specified location.
  """
  if pids is None:
    return corpus
  flat_pids = []
  for pid in pids:
    flat_pids.extend(pid)
  # Remove moved document from corpus.
  answers = {}
  for pid in flat_pids:
    answers[pid] = corpus[pid]
  corpus_without_moved_docs = {
      pid: passage for (pid, passage) in corpus.items() if pid not in flat_pids
  }

  # Get pid positions to insert.
  insert_position = int(documents_position * len(corpus_without_moved_docs))
  # Insert. All moved documents are grouped together for now.
  corpus_without_moved_keys = list(corpus_without_moved_docs.keys())
  final_corpus = {
      k: v
      for (k, v) in list(corpus_without_moved_docs.items())[:insert_position]
  }
  for pid, answer in answers.items():
    final_corpus[pid] = answer
  for i in range(insert_position, len(corpus_without_moved_docs)):
    final_corpus[corpus_without_moved_keys[i]] = corpus_without_moved_docs[
        corpus_without_moved_keys[i]
    ]
  return final_corpus


def add_corpus_chunks(
    chunks: list[ContentChunk],
    corpus_document_boundaries: list[tuple[int, int]],
    loft_data: loft_utils.LOFTData,
    pid_mapper: dict[str, str],
    gold_pids: list[list[str]],
    corpus_format: str = common_constants.CORPUS_FORMAT,
    fixed_pid_mapper: Optional[dict[str, str]] = None,
    add_image_chunk: bool = False,
    shuffle_seed: int | None = None,
    gold_documents_position: float | None = None,
    few_shot_documents_position: float | None = None,
    few_shot_pids: list[list[str]] | None = None,
    **kwargs,
) -> None:
  """Adds corpus chunks to the list of chunks."""
  del kwargs
  if fixed_pid_mapper:
    pid_mapper.update(fixed_pid_mapper)
  corpus = loft_data.corpus
  # First, shuffle the corpus if requested.
  if shuffle_seed is not None:
    corpus_items = list(corpus.items())
    random.Random(shuffle_seed).shuffle(corpus_items)
    corpus = {k: v for (k, v) in corpus_items}
  # Then, position the gold document at the specified location if requested.
  if gold_documents_position is not None:
    corpus = move_documents_in_corpus(
        corpus, pids=gold_pids, documents_position=gold_documents_position
    )
  # Then, position the few-shot examples at the specified location if requested.
  if few_shot_documents_position is not None:
    corpus = move_documents_in_corpus(
        corpus,
        pids=few_shot_pids,
        documents_position=few_shot_documents_position,
    )
  corpus_items = (
      list(corpus.items())
      if fixed_pid_mapper is None
      else [(pid, corpus[pid]) for pid, _ in fixed_pid_mapper.items()]
  )
  for pid, passage_data in corpus_items:
    chunk_text = corpus_format.format(
        pid=pid_mapper[pid],
        title=passage_data["title"] if passage_data["title"] else DEFAULT_TITLE,
        passage=passage_data["passage"],
    )
    doc_start = len(chunks)

    if "audio_path" in passage_data["metadata"] and not passage_data["passage"]:
      chunks.append(get_text_chunk(corpus_format.format(pid=pid_mapper[pid])))
      chunks.append(
          ContentChunk(
              _path=passage_data["metadata"]["audio_path"],
              mime_type=MimeType.AUDIO_WAV,
          )
      )
      continue
    if "audio_path" in passage_data["metadata"] and passage_data["passage"]:
      chunks.append(
          get_text_chunk(
              corpus_format.format(
                  pid=pid_mapper[pid],
                  passage=passage_data["passage"],
              )
          )
      )
      continue

    # Skip empty text chunks.
    if chunk_text:
      chunks.append(get_text_chunk(chunk_text))

    if add_image_chunk:
      if "img_path" in passage_data["metadata"]:
        chunks.extend([
            ContentChunk(
                _path=img_path,
                mime_type=MimeType.IMAGE_JPEG,
            )
            for img_path in passage_data["metadata"]["img_path"]
        ])
      elif "image_bytes" in passage_data["metadata"]:
        chunks.append(
            ContentChunk(
                _embedding=passage_data["metadata"]["soft_tokens"],
                mime_type=MimeType.IMAGE_JPEG,
                _image_bytes=passage_data["metadata"]["image_bytes"],
            )
        )

    # Add chunk indices if a chunk was added for this document.
    if doc_start != len(chunks):
      corpus_document_boundaries.append((doc_start, len(chunks)))


def add_corpus_chunks_and_query_turns_from_few_shot_examples(
    chunks: list[ContentChunk],
    corpus_document_boundaries: list[tuple[int, int]],
    loft_data: loft_utils.LOFTData,
    pid_mapper: dict[str, str],
    gold_pids: list[list[str]],
    corpus_format: str = common_constants.CORPUS_FORMAT,
    fixed_pid_mapper: Optional[dict[str, str]] = None,
    add_image_chunk: bool = False,
    shuffle_seed: int | None = None,
    gold_documents_position: float | None = None,
    few_shot_documents_position: float | None = None,
    few_shot_prompt_name: str = "",
    few_shot_max_examples: int | None = None,
    few_shot_examples: Optional[list[loft_utils.Example]] | None = None,
    **kwargs,
) -> None:
  """Combines add_corpus_chunks and add_query_turns_from_examples."""
  # First, get the few-shot (if not provided) so we know what their pids are.
  if few_shot_examples is None:
    few_shot_examples = PromptRegistry.get_examples(few_shot_prompt_name)

  if few_shot_max_examples is not None:
    few_shot_examples = few_shot_examples[:few_shot_max_examples]
  few_shot_pids = []
  for example in few_shot_examples:
    few_shot_pids.extend([list(pids) for pids in example.gold_pids])
  # Then, add corpus chunks.
  add_corpus_chunks(
      chunks=chunks,
      corpus_document_boundaries=corpus_document_boundaries,
      loft_data=loft_data,
      pid_mapper=pid_mapper,
      gold_pids=gold_pids,
      corpus_format=corpus_format,
      fixed_pid_mapper=fixed_pid_mapper,
      add_image_chunk=add_image_chunk,
      shuffle_seed=shuffle_seed,
      gold_documents_position=gold_documents_position,
      few_shot_documents_position=few_shot_documents_position,
      few_shot_pids=few_shot_pids,
      **kwargs,
  )

  # Lastly, add the few-shot examples.
  for example in few_shot_examples:
    query_turns = example.query_turns
    for query_turn in query_turns:
      chunks.extend(list(query_turn.chunks))


def add_corpus_chunks_from_qid(
    chunks: list[ContentChunk],
    loft_data: loft_utils.LOFTData,
    pid_mapper: dict[str, str],
    qid: str,
    shuffle_targets: bool = False,
    corpus_format: str = common_constants.CORPUS_FORMAT,
    **kwargs,
) -> None:
  """Adds corpus chunks to the list of chunks."""
  del kwargs
  del shuffle_targets
  for pid in loft_data.metadata[qid]["candidate_pids"]:
    passage_data = loft_data.corpus[pid]
    chunks.append(
        get_text_chunk(
            corpus_format.format(
                pid=pid_mapper[pid],
                title=passage_data["title"]
                if passage_data["title"]
                else DEFAULT_TITLE,
                passage=passage_data["passage"],
            )
        )
    )


def add_few_shot_corpus_chunks(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_format: str,
    use_example_id: bool = False,
    example_id: Optional[str] = None,
    **kwargs,
) -> None:
  """Adds query turn to the list of query turns."""
  del query_format, use_example_id, example_id
  candidate_pids = loft_data.metadata[qid]["candidate_pids"]
  num_topk_targets = kwargs.get("num_candidates", None)
  is_multi_turn = kwargs.get("is_multi_turn", False)
  seed = kwargs.get("shuffle_seed", None)
  corpus_format = kwargs.get("corpus_format", common_constants.CORPUS_FORMAT)

  if not is_multi_turn:
    candidate_pids = [candidate_pids]

  for turn_idx, final_pids in enumerate(candidate_pids):
    if num_topk_targets is None:
      num_topk_targets = len(final_pids)
    final_pids = final_pids[:num_topk_targets]

    if seed is not None:
      random.seed(seed)
      random.shuffle(final_pids)

    for i, pid in enumerate(final_pids):
      passage_data = loft_data.corpus[pid]
      query_turns[turn_idx].append(
          get_text_chunk(
              corpus_format.format(
                  pid=str(i),
                  title=passage_data["title"]
                  if passage_data["title"]
                  else DEFAULT_TITLE,
                  passage=passage_data["passage"],
              )
          )
      )


def add_test_corpus_chunks(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_format: str,
    **kwargs):
  """Adds test example specific corpus chunks to the test example."""
  del query_format
  candidate_pids = loft_data.metadata[qid]["candidate_pids"]
  num_topk_targets = kwargs.get("num_candidates", None)
  is_multi_turn = kwargs.get("is_multi_turn", False)
  seed = kwargs.get("shuffle_seed", None)
  corpus_format = kwargs.get("corpus_format", common_constants.CORPUS_FORMAT)

  if not is_multi_turn:
    candidate_pids = [candidate_pids]

  for turn_idx, final_pids in enumerate(candidate_pids):
    if num_topk_targets is None:
      num_topk_targets = len(final_pids)
    final_pids = final_pids[:num_topk_targets]

    if seed is not None:
      random.seed(seed)
      random.shuffle(final_pids)

    for i, pid in enumerate(final_pids):
      passage_data = loft_data.corpus[pid]
      query_turns[turn_idx].append(
          get_text_chunk(
              corpus_format.format(
                  pid=str(i),
                  title=passage_data["title"]
                  if passage_data["title"]
                  else DEFAULT_TITLE,
                  passage=passage_data["passage"],
              )
          )
      )


def get_bbh_example_chunk(
    example: dict[str, Any],
    corpus_format: str,
    add_target: bool = False,
    target_proposition: str = "",  # pylint: disable=unused-argument
    ending_notation: str = "",  # pylint: disable=unused-argument
) -> ContentChunk:
  return get_text_chunk(
      corpus_format.format(
          input=example["input"],
          target=example["target"].split(" ") if add_target else "",
      )
  )


def get_long_icl_bench_dialogue_re_example_chunk(
    example: dict[str, Any],
    corpus_format: str,
    add_target: bool = False,
    target_proposition: str = "",
    ending_notation: str = "",
) -> ContentChunk:
  """Reformat the example results for evaluations."""
  pair_list = [
      r["subject_entity"] + "\t" + r["object_entity"]
      for r in example["relations"]
  ]
  relation_list = [r["relation"] for r in example["relations"]]
  relation_list_str = "\n" +"\n".join(relation_list)
  return get_text_chunk(
      corpus_format.format(
          target_proposition=target_proposition if (not add_target) else "",
          dialogue=example["dialogue"],
          pair_list="\n".join(pair_list),
          pair_length=len(pair_list),
          relation_list=relation_list_str if add_target else "",
          ending_notation=ending_notation if add_target else "",
      )
  )


def add_many_shot_chunks(
    chunks: list[ContentChunk],
    corpus_document_boundaries: list[tuple[int, int]],
    loft_data: loft_utils.LOFTData,
    chunk_format_fn: Callable[..., Any],
    corpus_format: str = common_constants.CORPUS_FORMAT,
    add_target: bool = True,
    target_proposition: str = "",
    ending_notation: str = "",
    **kwargs,
) -> None:
  """Adds corpus chunks to the list of chunks."""
  del kwargs
  for corpus_item in loft_data.corpus.items():
    _, example = corpus_item
    doc_start = len(chunks)
    chunks.append(
        chunk_format_fn(
            example,
            corpus_format,
            add_target=add_target,
            target_proposition=target_proposition,
            ending_notation=ending_notation,
        )
    )
    corpus_document_boundaries.append((doc_start, len(chunks)))


def add_query_turns(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_format: str,
    use_example_id: bool = False,
    example_id: Optional[str] = None,
    **kwargs,
) -> None:
  """Adds query turn to the list of query turns."""
  del kwargs
  queries = loft_data.queries[qid]
  if isinstance(queries, str):
    queries = [queries]

  for turn_idx, query in enumerate(queries):
    turn_example_id = (
        f"{example_id}-{turn_idx+1}" if len(queries) > 1 else example_id
    )
    query_turns.append([
        get_text_chunk(
            query_format.format(
                query=query,
                # Used for few-shot examples.
                example_id=turn_example_id if use_example_id else None,
            )
        )
    ])


def add_query_turns_with_corpus(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_format: str,
    corpus_format: str,
    follow_up_query_format: str,
    corpus_passage_getter: Callable[[dict[str, Any]], str] | None = None,
    **kwargs,
) -> None:
  """Adds query turn to the list of query turns."""
  del kwargs
  queries = loft_data.queries[qid]
  if isinstance(queries, str):
    queries = [queries]

  for turn_idx, curr_query_turn in enumerate(queries):
    if turn_idx == 0:
      corpus_texts = []
      for pid in loft_data.metadata[qid]["candidate_pids"]:
        passage_data = loft_data.corpus[pid]
        if corpus_passage_getter:
          passage = corpus_passage_getter(passage_data)
        else:
          passage = passage_data["passage"]
        corpus_texts.append(
            corpus_format.format(
                title=passage_data["title"],
                passage=passage,
            )
        )
      corpus_texts = "\n\n".join(corpus_texts)
      turn_chunks = [
          get_text_chunk(
              query_format.format(
                  query=curr_query_turn,
                  corpus=corpus_texts,
              )
          )
      ]
    else:
      query_format_no_corpus = follow_up_query_format.format(
          query=curr_query_turn
      )
      turn_chunks = [get_text_chunk(query_format_no_corpus)]
    query_turns.append(turn_chunks)


def add_query_turns_for_many_shot(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    chunk_format_fn: Callable[..., Any],
    corpus_format: str,
    target_proposition: str = "",
    ending_notation: str = "",
    **kwargs,
) -> None:
  """Adds query turn to the list of query turns."""
  del kwargs
  queries = loft_data.queries[qid]
  queries = [queries]  # Make it multi-turn.

  query_turns.append([
      chunk_format_fn(
          query,
          corpus_format,
          add_target=False,
          target_proposition=target_proposition,
          ending_notation=ending_notation,
      )
      for query in queries
  ])


def add_multimodal_query_turns(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_prefix_format: str,
    query_suffix_format: str,
    use_example_id: bool = False,
    example_id: Optional[str] = None,
    **kwargs,
) -> None:
  """Adds images to the list of query turns."""
  del kwargs
  query = loft_data.queries[qid]
  img_paths, audio_paths, soft_tokens = None, None, None
  if "img_path" in loft_data.metadata[qid]:
    img_paths = [loft_data.metadata[qid]["img_path"]]
  if "soft_tokens" in loft_data.metadata[qid]:
    soft_tokens = [loft_data.metadata[qid]["soft_tokens"]]
  if "audio_path" in loft_data.metadata[qid] and not query:
    audio_paths = loft_data.metadata[qid]["audio_path"]
  if soft_tokens is not None and img_paths is not None:
    assert len(soft_tokens) == len(img_paths)
    mm_chunks = [
        ContentChunk(
            _path=img_path,
            mime_type=MimeType.IMAGE_JPEG,
            _embedding=embedding,
        )
        for img_path, embedding in zip(img_paths, soft_tokens)
    ]
  elif img_paths is not None:
    mm_chunks = [
        ContentChunk(
            _path=img_path,
            mime_type=MimeType.IMAGE_JPEG,
        )
        for img_path in img_paths
    ]
  elif audio_paths is not None:
    mm_chunks = [
        ContentChunk(
            _path=audio_paths,
            mime_type=MimeType.AUDIO_WAV,
        )
    ]
  else:
    mm_chunks = []

  prefix_chunks = get_text_chunk(
      query_prefix_format.format(
          # Used for few-shot examples.
          example_id=example_id if use_example_id else None,
      )
  )
  suffix_chunks = get_text_chunk(
      query_suffix_format.format(
          query=query,
      )
  )
  if query:
    query_turns.append([prefix_chunks] + mm_chunks + [suffix_chunks])
  else:
    query_turns.append([prefix_chunks] + mm_chunks)


def add_images_in_query_turns(
    query_turns: list[list[ContentChunk]],
    loft_data: loft_utils.LOFTData,
    qid: str,
    query_prefix_format: str,
    query_suffix_format: str,
    use_example_id: bool = False,
    example_id: Optional[str] = None,
    **kwargs,
) -> None:
  """Adds images to the list of query turns."""
  del kwargs
  query = loft_data.queries[qid]
  assert "img_path" in loft_data.metadata[qid]
  img_path = loft_data.metadata[qid]["img_path"]
  soft_tokens = loft_data.metadata[qid]["soft_tokens"]

  query_turns.append(
      [
          get_text_chunk(
              query_prefix_format.format(
                  # Used for few-shot examples.
                  example_id=example_id
                  if use_example_id
                  else None,
              )
          )
      ]
      + [
          ContentChunk(
              _path=img_path,
              mime_type=MimeType.IMAGE_JPEG,
              _embedding=soft_tokens,
          )
      ]
      + [
          get_text_chunk(
              query_suffix_format.format(
                  query=query,
              )
          ),
      ]
  )


def convert_pids_into_gold_answers(
    gold_answers: list[Any],
    pid_mapper: dict[str, str],
    **kwargs,
) -> None:
  """Converts gold answers to the correct format."""
  del kwargs
  # Answers are given for each turn.
  for turn_idx in range(len(gold_answers)):
    gold_answers[turn_idx] = [
        pid_mapper[gold_answer[0]] for gold_answer in gold_answers[turn_idx]
    ]


def append_reasoning_to_query_turns(
    query_turns: list[list[ContentChunk]],
    qid: str,
    gold_pids: list[Any],
    gold_answers: list[Any],
    loft_data: loft_utils.LOFTData,
    pid_mapper: dict[str, str],
    qid2reasoning: Optional[
        dict[str, dict[str, Sequence[Sequence[tuple[str, str]]]]]
    ] = None,
    reasoning_format: Optional[str] = None,
    reasoning_version: str = "best",
    **kwargs,
) -> None:
  """Appends reasoning to the list of query turns."""
  is_train = kwargs.get("is_train", False)
  if not qid2reasoning and not reasoning_format:
    raise ValueError(
        "Either qid2reasoning or reasoning_format must be provided."
    )
  if qid2reasoning:
    qid2reasoning = qid2reasoning[reasoning_version]
    reasoning_formats_for_all_turns = [
        [format_str for _, format_str in qid2reasoning_single_turn]
        for qid2reasoning_single_turn in qid2reasoning[qid]
    ]
    reasoning_original_answers_for_all_turns = [
        [original_answer for original_answer, _ in qid2reasoning_single_turn]
        for qid2reasoning_single_turn in qid2reasoning[qid]
    ]
    for turn_idx, _ in enumerate(query_turns):
      if not all(
          reasoning_pid in gold_pids[turn_idx]
          for reasoning_pid in reasoning_original_answers_for_all_turns[
              turn_idx
          ]
      ):
        raise ValueError(
            "Original answers from the reasoning must be a subset of the gold"
            " answers."
        )
  else:
    reasoning_formats_for_all_turns = [
        [reasoning_format] * len(gold_pids[turn_idx])
        for turn_idx in range(len(gold_pids))
    ]

  for turn_idx, _ in enumerate(query_turns):
    for gold_pid, reasoning_format in zip(
        gold_pids[turn_idx],
        reasoning_formats_for_all_turns[turn_idx],
    ):
      gold_answer = gold_answers[turn_idx]
      args_dict = {
          "pid": pid_mapper[gold_pid],
          "title": (
              loft_data.corpus[gold_pid]["title"]
              if loft_data.corpus[gold_pid]["title"]
              else DEFAULT_TITLE
          ),
          "passage": loft_data.corpus[gold_pid]["passage"],
      }
      if "{answers}" in reasoning_format:
        args_dict["answers"] = gold_answer  # Later used for RAG.
      query_turns[turn_idx].append(
          get_text_chunk(
              reasoning_format.format(**args_dict), is_train=is_train
          )
      )


def add_query_output(
    query_turns: list[list[ContentChunk]],
    qid: str,
    loft_data: loft_utils.LOFTData,
    output_format: str,
    **kwargs,
) -> None:
  """Append answer to the test query turns."""
  del kwargs
  for turn_idx, _ in enumerate(query_turns):
    answers = loft_data.answers[qid]
    answer_formatted = ['"' + answer + '"' for answer in answers]
    answer_list = "[" + ", ".join(answer_formatted) + "]"
    output = output_format.format(final_answer=answer_list)
    query_turns[turn_idx].append(
        ContentChunk(mime_type=MimeType.TEXT, _text=output, role="assistant")
    )


def append_gold_answers_to_query_turns(
    query_turns: list[list[ContentChunk]],
    gold_answers: list[Any],
    loft_data: loft_utils.LOFTData,
    qid: str,
    answer_format: str,
    **kwargs,
) -> None:
  """Appends gold answers to the list of query turns."""
  del kwargs, loft_data, qid
  if isinstance(gold_answers[0], str):
    gold_answers = [gold_answers]

  if len(gold_answers) != len(query_turns):
    raise ValueError(
        "Length of gold answers does not match length of query turns."
    )

  for turn_idx, _ in enumerate(query_turns):
    query_turns[turn_idx].append(
        get_text_chunk(
            answer_format.format(
                final_answer=gold_answers[turn_idx],
            )
        )
    )


def append_gold_answers_to_corpus(
    chunks: list[ContentChunk],
    loft_data: loft_utils.LOFTData,
    answer_format: str = common_constants.CORPUS_FORMAT,
    **kwargs,
) -> None:
  """Appends gold answers at the end of few short examples."""
  assert "qid" in kwargs
  qid = kwargs["qid"]
  gold_answers = loft_data.answers[qid]
  chunks.append(get_text_chunk(answer_format.format(final_answer=gold_answers)))


def add_query_turns_from_examples(
    chunks: list[ContentChunk],
    prompt_name: str,
    max_examples: int | None = None,
    **kwargs,
) -> None:
  """Adds examples as chunks."""
  del kwargs
  examples = PromptRegistry.get_examples(prompt_name)
  if max_examples is not None:
    examples = examples[:max_examples]
  for example in examples:
    query_turns = example.query_turns
    for query_turn in query_turns:
      chunks.extend(list(query_turn.chunks))


def add_query_turns_from_baseline_examples(
    chunks: list[ContentChunk],
    prompt_name: str,
    max_examples: int | None = None,
    **kwargs,
) -> None:
  """Adds examples as chunks."""
  del kwargs
  examples = PromptRegistry.get_examples(prompt_name)
  if max_examples is not None:
    examples = examples[:max_examples]
  for example in examples:
    for query_turn in example.query_turns:
      chunks.extend(list(query_turn.chunks))
    chunks.extend(list(example.context_chunks))
