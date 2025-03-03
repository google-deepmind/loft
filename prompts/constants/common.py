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

"""Constants for prompt formatting."""

CORPUS_INSTRUCTION = """
You will be given a list of candidates such as documents, images, videos, audios, etc. You need to check them carefully and understand all of them. Then you will be given a query, and your goal is to find all candidates from the list that can help answer the query. Print out the ID of each candidate.
""".strip()

CORPUS_FORMAT = "ID: {pid} | TITLE: {title} | CONTENT: {passage}"
CORPUS_FORMAT_NOTITLE = "ID: {pid} | CONTENT: {passage}"
CORPUS_FORMAT_ECHO = (
    "ID: {pid} | TITLE: {title} | CONTENT: {passage} | END ID: {pid}"
)
CORPUS_FORMAT_ECHO_NOTITLE = "ID: {pid} | CONTENT: {passage} | END ID: {pid}"
CORPUS_FORMAT_ID_CONCAT = 'TITLE: "{title} {pid}" | CONTENT: {passage}'
CORPUS_FORMAT_ID_CONCAT_NOTITLE = 'CONTENT: "{passage} {pid}"'
CORPUS_FORMAT_RAG = (
    "[{pid}] ({title}) CONTENT: {passage}"
)

# Few-shot example answer formats
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE = "ID: {pid} | TITLE: {title}"
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_NOTITLE = "ID: {pid} | CONTENT: {passage}"
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE = "TITLE: {title} | ID: {pid}"
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE_NOTITLE = (
    "CONTENT: {passage} | ID: {pid}"
)
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_ONLY = "ID: {pid}"
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_CONCAT = 'TITLE: "{title} {pid}"'
FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_CONCAT_NOTITLE = 'CONTENT: "{passage} {pid}"'
FEW_SHOT_EXAMPLE_ANSWER_RAG = "[{pid}] ({title})"

# Final answer used for evaluation.
FINAL_ANSWER_FORMAT = "Final Answer: {final_answer}"

FEW_SHOT_SEPARATOR = "====== Example {example_id} ======\n"
TEST_QUERY_SEPARATOR = "====== Now let's start! ======\n"
