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

"""Prompt constants for RAG."""

from prompts.constants import common

# Fall back to another similar prompt if it is not found from the constants.
PROMPT_MAPPER = {
    "hotpotqa": "nq",
    "musique": "nq",
    "qampari": "nq",
    "quest": "nq",
    "topiocqa": "nq",
    "romqa": "nq",
}


############################# Corpus Formats ###################################

# All datasets have title.
CORPUS_FORMAT_SIMPLE = {
    "nq": common.CORPUS_FORMAT,
}

CORPUS_FORMAT_ECHO = {
    "nq": common.CORPUS_FORMAT_ECHO,
}

CORPUS_FORMAT_REMOVED = {
    "nq": "",
}


CORPUS_FORMAT_RAG = {
    "nq": common.CORPUS_FORMAT_RAG,
}

############################ Corpus Instruction ################################

FORMATTING_INSTRUCTION = """
Your final answer should be in a list, in the following format:
Final Answer: ['answer1', 'answer2', ...]
If there is only one answer, it should be in the format:
Final Answer: ['answer']
"""

CLOSED_BOOK_CORPUS_INSTRUCTION = {
    "nq": """
You will be given a query, and your goal is to answer the query.
""".strip(),
}

CORPUS_INSTRUCTION = {
    "nq": """
You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query, and your goal is to answer the query based on the documents you have read.
""".strip(),
}

BASELINE_CORPUS_INSTRUCTION = {
    "nq": """
You will be given a query and a list of documents. Your goal is to answer the query based on the documents you have read.
""".strip(),
}

############################# Query Formats ####################################
CLOSED_BOOK_QUERY_FORMAT = {
    "nq": """
====== Now let's start! ======
Can you answer the query? Format the answers into a list.
query: {query}
""".strip(),
}


QUERY_FORMAT_NO_COT = {
    "nq": """
Based on the documents above, can you answer the following query? Format the answer into a list.
query: {query}
""".strip(),
}


QUERY_FORMAT_SIMPLE = {
    "nq": """
Based on the documents above, can you answer the following query? Print out the ID and TITLE of the documents you use to answer. Then format the answers into a list.
query: {query}
""".strip(),
}

QUERY_FORMAT_RAG = {
    "nq": """
Based on the documents above, can you answer the following query? Print out the passage number and TITLE of the documents you use to answer. Then format the answers into a list.
query: {query}
""".strip(),
}


QUERY_FORMAT_SIMPLE_REVERSE = {
    k: v.replace("ID and TITLE", "TITLE and ID").replace(
        "ID and CONTENT", "CONTENT and ID"
    )
    for k, v in QUERY_FORMAT_SIMPLE.items()
}

QUERY_FORMAT_WITH_COT = {
    "nq": """
====== Now let's start! ======
Which document is most relevant to the query and can answer the query? Think step-by-step and then format the answers into a list.
query: {query}
""".strip(),
}

QUERY_FORMAT_NO_COT = {
    "nq": """
Based on the documents above, answer the following query? Format the answers into a list.
query: {query}
""".strip(),
}

QUERY_FORMAT_NO_COT_BASELINE = """
====== Now let's start! ======
Based on the documents below, answer the following query? Format the answers into a list.
query: {query}
documents:
""".strip()

################################ Few-shot Formats ##############################
CLOSED_BOOK_FEW_SHOT_QUERY_FORMAT = {
    "nq": """
====== Example {example_id} ======
Can you answer the query? Format the answers into a list.
query: {query}
""".strip(),
}

FEW_SHOT_QUERY_FORMAT_0 = {
    "nq": """
====== Example {example_id} ======
Based on the documents above, can you answer the following query? Format the answers into a list.
query: {query}
""".strip(),
}

FEW_SHOT_QUERY_FORMAT_WITH_COT = {
    "nq": """
====== Example {example_id} ======
Which document is most relevant to the query and can answer the query? Think step-by-step and then format the answers into a list.
query: {query}
""".strip(),
}

FEW_SHOT_QUERY_FORMAT_NO_COT_BASELINE = """
====== Example {example_id} ======
Based on the documents below, answer the following query? Format the answers into a list.
query: {query}
documents:
""".strip()


################## Few-Shot Example Reasoning Formats #######################
FEW_SHOT_EXAMPLE_COT_FORMAT_SIMPLE = {
    "nq": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
}
FEW_SHOT_EXAMPLE_COT_FORMAT_SIMPLE_REVERSE = {
    "nq": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE,
}
FEW_SHOT_EXAMPLE_COT_FORMAT_RAG = {
    "nq": common.FEW_SHOT_EXAMPLE_ANSWER_RAG,
}
