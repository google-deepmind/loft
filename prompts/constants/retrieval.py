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

"""Prompt constants for retrieval."""

from prompts.constants import common

# Fall back to another similar prompt if it is not found from the constants.
PROMPT_MAPPER = {
    "fever": "scifact",
    "msmarco": "fiqa",
    "musique": "hotpotqa",
    "qampari": "hotpotqa",
    "quest": "hotpotqa",
    "topiocqa": "nq",
}

############################ Corpus Instruction ################################

FORMATTING_INSTRUCTION = """
Your final answer should be a list of IDs, in the following format:
Final Answer: [id1, id2, ...]
If there is only one ID, it should be in the format:
Final Answer: [id1]

If there is no perfect answer output the closest one. Do not give an empty final answer.
"""

CLOSED_BOOK_CORPUS_INSTRUCTION = {
    "hotpotqa": """
You will be given a query, and your goal is to find the titles of Wikipedia articles that can help answer the query.
""".strip(),
    "nq": """
You will be given a query, and your goal is to find the titles of Wikipedia articles that can help answer the query.
""".strip(),
}

CORPUS_INSTRUCTION = {
    "arguana": """
You will be given a list of statements. You need to read carefully and understand all of them. Then you will be given a claim, and your goal is to find all statements from the list that can counterargue the claim.
""".strip(),
    "fiqa": """
You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. Print out the ID and CONTENT of each document.
""".strip(),
    "hotpotqa": """
You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query that may require you to use 1 or more documents to find the answer. Your goal is to find all documents from the list that can help answer the query. Print out the ID and TITLE of each document.
""".strip(),
    "nq": """
You will be given a list of documents. You need to read carefully and understand all of them. Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. Print out the ID and TITLE of each document.
""".strip(),
    "quora": """
You will be given a list of questions. You need to read carefully and understand all of them. Then you will be given a new question, and your goal is to find all questions from the list that are near duplicates of the new question. Print out the ID and CONTENT of each question.
""".strip(),
    "scifact": """
You will be given a list of passages. You need to read carefully and understand all of them. Then you will be given a claim, and your goal is to find all passages from the list that can help verify the claim as true of false. Print out the ID and TITLE of each passage.
""".strip(),
    "webis_touche2020": """
You will be given a list of arguments. You need to read carefully and understand all of them. Then you will be given a controversial debating topic, and your goal is to find arguments from the list that's relevant to the topic. Print out the ID and TITLE of each argument.
""".strip(),
}

############################# Corpus Formats ###################################
CORPUS_FORMAT_SIMPLE = {
    "arguana": common.CORPUS_FORMAT_NOTITLE,
    "fiqa": common.CORPUS_FORMAT_NOTITLE,
    "hotpotqa": common.CORPUS_FORMAT,
    "nq": common.CORPUS_FORMAT,
    "quora": common.CORPUS_FORMAT_NOTITLE,
    "scifact": common.CORPUS_FORMAT,
    "webis_touche2020": common.CORPUS_FORMAT,
}

CORPUS_FORMAT_ECHO = {
    "arguana": common.CORPUS_FORMAT_ECHO_NOTITLE,
    "fiqa": common.CORPUS_FORMAT_ECHO_NOTITLE,
    "hotpotqa": common.CORPUS_FORMAT_ECHO,
    "nq": common.CORPUS_FORMAT_ECHO,
    "quora": common.CORPUS_FORMAT_ECHO_NOTITLE,
    "scifact": common.CORPUS_FORMAT_ECHO,
    "webis_touche2020": common.CORPUS_FORMAT_ECHO,
}
################## Few-Shot Example Reasoning Formats #######################
FEW_SHOT_EXAMPLE_COT_FORMAT_SIMPLE = {
    "arguana": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_NOTITLE,
    "fiqa": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_NOTITLE,
    "hotpotqa": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
    "nq": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
    "quora": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_NOTITLE,
    "scifact": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
    "webis_touche2020": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
}
FEW_SHOT_EXAMPLE_COT_FORMAT_SIMPLE_REVERSE = {
    "arguana": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE_NOTITLE,
    "fiqa": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE_NOTITLE,
    "hotpotqa": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE,
    "nq": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE,
    "quora": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE_NOTITLE,
    "scifact": common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE,
    "webis_touche2020": (
        common.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE_NOTITLE
    ),
}

################ Query Formats for few-shots and test queries ################
CLOSED_BOOK_QUERY_FORMAT = {
    "hotpotqa": """
Which Wikipedia article is most relevant to the query and can answer the query? You will need two articles to answer the query. Format the titles into a list.
query: {query}
The following articles can help answer the query:
""".strip(),
    "nq": """
Which Wikipedia article is most relevant to the query and can answer the query? Format the titles into a list.
query: {query}
The following articles can help answer the query:
""".strip(),
}

QUERY_FORMAT_SIMPLE = {
    "arguana": """
Given a claim, which statements provide a counterargument? Print out the ID and CONTENT of each statement. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
claim: {query}
The following statements can counterargue the claim:
""".strip(),
    "fiqa": """
Which document is most relevant to answering the query? Print out the ID and CONTENT of the document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following documents can answer the query:
""".strip(),
    "hotpotqa": """
Which documents can help answer the query? Print out the ID and TITLE of each document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following documents can help answer the query:
""".strip(),
    "nq": """
Which document is most relevant to answer the query? Print out the ID and TITLE of the document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following documents can help answer the query:
""".strip(),
    "quora": """
Given the following query, which existing question is most similar to it? Print out the ID and CONTENT of the question. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following existing questions are most similar to the given query:
""".strip(),
    "scifact": """
Which passage is most relevant to the claim, and can help verify the claim as true or false? Print out the ID and TITLE of the document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
claim: {query}
The following passages can help verify this sentence:
""".strip(),
    "webis_touche2020": """
Which argument is most relevant to the query? Print out the ID and TITLE of the argument. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following argument is most relevant to the query:
""".strip(),
}

QUERY_FORMAT_SIMPLE_REVERSE = {
    k: v.replace("ID and TITLE", "TITLE and ID").replace(
        "ID and CONTENT", "CONTENT and ID"
    )
    for k, v in QUERY_FORMAT_SIMPLE.items()
}

QUERY_FORMAT_NO_COT = {
    k: v.replace("ID and TITLE", "ID").replace("ID and CONTENT", "ID")
    for k, v in QUERY_FORMAT_SIMPLE.items()
}


QUERY_FORMAT_WITH_COT = {
    "hotpotqa": """
Which documents are relevant to answering the query? Let's think step-by-step to find two relevant documents. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
""".strip(),
    "nq": """
Which document is most relevant to answering the query? Think step-by-step and then format the document IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: {query}
The following documents can help answer the query:
""".strip(),
}
