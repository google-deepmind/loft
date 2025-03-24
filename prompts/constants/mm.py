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

"""Prompt constants for multimodal retrieval."""
from prompts.constants import common as common_constants

# Fall back to another similar prompt if it is not found from the constants.
PROMPT_MAPPER = {
    "fleurs_en_tts": "fleurs_tts",
    "fleurs_es_tts": "fleurs_tts",
    "fleurs_hi_tts": "fleurs_tts",
    "fleurs_zh_tts": "fleurs_tts",
    "fleurs_fr_tts": "fleurs_tts",
    "fleurs_en_stt": "fleurs_stt",
    "fleurs_es_stt": "fleurs_stt",
    "fleurs_hi_stt": "fleurs_stt",
    "fleurs_zh_stt": "fleurs_stt",
    "fleurs_fr_stt": "fleurs_stt",
}

############################ Corpus Instruction ################################
CLOSED_BOOK_CORPUS_INSTRUCTION = {
    "oven": """
You will be given a input image and a question related to the image, and your goal is to find most relevant Wikipedia entry that can be used to best answer the question. Output the Wikipedia title."
""".strip(),
}

CORPUS_INSTRUCTION = {
    "oven": """
You will be given a list of Wikipedia entries which contains Wikipedia ID, Title and Description image. You need to watch carefully and memorize all of them. Then you will be given a input image and a question related to the image, and your goal is to find most relevant Wikipedia entry from the list that can be used to best answer the question. First output the Wikipedia title then output the Wikipedia ID."
""".strip(),
    "msrvtt": """
You will be given a list of videos which contains the video ID and video content (present as sequence of images, with timestamp in text), You need to wath carefully and memorize all of them. Then you will be given a text query, and your goal is to find most relevant video from the list that can best answer the question. Output the corresponding video ID.
""".strip(),
    "flickr30k": """
You will be given a list of images. You need to watch carefully and memorize all of them. Then you will be given a new sentence, and your goal is to find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""".strip(),
    "mscoco": """
You will be given a list of images. You need to watch carefully and memorize all of them. Then you will be given a new sentence, and your goal is to find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""".strip(),
    "fleurs_tts": """
You will be given a list of audio which contains Audio ID and audio. You need to listen carefully and memorize all of them. Then you will be given a transcript, and your goal is to find most relevant audio from the list that matches the given transcript. Print out the Audio ID of the audio presented in the list.

AUDIO CORPUS
""".strip(),
    "fleurs_stt": """
You will be given a list of transcripts which contains Transcript ID and transcript. You need to read carefully and memorize all of them. Then you will be given an audio, and your goal is to find most relevant transcript from the list that matches the given audio. Print out the Transcript ID of the transcript presented in the list.

TRANSCRIPT CORPUS
""".strip(),
}
CORPUS_FORMAT = {
    "oven": "ID: {pid} | TITLE: {title}",
    "msrvtt": "Video ID: {pid}",
    "flickr30k": "{pid}",
    "mscoco": "{pid}",
    "fleurs_tts": "Audio ID: {pid}",
    "fleurs_stt": "Transcript ID: {pid} | Transcript: {passage}",
}
CORPUS_FORMAT_RAG = {
    "oven": "ID: {pid} | TITLE: {title} | DESCRIPTION: {passage} | IMAGE:",
}

############################# Query Formats ####################################
CLOSED_BOOK_QUERY_FORMAT_PREFIX = {
    "oven": """
====== Now let's start! ======
Given a input Image and a Question, find the most relevant Wikipedia entry for the given question. First output the ID then output the TITLE. Then format the Wikipedia IDs into a list in Final Answer.
""".strip(),
}

QUERY_FORMAT_PREFIX = {
    "oven": """
====== Now let's start! ======
Given a input Image and a Question, find the most relevant Wikipedia entry from the above list for the given question. First output the ID then output the TITLE. Then format the Wikipedia IDs into a list in Final Answer.
""",
    "msrvtt": """
====== Now let's start! ======
Given the text query, find the most relevant video entry from the above list, and output the corresponding Video ID into a list in Final Answer.
""",
    "flickr30k": """
====== Now let's start! ======
Given a sentence, find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""",
    "mscoco": """
====== Now let's start! ======
Given a sentence, find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""",
    "fleurs_tts": """
====== Now let's start! ======
Given a transcript, find most relevant audio from the above list of audio in the corpus. Print out the Audio ID of the audio and then format the Audio ID into a list in Final Answer.
""",
    "fleurs_stt": """
====== Now let's start! ======
Given an audio, find most relevant transcript from the above list of transcript in the corpus. Print out the Transcript ID of the audio and then format the Transcript ID into a list in Final Answer.
Audio:
""",
}

CLOSED_BOOK_QUERY_FORMAT_SUFFIX = {
    "oven": """
Question: {query}
The following wikipedia entry can answer the question:
""".strip(),
}

QUERY_FORMAT_SUFFIX = {
    "oven": """
Question: {query}
The following wikipedia entry can answer the question:
""",
    "msrvtt": """
Query: {query}
""",
    "flickr30k": """
Sentence: {query}
The most relevant image is:
""".strip(),
    "mscoco": """
Sentence: {query}
The most relevant image is:
""".strip(),
    "fleurs_tts": "Transcript: {query}",
    "fleurs_stt": "",
}

################################ Few-shot Formats ##############################
FEW_SHOT_QUERY_FORMAT_PREFIX = {
    "oven": """
====== Example {example_id} ======
Given a input Image and a Question, find the most relevant Wikipedia entry from the above list for the given question. First output the ID then output the TITLE. Then format the Wikipedia IDs into a list in Final Answer.
""".strip(),
    "msrvtt": """
====== Example {example_id} ======
Given the text query, find the most relevant video from the above list, output the Video ID.
""".strip(),
    "flickr30k": """
====== Example {example_id} ======
Given a sentence, find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""".strip(),
    "mscoco": """
====== Example {example_id} ======
Given a sentence, find most relevant image from the list for the given sentence. Print out the image index, which is presented before image in the corpus.
""".strip(),
    "fleurs_tts": """
====== Example {example_id} ======
Given a transcript, find most relevant audio from the above list of audio in the corpus. Print out the Audio ID of the audio and then format the Audio ID into a list in Final Answer.
""".strip(),
    "fleurs_stt": """
====== Example {example_id} ======
Given an audio, find most relevant transcript from the above list of transcript in the corpus. Print out the Transcript ID of the audio and then format the Transcript ID into a list in Final Answer.
Audio:
""".strip(),
}

FEW_SHOT_QUERY_FORMAT_SUFFIX = {
    "oven": """Question: {query}
The following wikipedia entry can answer the question:
""".strip(),
    "msrvtt": """
Query: {query}
""".strip(),
    "flickr30k": """
Sentence: {query}
The most relevant image is:
""".strip(),
    "mscoco": """
Sentence: {query}
The most relevant image is:
""".strip(),
    "fleurs_tts": "Transcript: {query}",
    "fleurs_stt": "",
}

FEW_SHOT_EXAMPLE_ANSWER_FORMAT = {
    "oven": common_constants.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE,
    "msrvtt": common_constants.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_ONLY,
    "flickr30k": common_constants.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_ONLY,
    "mscoco": common_constants.FEW_SHOT_EXAMPLE_ANSWER_FORMAT_ID_ONLY,
    "fleurs_tts": "Audio ID: {pid}",
    "fleurs_stt": "Transcript ID: {pid}"
}
