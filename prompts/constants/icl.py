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

"""Prompt constants for ICL."""

############################ Corpus Instruction ################################
LONG_ICL_BENCH_DIALOGUE_RE_LABELS = [
    "per:alternate_names",
    "per:alumni",
    "per:place_of_residence",
    "per:employee_or_member_of",
    "per:girl/boyfriend",
    "per:title",
    "per:positive_impression",
    "gpe:residents_of_place",
    "org:employees_or_members",
    "per:children",
    "per:parents",
    "per:siblings",
    "per:spouse",
    "per:friends",
    "per:negative_impression",
    "per:client",
    "per:pet",
    "per:place_of_work",
    "per:boss",
    "per:subordinate",
    "per:acquaintance",
    "per:roommate",
    "per:dates",
    "per:other_family",
    "per:age",
    "per:visited_place",
    "gpe:visitors_of_place",
    "per:origin",
    "per:neighbor",
    "per:works",
    "per:schools_attended",
    "org:students",
    "per:major",
    "per:date_of_birth",
]

CORPUS_INSTRUCTION = {
    "bbh": """
Please answer the following questions and ensure you follow a consistent format. In particular, ensure your final answer always looks like `Output: ['your_answer_here']`.
""".strip(),
    "long_icl_bench_dialogue_re": (
        """
Given the dialogue, please find the name pair entities in the dialogue and their corresponding relation types in the strict format of the given examples. Please only strictly choose from the following relation types (note that the number of entities has to strictly have the same value as  the number of respective relation):
""".strip()
        + "\n"
        + "\n".join(LONG_ICL_BENCH_DIALOGUE_RE_LABELS)
        + "\n\n"
        + """Note that expected output is a series of relations from the provided relation list each in a new line.\nDo NOT include any information other than the relations separated by a new line.\nDo not print <ANALYSIS END> and <EXAMPLE END> markers.\nNote that the number of the relations should strictly match the number of entity pairs provided.\nYou can use the examples below to learn how to find the correct relations:
        """.strip()
    ),
}

CORPUS_FORMAT = {
    "bbh": "{input}\nOutput: {target}\n",
    "long_icl_bench_dialogue_re": """
{target_proposition}<EXAMPLE START>\n{dialogue}\n<ANALYSIS START>\nThe list of {pair_length} entity pairs are:\n{pair_list}\nThe {pair_length} respective relations between each entity pairs are:{relation_list}{ending_notation}""".rstrip(),
}

TARGET_PROPOSITION = (
    "Now look at the dialogue below and mark the relations for the given"
    " entity pairs:\n\n"
)
ENDING_NOTATION = "\n<ANALYSIS END>\n<EXAMPLE END>\n"
