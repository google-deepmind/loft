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

"""Evaluation mapper."""

import types

from evaluation import loft_evaluation
from evaluation import rag
from evaluation import retrieval
from evaluation import sql
from evaluation import utils


EVALUATION_TASKS = types.MappingProxyType({
    "rag": rag.RagEvaluation(
        config=loft_evaluation.EvaluationConfig(
            process_model_response_fns=[
                utils.convert_to_str,
                utils.normalize_answers,
            ],
            process_gold_answer_fns=[utils.normalize_answer],
        )
    ),
    "multi_value_rag": rag.MultiValueRagEvaluation(
        config=loft_evaluation.EvaluationConfig(
            process_model_response_fns=[
                utils.convert_to_str,
                utils.normalize_answers,
            ],
            process_gold_answer_fns=[utils.normalize_answer],
        )
    ),
    "retrieval": retrieval.RetrievalEvaluation(
        config=loft_evaluation.EvaluationConfig(
            process_model_response_fns=[
                utils.normalize_passage_ids,
            ],
            process_gold_answer_fns=[
                utils.extract_gold_passage_ids,
                utils.normalize_passage_id,
            ],
        )
    ),
    "sql": sql.SqlEvaluation(
        config=loft_evaluation.EvaluationConfig(
            process_model_response_fns=[],
            process_gold_answer_fns=[],
        )
    ),
})
