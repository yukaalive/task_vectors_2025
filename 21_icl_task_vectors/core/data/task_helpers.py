import string
from typing import Dict

from core.data.tasks.increment_task import IncrementTask
from core.data.tasks.list_operation_task import ListOperationTask
from core.data.tasks.token_operation_task import TokenOprationTask
from core.data.tasks.mapping_task import MappingTask
from core.data.tasks.translation_task import TranslationTask
from core.data.tasks.task import Task

from transformers import PreTrainedTokenizer

TASK_TYPE_TO_CLASS = {
    "increment": IncrementTask,
    "list_operation": ListOperationTask,
    "token_operation": TokenOprationTask,
    "mapping": MappingTask,
    "translation": TranslationTask,
    # "sentiment": SentimentTask,
}


ALL_TASKS = {
    # マルチトークン版（ja系のみ）
    # "translation_ja_en": {
    #     "task_type": "translation",
    #     "task_kwargs": {"mapping_type": "translation", "mapping_name": "ja_en_multi", "allow_prefix": True},
    # },
    # "translation_en_ja": {
    #     "task_type": "translation",
    #     "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_ja_multi", "allow_prefix": True},
    # },
    # マルチトークン版（JESC）
    "translation_ja_en_jesc": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "ja_en_multi_jesc", "allow_prefix": True},
    },
    "translation_en_ja_jesc": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_ja_multi_jesc", "allow_prefix": True},
    },
    # マルチトークン版（Easy）
    "translation_ja_en_easy": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "ja_en_multi_easy", "allow_prefix": True},
    },
    "translation_en_ja_easy": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_ja_multi_easy", "allow_prefix": True},
    },
    # シングルトークン版
    "translation_ja_en_single": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "ja_en_single", "allow_prefix": True},
    },
    "translation_en_ja_single": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_ja_single", "allow_prefix": True},
    },
    # Algorithmic
    "algorithmic_next_letter": {
        "task_type": "increment",
        "task_kwargs": {"increment": +1},
    },
    "algorithmic_prev_letter": {
        "task_type": "increment",
        "task_kwargs": {"increment": -1},
    },
    "algorithmic_list_first": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "first", "list_lenghts": range(2, 5)},
    },
    "algorithmic_list_last": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "last", "list_lenghts": range(2, 5)},
    },
    "algorithmic_list_min": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "min", "list_lenghts": range(2, 5), "elements_space": list(string.digits)},
    },
    "algorithmic_list_max": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "max", "list_lenghts": range(2, 5), "elements_space": list(string.digits)},
    },
    "algorithmic_list_length": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "length", "list_lenghts": range(1, 4)},
    },
    "algorithmic_to_upper": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "to_upper", "input_space": list(string.ascii_lowercase)},
    },
    "algorithmic_to_lower": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "to_lower", "input_space": list(string.ascii_uppercase)},
    },
    "algorithmic_char_to_int": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "char_to_int", "input_space": list(string.ascii_lowercase[:9])},
    },  # low performance
    "algorithmic_int_to_char": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "int_to_char", "input_space": list(string.digits[1:])},
    },
    # Translation（シングルトークン版 - ja以外）
    "translation_fr_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "fr_en_single", "allow_prefix": True},
    },
    "translation_it_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "it_en_single", "allow_prefix": True},
    },
    "translation_es_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "es_en_single", "allow_prefix": True},
    },
    "translation_en_fr": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_fr_single", "allow_prefix": True},
    },
    "translation_en_it": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_it_single", "allow_prefix": True},
    },
    "translation_en_es": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_es_single", "allow_prefix": True},
    },
    # Linguistic
    "linguistic_present_simple_gerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_gerund"},
    },
    "linguistic_present_simple_past_simple": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_simple"},
    },
    "linguistic_present_simple_past_perfect": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_perfect"},
    },
    "linguistic_singular_plural": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "singular_plural"},
    },
    "linguistic_plural_singular": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "plural_singular"},
    },
    "linguistic_antonyms": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "antonyms"},
    },
    # Knowledge
    "knowledge_country_capital": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "country_capital", "allow_prefix": True},
    },
    "knowledge_person_language": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "person_language", "allow_prefix": True},
    },
    "knowledge_location_continent": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "location_continent", "allow_prefix": True},
    },
    "knowledge_location_religion": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "location_religion", "allow_prefix": True},
    },
    # "sentiment": {
    #     "task_type": "sentiment",
    #     "task_kwargs": {"allow_prefix": True},
    # },
}

# task_typeをキーにして、TASK_TYPE_TO_CLASSからクラスを探し、task_kwargsを引数としてオブジェクト作る
def get_task(task_type: str, task_kwargs: Dict[str, str], tokenizer: PreTrainedTokenizer) -> Task:
    task = TASK_TYPE_TO_CLASS[task_type](**task_kwargs, tokenizer=tokenizer)
    return task

# task_nameというキーに対するのtask_type,task_kwargsを取り出す
def get_task_by_name(tokenizer: PreTrainedTokenizer, task_name: str) -> Task:
    task_args = ALL_TASKS[task_name]
    task = get_task(task_args["task_type"], task_args["task_kwargs"], tokenizer)
    return task

def get_all_tasks(tokenizer: PreTrainedTokenizer) -> Dict[str, Task]:
    # ALL_TASKSの中からtask_nameを取り出してget_task_by_nameにわたす
    tasks = {task_name: get_task_by_name(tokenizer, task_name) for task_name in ALL_TASKS}
    return tasks



# 以下のような辞書を作成している
# task = TranslationTask(
#     mapping_type="translation",
#     mapping_name="ja_en",
#     allow_prefix=True,
#     tokenizer=tokenizer
# )

# tasks = {} 
# for task_name in ALL_TASKS:
#     task_object = get_task_by_name(tokenizer, task_name)
#     tasks[task_name] = task_object
# return tasks