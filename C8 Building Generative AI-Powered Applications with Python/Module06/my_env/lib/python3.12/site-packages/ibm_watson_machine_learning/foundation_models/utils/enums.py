#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum

__all__ = [
    "ModelTypes",
    "DecodingMethods",
    "PromptTuningTypes",
    "PromptTuningInitMethods",
    "TuneExperimentTasks",
    "PromptTemplateFormats",
]


class ModelTypes(Enum):
    """Supported foundation models."""
    FLAN_T5_XXL = "google/flan-t5-xxl"
    FLAN_UL2 = "google/flan-ul2"
    MT0_XXL = "bigscience/mt0-xxl"
    GPT_NEOX = 'eleutherai/gpt-neox-20b'
    MPT_7B_INSTRUCT2 = 'ibm/mpt-7b-instruct2'
    STARCODER = 'bigcode/starcoder'
    LLAMA_2_70B_CHAT = 'meta-llama/llama-2-70b-chat'
    LLAMA_2_13B_CHAT = 'meta-llama/llama-2-13b-chat'
    GRANITE_13B_INSTRUCT = 'ibm/granite-13b-instruct-v1'
    GRANITE_13B_CHAT = 'ibm/granite-13b-chat-v1'
    FLAN_T5_XL = 'google/flan-t5-xl'
    GRANITE_13B_CHAT_V2 = 'ibm/granite-13b-chat-v2'
    GRANITE_13B_INSTRUCT_V2 = 'ibm/granite-13b-instruct-v2'
    ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT = 'elyza/elyza-japanese-llama-2-7b-instruct'
    MIXTRAL_8X7B_INSTRUCT_V01_Q = 'ibm-mistralai/mixtral-8x7b-instruct-v01-q'
    CODELLAMA_34B_INSTRUCT_HF = "codellama/codellama-34b-instruct-hf"
    GRANITE_20B_MULTILINGUAL = "ibm/granite-20b-multilingual"


class DecodingMethods(Enum):
    """Supported decoding methods for text generation."""
    SAMPLE = "sample"
    GREEDY = "greedy"


class PromptTuningTypes:
    PT = "prompt_tuning"


class PromptTuningInitMethods:
    """Supported methods for prompt initialization in prompt tuning."""
    RANDOM = "random"
    TEXT = "text"
    # PRESET ?


class TuneExperimentTasks(Enum):
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    RETRIEVAL_AUGMENTED_GENERATION = "retrieval_augmented_generation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    CODE_GENERATION_AND_CONVERSION = "code"
    EXTRACTION = "extraction"


class PromptTemplateFormats(Enum):
    """Supported formats of loaded prompt template."""
    PROMPTTEMPLATE = "prompt"
    STRING = "string"
    LANGCHAIN = "langchain"
