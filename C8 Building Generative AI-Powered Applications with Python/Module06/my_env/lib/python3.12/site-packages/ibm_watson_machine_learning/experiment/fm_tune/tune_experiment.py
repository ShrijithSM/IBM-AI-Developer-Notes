#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum

from ibm_watson_machine_learning.foundation_models.prompt_tuner import PromptTuner
from ibm_watson_machine_learning.foundation_models.utils.enums import PromptTuningTypes, PromptTuningInitMethods, \
    TuneExperimentTasks
from ibm_watson_machine_learning.foundation_models.utils.utils import get_model_specs_with_prompt_tuning_support, _check_model_state
from ibm_watson_machine_learning.experiment.base_experiment import BaseExperiment
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning import APIClient

from .tune_runs import TuneRuns


class TuneExperiment(BaseExperiment):
    """TuneExperiment class for tuning models with prompts.

    :param credentials: credentials to Watson Machine Learning instance
    :type credentials: dict

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio Space
    :type space_id: str, optional

    :param verify: user can pass as verify one of following:

        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.experiment import TuneExperiment

        experiment = TuneExperiment(
            credentials={
                "apikey": "...",
                "iam_apikey_description": "...",
                "iam_apikey_name": "...",
                "iam_role_crn": "...",
                "iam_serviceid_crn": "...",
                "instance_id": "...",
                "url": "https://us-south.ml.cloud.ibm.com"
            },
            project_id="...",
            space_id="...")
    """

    def __init__(self,
                 credentials: dict,
                 project_id: str = None,
                 space_id: str = None,
                 verify=None) -> None:

        self.client = APIClient(credentials, verify=verify)
        if not self.client.CLOUD_PLATFORM_SPACES and self.client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        if project_id:
            self.client.set.default_project(project_id)
        else:
            self.client.set.default_space(space_id)

        self.PromptTuningTypes = PromptTuningTypes
        self.PromptTuningInitMethods = PromptTuningInitMethods

        self.Tasks = self._get_tasks_enum()  # Note: Dynamically create enum with supported ENUM Tasks

        self.runs = TuneRuns(client=self.client)

    def runs(self, *, filter: str) -> 'TuneRuns':
        """Get the historical tuning runs but with name filter.

        :param filter: filter, user can choose which runs to fetch specifying tuning name
        :type filter: str

        **Examples**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import TuneExperiment

            experiment = TuneExperiment(...)
            experiment.runs(filter='prompt tuning name').list()
        """
        return TuneRuns(client=self.client, filter=filter)

    def prompt_tuner(self,
                     name: str,  # Note: Rest API does not require name,
                     task_id: str,
                     description: str = None,
                     base_model: str = None,
                     accumulate_steps: int = None,
                     batch_size: int = None,
                     init_method: str = None,
                     init_text: str = None,
                     learning_rate: float = None,
                     max_input_tokens: int = None,
                     max_output_tokens: int = None,
                     num_epochs: int = None,
                     verbalizer: str = None,
                     tuning_type: str = None,
                     auto_update_model: bool = True,
                     group_by_name: bool = False) -> PromptTuner:
        """Initialize a PromptTuner module.

        :param name: name for the PromptTuner
        :type name: str

        :param task_id: task that is targeted for this model. Example: `experiment.Tasks.CLASSIFICATION`

            Possible values:

            - experiment.Tasks.CLASSIFICATION: 'classification' (default)
            - experiment.Tasks.QUESTION_ANSWERING: 'question_answering'
            - experiment.Tasks.SUMMARIZATION: 'summarization'
            - experiment.Tasks.RETRIEVAL_AUGMENTED_GENERATION: 'retrieval_augmented_generation'
            - experiment.Tasks.GENERATION: 'generation'
            - experiment.Tasks.CODE_GENERATION_AND_CONVERSION: 'code'
            - experiment.Tasks.EXTRACTION: 'extraction

        :type task_id: str

        :param description: description
        :type description: str, optional

        :param base_model: model id of the base model for this prompt tuning. Example: google/flan-t5-xl
        :type base_model: str, optional

        :param accumulate_steps: Number of steps to be used for gradient accumulation. Gradient accumulation 
            refers to a method of collecting gradient for configured number of steps instead of updating 
            the model variables at every step and then applying the update to model variables. 
            This can be used as a tool to overcome smaller batch size limitation. 
            Often also referred in conjunction with "effective batch size". Possible values: 1 ≤ value ≤ 128,
            default value: 16
        :type accumulate_steps: int, optional

        :param batch_size: The batch size is a number of samples processed before the model is updated. 
            Possible values: 1 ≤ value ≤ 16, default value: 16
        :type batch_size: int, optional

        :param init_method: `text` method requires `init_text` to be set. Allowable values: [`random`, `text`],
            default value: `random`
        :type init_method: str, optional

        :param init_text: initialization text to be used if `init_method` is set to `text` otherwise this will be ignored.
        :type init_text: str, optional

        :param learning_rate: learning rate to be used while tuning prompt vectors. Possible values: 0.01 ≤ value ≤ 0.5,
            default value: 0.3
        :type learning_rate: float, optional

        :param max_input_tokens: maximum length of input tokens being considered. Possible values: 1 ≤ value ≤ 256,
            default value: 256
        :type max_input_tokens: int, optional

        :param max_output_tokens: maximum length of output tokens being predicted. Possible values: 1 ≤ value ≤ 128
            default value: 128
        :type max_output_tokens: int, optional

        :param num_epochs: number of epochs to tune the prompt vectors, this affects the quality of the trained model.
            Possible values: 1 ≤ value ≤ 50, default value: 20
        :type num_epochs: int, optional

        :param verbalizer: verbalizer template to be used for formatting data at train and inference time. 
            This template may use brackets to indicate where fields from the data model must be rendered. 
            The default value is "{{input}}" which means use the raw text, default value: `Input: {{input}} Output:`
        :type verbalizer: str, optional

        :param tuning_type: type of Peft (Parameter-Efficient Fine-Tuning) config to build. 
            Allowable values: [`experiment.PromptTuningTypes.PT`], default value: `experiment.PromptTuningTypes.PT`
        :type tuning_type: str, optional

        :param auto_update_model: define if model should be automatically updated, default value: `True`
        :type auto_update_model: bool, optional

        :param group_by_name: define if tunings should be grouped by name, default value: `False`
        :type group_by_name: bool, optional

        **Examples**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import TuneExperiment

            experiment = TuneExperiment(...)

            prompt_tuner = experiment.prompt_tuner(
                name="prompt tuning name",
                task_id=experiment.Tasks.CLASSIFICATION,
                base_model='google/flan-t5-xl',
                accumulate_steps=32,
                batch_size=16,
                learning_rate=0.2,
                max_input_tokens=256,
                max_output_tokens=2,
                num_epochs=6,
                tuning_type=experiment.PromptTuningTypes.PT,
                verbalizer="Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: ",
                auto_update_model=True)
        """

        task_id, base_model = [enum_possible_field.value if isinstance(enum_possible_field, Enum) else enum_possible_field for enum_possible_field in (task_id, base_model)]

        prompt_tuning_supported_models = [model_spec['model_id'] for model_spec in
                                          get_model_specs_with_prompt_tuning_support(
                                              url=self.client.wml_credentials.get('url')).get('resources', [])]

        if base_model not in prompt_tuning_supported_models:
            raise WMLClientError(
                f"Base model '{base_model}' is not supported. Supported models: {prompt_tuning_supported_models}")
        
        # check if model is in constricted mode
        _check_model_state(self.client.wml_credentials.get('url'), base_model)

        prompt_tuner = PromptTuner(name=name,
                                   task_id=task_id,
                                   description=description,
                                   base_model=base_model,
                                   accumulate_steps=accumulate_steps,
                                   batch_size=batch_size,
                                   init_method=init_method,
                                   init_text=init_text,
                                   learning_rate=learning_rate,
                                   max_input_tokens=max_input_tokens,
                                   max_output_tokens=max_output_tokens,
                                   num_epochs=num_epochs,
                                   tuning_type=tuning_type,
                                   verbalizer=verbalizer,
                                   auto_update_model=auto_update_model,
                                   group_by_name=group_by_name)

        prompt_tuner._client = self.client

        return prompt_tuner

    def _get_tasks_enum(self):
        try:
            from ibm_watson_machine_learning.foundation_models.utils.utils import get_all_supported_tasks_dict
            return Enum(value='TuneExperimentTasks',
                        names=get_all_supported_tasks_dict(url=self.client.wml_credentials.get('url')))
        except Exception:
            return TuneExperimentTasks
