#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from string import Formatter
from typing import Any, Sequence, List, Dict, Union, Optional, Generator
from dataclasses import dataclass, KW_ONLY, asdict
from json import loads as json_loads
import warnings

from ibm_watson_machine_learning._wrappers import requests
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.messages.messages import Messages
from ibm_watson_machine_learning.wml_client_error import WMLClientError, InvalidMultipleArguments, InvalidValue
from ibm_watson_machine_learning.utils import NextResourceGenerator
from ibm_watson_machine_learning.utils.autoai.utils import load_file_from_file_system_nonautoai
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.lifecycle import SpecStates
from ibm_watson_machine_learning import APIClient


@dataclass
class PromptTuningParams:
    base_model: dict
    _: KW_ONLY
    accumulate_steps: int = None
    batch_size: int = None
    init_method: str = None
    init_text: str = None
    learning_rate: float = None
    max_input_tokens: int = None
    max_output_tokens: int = None
    num_epochs: int = None
    task_id: str = None
    tuning_type: str = None
    verbalizer: str = None

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def _get_foundation_models_spec(url, operation_name, additional_params: dict = None):
    params = {"version": "2023-09-30"}
    if additional_params:
        params.update(additional_params)
    response = requests.get(url,
                            params=params,
                            headers={'X-WML-User-Client': 'PythonClient'})
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise WMLClientError(Messages.get_message(message_id="fm_prompt_tuning_no_foundation_models"), logg_messages=False)
    else:
        msg = f"{operation_name} failed. Reason: {response.text}"
        raise WMLClientError(msg)


def get_model_specs(url: str, model_id: Optional[str] = None) -> dict:
    """
    Operations to query the details of the deployed foundation models.

    :param url: environment url
    :type url: str

    :param model_id: Id of the model, defaults to None (all models specs are returned).
    :type model_id: Optional[str, ModelTypes], optional

    :return: list of deployed foundation model specs
    :rtype: dict

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models import get_model_specs

        # GET ALL MODEL SPECS
        get_model_specs(
            url="https://us-south.ml.cloud.ibm.com"
            )

        # GET MODEL SPECS BY MODEL_ID
        get_model_specs(
            url="https://us-south.ml.cloud.ibm.com",
            model_id="google/flan-ul2"
            )
    """
    try:
        if model_id:
            if isinstance(model_id, ModelTypes):
                model_id = model_id.value

            try:
                return [res for res in _get_foundation_models_spec(f"{url}/ml/v1/foundation_model_specs",
                                                                   "Get available foundation models")['resources']
                        if res['model_id'] == model_id][0]
            except WMLClientError:  # Remove on CPD 5.0 release
                return [res for res in _get_foundation_models_spec(f"{url}/ml/v1-beta/foundation_model_specs",
                                                                   "Get available foundation models")['resources']
                        if res['model_id'] == model_id][0]
        else:
            try:
                return _get_foundation_models_spec(f"{url}/ml/v1/foundation_model_specs",
                                                   "Get available foundation models")
            except WMLClientError:  # Remove on CPD 5.0 release
                return _get_foundation_models_spec(f"{url}/ml/v1-beta/foundation_model_specs",
                                                   "Get available foundation models")
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"), e)


def get_custom_model_specs(credentials: dict = None,
                           api_client: 'APIClient' = None,
                           model_id: Optional[str] = None,
                           limit: int = 100, 
                           asynchronous: bool = False, 
                           get_all: bool = False,
                           verify=None) -> Union[dict, Generator[dict, None, None]]:
    """Get details on available custom model(s) as dict or as generator (``asynchronous``). 
    If ``asynchronous`` or ``get_all`` is set, then ``model_id`` is ignored.

    :param credentials: credentials to watsonx.ai instance
    :type credentials: dict, optional

    :param api_client: API client to connect to service
    :type api_client: APIClient, optional

    :param model_id: Id of the model, defaults to None (all models specs are returned).
    :type model_id: str, optional

    :param limit:  limit number of fetched records. Possible values: 1 ≤ value ≤ 200, default value: 100
    :type limit: int, optional

    :param asynchronous:  if True, it will work as a generator
    :type asynchronous: bool, optional

    :param get_all:  if True, it will get all entries in 'limited' chunks
    :type get_all: bool, optional

    :param verify: user can pass as verify one of following:

        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    :return: details of supported custom models
    :rtype: dict

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models import get_custom_model_specs

        get_custom_models_spec(api_client=client)
        get_custom_models_spec(credentials=credentials)
        get_custom_models_spec(api_client=client, model_id='mistralai/Mistral-7B-Instruct-v0.2')
        get_custom_models_spec(api_client=client, limit=20)
        get_custom_models_spec(api_client=client, limit=20, get_all=True)
        for spec in get_custom_model_specs(api_client=client, limit=20, asynchronous=True, get_all=True):
            print(spec, end="")

    """
    warnings.warn("Model needs to be first stored via client.repository.store_model(model_id, meta_props=metadata)"
                  " and deployed via client.deployments.create(asset_id, metadata) to be used.")

    if credentials is None and api_client is None:
        raise InvalidMultipleArguments(params_names_list=["credentials", "api_client"],
                                       reason="None of the arguments were provided.")
    elif credentials:
        client = APIClient(wml_credentials=credentials, verify=verify)
    else:
        client = api_client

    url = client.wml_credentials['url']

    params = client._params(skip_for_create=True)
    if limit < 1 or limit > 200:
        raise InvalidValue(value_name="limit", reason=f"The given value {limit} is not in the range <1, 200>")
    else:
        params.update({'limit': limit})

    if asynchronous or get_all:
        resource_generator = NextResourceGenerator(client,
                                                   url=url,
                                                   href="ml/v4/custom_foundation_models",
                                                   params=params,
                                                   _all=get_all)

        if asynchronous:
            return resource_generator

        resources = []
        for entry in resource_generator:
            resources.extend(entry['resources'])
        return {
            "resources": resources
        }

    response = requests.get(f"{url}/ml/v4/custom_foundation_models", 
                            params=params,
                            headers=client._get_headers()) 
    if response.status_code == 200:
        if model_id:
            resources = [res for res in response.json()['resources'] if res['model_id'] == model_id]
            return resources[0] if resources else {}
        else:
            return response.json()
    elif response.status_code == 404:
        raise WMLClientError(Messages.get_message(url, message_id="custom_models_no_model_specs"), url)
    else:
        msg = f"Getting failed. Reason: {response.text}"
        raise WMLClientError(msg)


def get_model_lifecycle(url: str, model_id: str) -> Union[list, None]:
    """
    Operation to retrieve the list of model lifecycle data.

    :param url: environment url
    :type url: str

    :param model_id: the type of model to use
    :type model_id: str

    :return: list of deployed foundation model lifecycle data
    :rtype: list

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models import get_model_lifecycle

        get_model_lifecycle(
            url="https://us-south.ml.cloud.ibm.com",
            model_id="ibm/granite-13b-instruct-v2"
            )
    """
    model_specs = get_model_specs(url=url)
    model_spec = next((model_metadata for model_metadata in model_specs.get('resources', [])
                       if model_metadata.get('model_id') == model_id), None)
    return model_spec.get('lifecycle') if model_spec is not None else None


def _check_model_state(url, model_id):

    default_warning_template = (
        "Model '{model_id}' is in {state} state from {start_date} until {withdrawn_start_date}. "
        "IDs of alternative models: {alternative_model_ids}. "
        "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp")

    lifecycle = get_model_lifecycle(url, model_id)

    modes_list = [ids.get("id") for ids in (lifecycle or [])]
    deprecated_or_constricted_warning_template_cpd = (
            "Model '{model_id}' is in {state} state. "
            "IDs of alternative models: {alternative_model_ids}. "
            "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp")

    if lifecycle and SpecStates.DEPRECATED.value in modes_list:
        model_lifecycle = next((el for el in lifecycle if el.get('id') == SpecStates.DEPRECATED.value), None)
        if model_lifecycle.get('since_version'):
            warnings.warn(deprecated_or_constricted_warning_template_cpd.format(
                model_id=model_id,
                state=(model_lifecycle.get('label') or SpecStates.DEPRECATED.value),
                alternative_model_ids=', '.join(model_lifecycle.get('alternative_model_ids', ["None"]))
            ), category=LifecycleWarning)
        else:
            warnings.warn(default_warning_template.format(
                model_id=model_id,
                state=(model_lifecycle.get('label') or SpecStates.DEPRECATED.value),
                start_date=model_lifecycle.get('start_date'),
                withdrawn_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.WITHDRAWN.value), None),
                alternative_model_ids=', '.join(model_lifecycle.get('alternative_model_ids', ["None"]))
            ), category=LifecycleWarning)

    elif lifecycle and SpecStates.CONSTRICTED.value in modes_list:
        model_lifecycle = next((el for el in lifecycle if el.get('id') == SpecStates.CONSTRICTED.value), None)
        if model_lifecycle.get('since_version'):
            warnings.warn(deprecated_or_constricted_warning_template_cpd.format(
                model_id=model_id,
                state=(model_lifecycle.get('label') or SpecStates.CONSTRICTED.value),
                alternative_model_ids=', '.join(model_lifecycle.get('alternative_model_ids', ["None"]))
            ), category=LifecycleWarning)
        else:
            warnings.warn(default_warning_template.format(
                model_id=model_id,
                state=(model_lifecycle.get('label') or SpecStates.CONSTRICTED.value),
                start_date=model_lifecycle.get('start_date'),
                withdrawn_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.WITHDRAWN.value), None),
                alternative_model_ids=', '.join(model_lifecycle.get('alternative_model_ids', ["None"]))
            ), category=LifecycleWarning)


def get_model_specs_with_prompt_tuning_support(url: str) -> dict:
    """
    Operations to query the details of the deployed foundation models with prompt tuning support.

    :param url: environment url
    :type url: str

    :return: list of deployed foundation model specs with prompt tuning support
    :rtype: dict

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models import get_model_specs_with_prompt_tuning_support

        get_model_specs_with_prompt_tuning_support(
            url="https://us-south.ml.cloud.ibm.com"
            )
    """
    try:
        try:
            return _get_foundation_models_spec(url=f"{url}/ml/v1/foundation_model_specs",
                                               operation_name="Get available foundation models",
                                               additional_params={"filters": "function_prompt_tune_trainable"})
        except WMLClientError:  # Remove on CPD 5.0 release
            return _get_foundation_models_spec(url=f"{url}/ml/v1-beta/foundation_model_specs",
                                               operation_name="Get available foundation models",
                                               additional_params={"filters": "function_prompt_tune_trainable"})
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"), e)


def get_supported_tasks(url: str) -> dict:
    """
    Operation to retrieve the list of tasks that are supported by the foundation models.

    :param url: environment url
    :type url: str

    :return: list of tasks that are supported by the foundation models
    :rtype: dict

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models import get_supported_tasks

        get_supported_tasks(
            url="https://us-south.ml.cloud.ibm.com"
            )
    """
    try:
        try:
            return _get_foundation_models_spec(f"{url}/ml/v1/foundation_model_tasks",
                                               "Get tasks that are supported by the foundation models.")
        except WMLClientError:  # Remove on CPD 5.0 release
            return _get_foundation_models_spec(f"{url}/ml/v1-beta/foundation_model_tasks",
                                               "Get tasks that are supported by the foundation models.")
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_supported_tasks"), e)


def get_all_supported_tasks_dict(url="https://us-south.ml.cloud.ibm.com") -> dict:
    tasks_dict = dict()
    for task_spec in get_supported_tasks(url).get('resources', []):
        tasks_dict[task_spec['label'].replace("-", "_").replace(" ", "_").upper()] = task_spec['task_id']
    return tasks_dict


def load_request_json(run_id, wml_client, run_params = None) -> dict:
    if run_params is None:
        run_params = wml_client.training.get_details(run_id)

    model_request_path = run_params['entity'].get('results_reference', {}).get('location', {}).get('model_request_path')

    if model_request_path is None:
        raise WMLClientError("Missing model_request_path in run_params. Verify if the training run has been completed.")

    if wml_client.CLOUD_PLATFORM_SPACES:
        results_reference = DataConnection._from_dict(run_params['entity']['results_reference'])

        if run_params['entity']['results_reference']['type'] == DataConnectionTypes.CA:
            results_reference.location.file_name = model_request_path
        else:
            results_reference.location.path = model_request_path

        results_reference.set_client(wml_client)
        request_json_bytes = results_reference.read(raw=True, binary=True)

        # download from cos

    elif wml_client.CPD_version >= 4.8:
        asset_parts = model_request_path.split('/')
        model_request_asset_url = '/'.join(asset_parts[asset_parts.index('assets') + 1:])
        request_json_bytes = load_file_from_file_system_nonautoai(wml_client=wml_client,
                                                                  file_path=model_request_asset_url).read()
    else:
        raise WMLClientError("Unsupported environment for this action")

    return json_loads(request_json_bytes.decode())


def is_training_prompt_tuning(training_id, wml_client):
    """Returns True if training_id is connected to prompt tuning"""
    if training_id is None:
        return False
    run_params = wml_client.training.get_details(training_uid=training_id)
    return bool(run_params['entity'].get('prompt_tuning'))


class TemplateFormatter(Formatter):
    def check_unused_args(self, 
                          used_args: List[(int | str)], 
                          args: Sequence, 
                          kwargs: Dict[str, Any]) -> None:
        """Check for unused args."""
        extra_args = set(kwargs).difference(used_args)
        if extra_args:
            raise KeyError(extra_args)


class HAPDetectionWarning(UserWarning): ...


class PIIDetectionWarning(UserWarning): ...


class LifecycleWarning(UserWarning): ...


class WatsonxLLMDeprecationWarning(UserWarning): ...


def _raise_watsonxllm_deprecation_warning() -> None:
    warnings.warn(f"ibm_watson_machine_learning.foundation_models.extensions.langchain.WatsonxLLM"
                  f" is deprecated and will not be supported in the future. "
                  f"Please import from langchain-ibm instead.\n"
                  "To install langchain-ibm run `pip install -U langchain-ibm`.",
                  category=WatsonxLLMDeprecationWarning,
                  stacklevel=2
                  )
