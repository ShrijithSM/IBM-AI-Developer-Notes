#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function, annotations

from typing import Optional, List, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import langchain
from dataclasses import dataclass
from datetime import datetime

import pandas

import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import (WMLClientError, ValidationError,
                                                          InvalidValue, InvalidMultipleArguments, PromptVariablesError)
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, PromptTemplateFormats
from ibm_watson_machine_learning.foundation_models.utils.utils import TemplateFormatter


@dataclass
class PromptTemplateLock:
    """Storage for lock object.
    """
    locked: bool
    locked_by: Optional[str] = None


class PromptTemplate:
    """Storage for prompt template parameters.

    :param prompt_id: Id of prompt template, defaults to None.
    :type prompt_id: Optional[str], attribute setting not allowed

    :param created_at: Time the prompt was created (UTC), defaults to None.
    :type created_at: Optional[str], attribute setting not allowed 

    :param lock: Locked state of asset, defaults to None.
    :type lock: Optional[PromptTemplateLock], attribute setting not allowed

    :param is_template: True if prompt is a template, False otherwise; defaults to None.
    :type is_template: Optional[bool], attribute setting not allowed 

    :param name: Prompt template name, defaults to None.
    :type name: Optional[str], optional

    :param model_id: Foundation model id, defaults to None.
    :type model_id: Optional[ModelTypes], optional

    :param model_params: Model parameters, defaults to None.
    :type model_params: Optional[Dict], optional

    :param template_version: Semvar version for tracking in IBM AI Factsheets, defaults to None.
    :type template_version: Optional[str], optional

    :param task_ids: List of task ids, defaults to None.
    :type task_ids: Optional[List[str]], optional

    :param description: Prompt template asset description, defaults to None.
    :type description: Optional[str], optional

    :param input_text: Input text for prompt, defaults to None.
    :type input_text: Optional[str], optional

    :param input_variables: Input variables can be present in fields: `instruction`, 
                            `input_prefix`, `output_prefix`, `input_text`, `examples`
                            and are indentified by braces ('{' and '}'), defaults to None.
    :type temaplate_parameters: (List | Dict[str, Dict[str, str]] | None), optional

    :param instruction: Instruction for model, defaults to None.
    :type instruction: Optional[str], optional

    :param input_prefix: Prefix string placed before input text, defaults to None.
    :type input_prefix: Optional[str], optional

    :param output_prefix: Prefix before model response, defaults to None.
    :type output_prefix: Optional[str], optional

    :param exmaples: Examples may help the model to adjust the response; [[input1, output1], ...], defaults to None.
    :type exmaples: Optional[List[List[str]]], optional

    :param validate_template: If True, the Prompt Template is validated for the presence of input variables, defaults to True.
    :type validate_template: bool, optional


    **Examples**

    Example of invalid Prompt Template:

    .. code-block:: python

        prompt_template = PromptTemplate(input_text='What are the most famous monuments in ?',
                                         input_variables=['country'])

        Traceback (most recent call last):
            ...
        ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

    Example of valid Prompt Template:

    .. code-block:: python

        prompt_template = PromptTemplate(input_text='What are the most famous monuments in {country}?',
                                         input_variables=['country'])

    """

    def __init__(self,
                 name: Optional[str] = None,
                 model_id: Optional[ModelTypes] = None,
                 model_params: Optional[Dict] = None,
                 template_version: Optional[str] = None,
                 task_ids: Optional[List[str]] = None,
                 description: Optional[str] = None,
                 input_text: Optional[str] = None,
                 input_variables: (List | Dict[str, Dict[str, str]] | None) = None,
                 instruction: Optional[str] = None,
                 input_prefix: Optional[str] = None,
                 output_prefix: Optional[str] = None,
                 examples: Optional[List[List[str]]] = None,
                 validate_template: bool = True) -> None:
        self.name = name
        self._prompt_id = None
        self._created_at = None
        self._lock = None
        self._is_template = None
        self.model_id = model_id
        if isinstance(self.model_id, ModelTypes):
            self.model_id = self.model_id.value
        self.model_params = model_params.copy() if model_params is not None else model_params
        self.task_ids = task_ids.copy() if task_ids is not None else task_ids
        self.template_version = template_version
        self.description = description
        self.input_text = input_text
        self.input_variables = input_variables.copy() if input_variables is not None else input_variables
        self.instruction = instruction
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.examples = examples.copy() if examples is not None else examples

        # template validation
        if validate_template:
            self._validation()

    def __repr__(self):
        args = [f"{key}={value!r}" for key, value in self.__dict__.items() 
                if not key.startswith('_') and value is not None]
        return f"{type(self).__name__}({ ', '.join(args)})"
    
    @property
    def prompt_id(self):
        return self._prompt_id

    @property
    def created_at(self):
        return str(datetime.utcfromtimestamp(self._created_at / 1000)).split(".")[0]

    @property
    def lock(self):
        return self._lock

    @property
    def is_template(self):
        return self._is_template

    def _validation(self):
        """Validate template structure.

        :raises ValidationError: raises when input_variables does not fit placeholders in input body.
        """
        input_variables = self.input_variables if self.input_variables is not None else []
        template_text = " ".join(filter(None, [self.instruction,
                                               self.input_prefix,
                                               self.output_prefix]))
        if self.examples:
            for example in self.examples:
                template_text += " ".join(example)
        try:
            def _validate(input_text):
                dummy_inputs = {input_variable: "wx" for input_variable in input_variables}
                TemplateFormatter().format("".join([template_text, input_text]), **dummy_inputs)

            if self.input_text:
                _validate(template_text + self.input_text)
            else:
                if template_text:
                    _validate(template_text)
        except KeyError as key:
            raise ValidationError(key)


class PromptTemplateManager(WMLResource):
    """Instantiate the prompt template manager.

    :param credentials: Credentials to watsonx.ai instance.
    :type credentials: dict

    :param project_id: ID of project
    :type project_id: str

    :param space_id: ID of project
    :type space_id: str

    :param verify: user can pass as verify one of following:
        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    .. note::
        One of these parameters is required: ['project_id ', 'space_id']

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplate, PromptTemplateManager
        from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

        prompt_mgr = PromptTemplateManager(
                        credentials={
                            "apikey": "***",
                            "url": "https://us-south.ml.cloud.ibm.com"
                        },
                        project_id="*****"
                        )

        prompt_template = PromptTemplate(name="My prompt",
                                         model_id=ModelTypes.GRANITE_13B_CHAT_V2,
                                         input_prefix="Human:",
                                         output_prefix="Assistant:",
                                         input_text="What is {object} and how does it work?",
                                         input_variables=['object'],
                                         examples=[['What is the Stock Market?',
                                                    'A stock market is a place where investors buy and sell shares of publicly traded companies.']])

        stored_prompt_template = prompt_mgr.store_prompt(prompt_template)
        print(stored_prompt_template.prompt_id)   # id of prompt template asset
    """

    def __init__(self,
                 credentials: Optional[dict] = None,
                 *,
                 project_id: Optional[str] = None,
                 space_id: Optional[str] = None,
                 verify=None,
                 api_client: APIClient = None) -> None:

        self.project_id = project_id
        self.space_id = space_id
        if credentials:
            self._client = APIClient(credentials, verify=verify)
        elif api_client:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(params_names_list=["credentials", "api_client"],
                                           reason="None of the arguments were provided.")

        if self.space_id is not None and self.project_id is not None:
            raise InvalidMultipleArguments(params_names_list=["project_id", "space_id"],
                                           reason="Both arguments were provided.")
        self.params = {}
        if self.space_id:
            self._client.set.default_space(space_id)
            self.params = {'space_id': self.space_id}
        elif self.project_id:
            self._client.set.default_project(project_id)
            self.params = {'project_id': self.project_id}
        elif api_client:
            if (project_id := self._client.default_project_id):
                self.params = {'project_id': project_id}
            elif (space_id := self._client.default_space_id):
                self.params = {'space_id': space_id}
            else:
                pass
        elif not api_client:
            raise InvalidMultipleArguments(params_names_list=["space_id", "project_id"],
                                           reason="None of the arguments were provided.")
        
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    def _create_request_body(self, prompt_template: PromptTemplate) -> Dict:
        """Method is used to create request body from PromptTemplate object.

        :param prompt_template: Object of type PromptTemplate based on which the request
                                body will be created.
        :type prompt_template: PromptTemplate

        :return: Request body
        :rtype: Dict
        """
        json_data: Dict = {'prompt': dict()}
        if prompt_template.description is not None:
            json_data.update({'description': prompt_template.description})
        if prompt_template.input_variables is not None:
            PromptTemplateManager._validate_type(prompt_template.input_variables, u'input_variables', [dict, list],
                                                 False)
            if isinstance(prompt_template.input_variables, list):
                json_data.update({'prompt_variables': {key: {} for key in prompt_template.input_variables}})
            else:
                json_data.update({'prompt_variables': prompt_template.input_variables})
        if prompt_template.task_ids is not None:
            PromptTemplateManager._validate_type(prompt_template.task_ids, u'task_ids', list, False)
            json_data.update({'task_ids': prompt_template.task_ids})
        if prompt_template.template_version is not None:
            json_data.update({"model_version": {"number": prompt_template.template_version}})

        if prompt_template.input_text:
            PromptTemplateManager._validate_type(prompt_template.input_text, u'input_text', str, False)
            json_data['prompt'].update({'input': [[prompt_template.input_text, '']]})

        PromptTemplateManager._validate_type(prompt_template.model_id, u'model_id', str, True)
        if prompt_template.model_id is not None:
            json_data['prompt'].update({'model_id': prompt_template.model_id})

        if prompt_template.model_params is not None:
            PromptTemplateManager._validate_type(prompt_template.model_params, u'model_parameters', dict, False)
            json_data['prompt'].update({'model_parameters': prompt_template.model_params})

        data: Dict = dict()
        if prompt_template.instruction is not None:
            data.update({'instruction': prompt_template.instruction})

        if prompt_template.input_prefix is not None:
            data.update({'input_prefix': prompt_template.input_prefix})

        if prompt_template.output_prefix is not None:
            data.update({'output_prefix': prompt_template.output_prefix})
        if prompt_template.examples is not None:
            PromptTemplateManager._validate_type(prompt_template.examples, u'examples', list, False)
            data.update({'examples': prompt_template.examples})

        json_data['prompt'].update({'data': data})

        return json_data

    def _from_json_to_prompt(self, response: Dict) -> PromptTemplate:
        """Convert json response to PromptTemplate object.

        :param response: Response body after request operation.
        :type response: Dict

        :return: PromptTemplate object with given details.
        :rtype: PromptTemplate
        """
        prompt_field: Dict = response.get('prompt', dict())
        data_field: Dict = prompt_field.get('data', dict())
        prompt_template = PromptTemplate(name=response.get('name'),
                                         description=response.get('description'),
                                         model_id=prompt_field.get('model_id'),
                                         model_params=prompt_field.get('model_parameters'),
                                         task_ids=response.get("task_ids"),
                                         template_version=response.get("model_version", dict()).get("number"),
                                         input_variables = response.get('prompt_variables'),
                                         input_text = prompt_field.get('input', [[None, None]])[0][0],
                                         instruction = data_field.get('instruction'),
                                         input_prefix = data_field.get('input_prefix'),
                                         output_prefix = data_field.get('output_prefix'),
                                         examples = data_field.get('examples'),
                                         validate_template=False
                                         )

        prompt_template._prompt_id = response.get('id')
        prompt_template._created_at = response.get('created_at')
        if "lock_type" in response.get('lock', dict()):
            del response['lock']["lock_type"]
        prompt_template._lock = PromptTemplateLock(**response.get('lock', {"locked": None,
                                                                           "locked_by": None}))
        prompt_template._is_template = response.get('is_template')

        return prompt_template

    def _get_details(self, limit: Optional[int] = None) -> List:
        """Method retrives details of all prompt templates. If limit is set to None
        then all prompt templates are fetched.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: Optional[int]

        :return: List of prompts metadata 
        :rtype: List
        """
        headers = self._client._get_headers()
        url = self._client.service_instance._href_definitions.get_prompts_all_href()
        json_data = {"query": u"asset.asset_type:wx_prompt",
                     u"sort": "-asset.created_at<string>"}
        if limit is not None:
            if limit < 1:
                raise WMLClientError('Limit cannot be lower than 1.')
            elif limit > 200:
                raise WMLClientError('Limit cannot be larger than 200.')

            json_data.update({'limit': limit})
        else:
            json_data.update({'limit': 200})
        prompts_list = []
        bookmark = True
        while bookmark is not None:
            response = requests.post(url=url, json=json_data,
                                     headers=headers,
                                     params=self.params)
            details_json = self._handle_response(200, "Get next details", response)
            bookmark = details_json.get('next', {'href': None}).get('bookmark', None)
            prompts_list.extend(details_json.get('results', []))
            if limit is not None:
                break
            json_data.update({'bookmark': bookmark})
        return prompts_list

    def _change_lock(self, prompt_id: str, locked: bool, force: bool = False) -> Dict:
        """Change prompt template lock state.

        :param prompt_id: Id of prompt template.
        :type prompt_id: str

        :param locked: New lock state.
        :type locked: bool

        :param force: force lock state overwrite, defaults to False.
        :type force: bool, optional

        :return: Response content after lock state change.
        :rtype: Dict
        """
        headers = self._client._get_headers()
        params = (self.params | {"prompt_id": prompt_id, "force": force})
        json_data = {"locked": locked}

        url = self._client.service_instance._href_definitions.get_prompts_href() + f"/{prompt_id}/lock"
        response = requests.put(url=url,
                                json=json_data,
                                headers=headers,
                                params=params)

        return self._handle_response(200, u'change_lock', response)

    def load_prompt(self,
                    prompt_id: str,
                    astype: PromptTemplateFormats = PromptTemplateFormats.PROMPTTEMPLATE,
                    *,
                    prompt_variables: Optional[Dict[str, str]] = None):
        """Retrive a prompt template asset.

        :param prompt_id: Id of prompt template which is processed.
        :type prompt_id: str

        :param astype: Type of return object.
        :type astype: PromptTemplateFormats

        :param prompt_variables: Dictionary of input variables and values with which input variables will be replaced.
        :type prompt_variables: Dict[str, str]

        :return: Prompt template asset.
        :rtype: PromptTemplate | str | langchain.prompts.PromptTemplate

        **Example**

        .. code-block:: python

            loaded_prompt_template = prompt_mgr.load_prompt(prompt_id)
            loaded_prompt_template_lc = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.LANGCHAIN)
            loaded_prompt_template_string = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.STRING)
        """
        headers = self._client._get_headers()
        params = (self.params | {"prompt_id": prompt_id})
        url = self._client.service_instance._href_definitions.get_prompts_href() + f"/{prompt_id}"

        if isinstance(astype, PromptTemplateFormats):
            astype = astype.value

        if astype == 'prompt':
            response = requests.get(url=url,
                                    headers=headers,
                                    params=params)
            return self._from_json_to_prompt(self._handle_response(200, u'_load_json_prompt', response))
        elif astype in ('langchain', 'string'):
            response = requests.post(url=url + u"/input",
                                     headers=headers,
                                     params=params)
            response_input = self._handle_response(200, u'load_prompt', response).get("input")
            if astype == 'string':
                try:
                    return response_input if prompt_variables is None else response_input.format(**prompt_variables)
                except KeyError as key:
                    raise PromptVariablesError(key)
            else:
                from langchain.prompts import PromptTemplate as LcPromptTemplate
                return LcPromptTemplate.from_template(response_input)
        else:
            raise InvalidValue(u'astype')

    def list(self, *, limit=None) -> pandas.core.frame.DataFrame:
        """List all available prompt templates in the DataFrame format.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: Optional[int]

        :return: Dataframe of fundamental properties of availabale prompts.
        :rtype: pandas.core.frame.DataFram

        **Example**

        .. code-block:: python

            prompt_mgr.list(limit=5)    # list of 5 recent created prompt template assets

        .. hint::
            Additionally you can sort available prompt templates by "LAST MODIFIED" field.

            .. code-block:: python

                df_prompts = prompt_mgr.list()
                df_prompts.sort_values("LAST MODIFIED", ascending=False)

        """
        details = ['metadata.asset_id', 'metadata.name', 'metadata.created_at', 'metadata.usage.last_updated_at']
        prompts_details = self._get_details(limit=limit)

        data_normalize = pandas.json_normalize(prompts_details)
        prompts_data = data_normalize.reindex(columns=details)

        df_details = pandas.DataFrame(prompts_data, columns=details)

        df_details.rename(columns={'metadata.asset_id': 'ID',
                                   'metadata.name': 'NAME',
                                   'metadata.created_at': 'CREATED',
                                   'metadata.usage.last_updated_at': 'LAST MODIFIED',
                                   }, inplace=True)

        return df_details

    def store_prompt(self, prompt_template: Union[PromptTemplate, langchain.prompts.PromptTemplate]) -> PromptTemplate:
        """Store a new prompt template.

        :param prompt_template: PromptTemplate to be stored.
        :type prompt_template: (PromptTemplate | langchain.prompts.PromptTemplate)

        :return: PromptTemplate object initialized with values provided in the server response object.
        :rtype: PromptTemplate
        """
        if isinstance(prompt_template, PromptTemplate):
            pass
        else:
            from langchain.prompts import PromptTemplate as LcPromptTemplate
            if isinstance(prompt_template, LcPromptTemplate):
                prompt_template = PromptTemplate(name="My prompt",
                                                 model_id=ModelTypes.FLAN_UL2,
                                                 input_text=prompt_template.template,
                                                 input_variables=prompt_template.input_variables)
            else:
                raise WMLClientError(error_msg="Unsupported type for `prompt_template`")
            
        headers = self._client._get_headers()

        PromptTemplateManager._validate_type(prompt_template.name,
                                             u'prompt_template.name',
                                             str, True)
        json_data: Dict = {
            "name": prompt_template.name,
            "lock": {"locked": True},
            "prompt": dict()
        }

        json_data.update(self._create_request_body(prompt_template))

        url = self._client.service_instance._href_definitions.get_prompts_href()
        response = requests.post(url=url,
                                 json=json_data,
                                 headers=headers,
                                 params=self.params)
        response = self._handle_response(201, u'store_prompt', response)

        return self._from_json_to_prompt(response)

    def delete_prompt(self, prompt_id: str, *, force: bool = False) -> str:
        """Remove prompt template from project or space.

        :param prompt_id: Id of prompt template that will be delete. 
        :type prompt_id: str

        :param force: If True then prompt template is unlocked and then delete, defaults to False.
        :type force: bool

        :return: Status 'SUCESS' if the prompt template is successfully deleted.
        :rtype: str
        
        **Example**

        .. code-block:: python

            prompt_mgr.delete_prompt(prompt_id)  # delete if asset is unclocked
        """
        if force:
            self.unlock(prompt_id)

        headers = self._client._get_headers()
        params = (self.params | {"prompt_id": prompt_id})

        url = self._client.service_instance._href_definitions.get_prompts_href() + f"/{prompt_id}"
        response = requests.delete(url=url, headers=headers,
                                   params=params)

        return self._handle_response(204, u'delete_prompt', response)

    def update_prompt(self, prompt_id: str, prompt_template: PromptTemplate) -> Dict:
        """Update prompt template data.

        :param prompt_id: Id of the updated prompt template.
        :type prompt_id: str

        :param prompt: PromptTemplate with new data.
        :type prompt: PromptTemplate 

        :return: metadata of updated deployment
        :rtype: dict

        **Example**

        .. code-block:: python

            updataed_prompt_template = PromptTemplate(name="New name")
            prompt_mgr.update_prompt(prompt_id, prompt_template)  # {'name': 'New name'} in metadata  
            
        """
        headers = self._client._get_headers()
        params = (self.params | {"prompt_id": prompt_id})

        new_body: Dict = dict()
        current_prompt_template = self.load_prompt(prompt_id)

        for attribute in prompt_template.__dict__:
            if getattr(prompt_template, attribute) is not None and not attribute.startswith("_"):
                setattr(current_prompt_template, attribute, getattr(prompt_template, attribute))

        if current_prompt_template.name is not None:
            new_body.update({'name': current_prompt_template.name})

        new_body.update(self._create_request_body(current_prompt_template))

        url = self._client.service_instance._href_definitions.get_prompts_href() + f"/{prompt_id}"

        response = requests.patch(url=url,
                                  json=new_body,
                                  headers=headers,
                                  params=params)
        return self._handle_response(200, u'update_prompt', response)

    def get_lock(self, prompt_id: str) -> Dict:
        """Get the current locked state of a prompt template.

        :param prompt_id: Id of prompt template
        :type prompt_id: str

        :return: Information about locked state of prompt template asset.
        :rtype: Dict

        **Example**

        .. code-block:: python

            print(prompt_mgr.get_lock(prompt_id))
        """
        headers = self._client._get_headers()
        params = (self.params | {"prompt_id": prompt_id})
        url = self._client.service_instance._href_definitions.get_prompts_href() + f"/{prompt_id}/lock"

        response = requests.get(url=url,
                                headers=headers,
                                params=params)

        return self._handle_response(200, u'get_lock', response)

    def lock(self, prompt_id: str, force: bool = False) -> Dict:
        """Lock the prompt template if it is unlocked and user has permission to do that. 

        :param promp_id: Id of prompt template.
        :type promp_id: str

        :param force: If True, method forcefully overwrite a lock.
        :type force: bool

        :return: Status 'SUCCESS' or response content after an attempt to lock prompt template.
        :rtype: (str | Dict)

        **Example**

        .. code-block:: python

            prompt_mgr.lock(prompt_id)

        """
        return self._change_lock(prompt_id=prompt_id,
                                 locked=True,
                                 force=force)

    def unlock(self, prompt_id: str) -> Dict:
        """Unlock the prompt template if it is locked and user has permission to do that.

        :param promp_id: Id of prompt template.
        :type promp_id: str

        :return: Response content after an attempt to unlock prompt template.
        :rtype: Dict

        **Example**

        .. code-block:: python

            prompt_mgr.unlock(prompt_id)
        """
        # server returns status code 400 after trying to unlock unlocked prompt
        lock_state = self.get_lock(prompt_id)
        if lock_state['locked']:
            return self._change_lock(prompt_id=prompt_id,
                                     locked=False,
                                     force=False)
        else:
            return lock_state
