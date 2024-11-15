#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import warnings
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Generator
from ibm_watson_machine_learning.wml_resource import WMLResource

__all__ = [
    "FMModelInference"
]

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ParamOutOfRange, InvalidMultipleArguments
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.utils.utils import get_model_specs, _check_model_state
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
from ibm_watson_machine_learning.messages.messages import Messages
from .base_model_inference import BaseModelInference


class FMModelInference(BaseModelInference):
    """Base abstract class for the model interface."""

    def __init__(self,
                 *,
                 model_id: str = None,
                 params: dict = None,
                 api_client: APIClient = None) -> None:

        self.model_id = model_id
        if isinstance(self.model_id, ModelTypes):
            self.model_id = self.model_id.value

        self.params = params
        FMModelInference._validate_type(params, u'params', dict, False)

        self._client = api_client

        supported_models = [model_spec['model_id'] for model_spec in get_model_specs(self._client.wml_credentials.get('url')).get('resources', [])]
        if self.model_id not in supported_models:
            raise WMLClientError(error_msg=f"Model '{self.model_id}' is not supported for this environment. "
                                           f"Supported models: {supported_models}")
        
        # check if model is in constricted mode
        _check_model_state(self._client.wml_credentials.get('url'), self.model_id)
        
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        BaseModelInference.__init__(self, __name__, self._client)

    def get_details(self):
        """Get model's details

        :return: details of model or deployment
        :rtype: dict
        """
        models = get_model_specs(self._client.wml_credentials['url']).get('resources', [])
        return next((item for item in models if item['model_id'] == self.model_id), None)

    def generate(self,
                 prompt,
                 params: dict = None,
                 guardrails: bool = False,
                 guardrails_hap_params: Optional[dict] = None,
                 guardrails_pii_params: Optional[dict] = None,
                 concurrency_limit: int = 10,
                 async_mode: bool = False) -> Union[dict, List[dict], Generator[dict, str, None]]:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        self._validate_type(prompt, u'prompt', [str, list], True, raise_error_for_list=True)
        self._validate_type(guardrails_hap_params, u'guardrails_hap_params', dict, mandatory=False)
        self._validate_type(guardrails_pii_params, u'guardrails_pii_params', dict, mandatory=False)

        generate_text_url = self._client.service_instance._href_definitions.get_fm_generation_href('text')

        if async_mode:
            return self._generate_with_url_async(prompt=prompt,
                                                 params=params or self.params,
                                                 generate_url=generate_text_url,
                                                 guardrails=guardrails,
                                                 guardrails_hap_params=guardrails_hap_params,
                                                 guardrails_pii_params=guardrails_pii_params,
                                                 concurrency_limit=concurrency_limit)
        else:
            return self._generate_with_url(prompt=prompt,
                                           params=params,
                                           generate_url=generate_text_url,
                                           guardrails=guardrails,
                                           guardrails_hap_params=guardrails_hap_params,
                                           guardrails_pii_params=guardrails_pii_params,
                                           concurrency_limit=concurrency_limit)

    def generate_text_stream(self,
                             prompt,
                             params=None,
                             raw_response=False,
                             guardrails: bool = False,
                             guardrails_hap_params: Optional[dict] = None,
                             guardrails_pii_params: Optional[dict] = None):
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        self._validate_type(prompt, u'prompt', str, True)

        if self._client._use_fm_ga_api:
            generate_text_stream_url = self._client.service_instance._href_definitions.get_fm_generation_stream_href()
        else:
            generate_text_stream_url = self._client.service_instance._href_definitions.get_fm_generation_href(f'text_stream')  # Remove on CPD 5.0 release

        return self._generate_stream_with_url(prompt=prompt,
                                              params=params,
                                              raw_response=raw_response,
                                              generate_stream_url=generate_text_stream_url,
                                              guardrails=guardrails,
                                              guardrails_hap_params=guardrails_hap_params,
                                              guardrails_pii_params=guardrails_pii_params)

    def tokenize(self,
                 prompt,
                 return_tokens: bool = False):
        """
        Given a text prompt as input, and return_tokens parameter will return tokenized input text.
        """
        self._validate_type(prompt, u'prompt', str, True)
        generate_tokenize_url = self._client.service_instance._href_definitions.get_fm_tokenize_href()

        return self._tokenize_with_url(prompt=prompt,
                                       tokenize_url=generate_tokenize_url,
                                       return_tokens=return_tokens)

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return {
            "model_id": self.model_id,
            "params": self.params,
            "project_id": self._client.default_project_id,
            "space_id": self._client.default_space_id
        }

    def _prepare_inference_payload(self,
                                   prompt: str,
                                   params: dict = None,
                                   guardrails: bool = False,
                                   guardrails_hap_params: Optional[dict] = None,
                                   guardrails_pii_params: Optional[dict] = None) -> dict:
        payload = {
            "model_id": self.model_id,
            "input": prompt,
        }
        if guardrails:
            if guardrails_hap_params is None:
                guardrails_hap_params = dict(input=True, output=True)  # HAP enabled if guardrails = True

            for guardrail_type, guardrails_params in zip(('hap', 'pii'),
                                                         (guardrails_hap_params, guardrails_pii_params)):
                if guardrails_params is not None:
                    if "moderations" not in payload:
                        payload["moderations"] = {}
                    payload["moderations"].update({guardrail_type: self._update_moderations_params(guardrails_params)})

        if params:
            payload['parameters'] = params
        elif self.params:
            payload['parameters'] = self.params

        if 'parameters' in payload and GenTextParamsMetaNames.DECODING_METHOD in payload[
            'parameters']:
            if isinstance(payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD], DecodingMethods):
                payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD] = \
                    payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD].value

        if self._client.default_project_id:
            payload['project_id'] = self._client.default_project_id
        elif self._client.default_space_id:
            payload['space_id'] = self._client.default_space_id

        if 'parameters' in payload and 'return_options' in payload['parameters']:
            if not (payload['parameters']['return_options'].get("input_text", False) or
                    payload['parameters']['return_options'].get("input_tokens", False)):
                raise WMLClientError(Messages.get_message(message_id="fm_required_parameters_not_provided"))

        return payload

    def _prepare_beta_inference_payload(self,  # Remove on CPD 5.0 release
                                        prompt: str,
                                        params: dict = None,
                                        guardrails: bool = False,
                                        guardrails_hap_params: Optional[dict] = None,
                                        guardrails_pii_params: Optional[dict] = None) -> dict:
        payload = {
            "model_id": self.model_id,
            "input": prompt,
        }
        if guardrails:
            default_moderations_params = {
                'input': True,
                'output': True
            }
            payload.update({
                "moderations": {
                    "hap": (default_moderations_params | (guardrails_hap_params or {}))
                }
            })

            if guardrails_pii_params is not None:
                payload['moderations'].update({"pii": guardrails_pii_params})

        if params:
            payload['parameters'] = params
        elif self.params:
            payload['parameters'] = self.params

        if 'parameters' in payload and GenTextParamsMetaNames.DECODING_METHOD in payload['parameters']:
            if isinstance(payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD], DecodingMethods):
                payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD] = \
                    payload['parameters'][GenTextParamsMetaNames.DECODING_METHOD].value

        if self._client.default_project_id:
            payload['project_id'] = self._client.default_project_id
        elif self._client.default_space_id:
            payload['space_id'] = self._client.default_space_id

        if 'parameters' in payload and 'return_options' in payload['parameters']:
            if not (payload['parameters']['return_options'].get("input_text", False) or
                    payload['parameters']['return_options'].get("input_tokens", False)):
                raise WMLClientError(Messages.get_message(message_id="fm_required_parameters_not_provided"))

        return payload


