#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import time
import json
import warnings

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ibm_watson_machine_learning.foundation_models.utils.utils import HAPDetectionWarning, PIIDetectionWarning
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError

__all__ = [
    "BaseModelInference"
]


class BaseModelInference(WMLResource, ABC):
    """Base interface class for the model interface."""

    def __init__(self, name, client):
        WMLResource.__init__(self, name, client)

    @abstractmethod
    def get_details(self):
        """Get model interface's details

                :return: details of model or deployment
                :rtype: dict
        """
        pass

    @abstractmethod
    def generate(self, prompt, params: dict, concurrency_limit: int, async_mode: bool) -> Union[dict, List[dict]]:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        pass

    @abstractmethod
    def generate_text_stream(self, prompt, params=None):
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        pass

    @abstractmethod
    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        pass

    def _prepare_inference_payload(self,
                                   prompt: str,
                                   params: Optional[dict] = None,
                                   guardrails: bool = False,
                                   guardrails_hap_params: Optional[dict] = None,
                                   guardrails_pii_params: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def _prepare_beta_inference_payload(self,
                                        prompt: str,
                                        params: Optional[dict] = None,
                                        guardrails: bool = False,
                                        guardrails_hap_params: Optional[dict] = None,
                                        guardrails_pii_params: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def _send_inference_payload(self,
                                prompt: str,
                                params: dict,
                                generate_url: str,
                                guardrails: bool = False,
                                guardrails_hap_params: Optional[dict] = None,
                                guardrails_pii_params: Optional[dict] = None
                                ):
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(prompt,
                                                      params=params,
                                                      guardrails=guardrails,
                                                      guardrails_hap_params=guardrails_hap_params,
                                                      guardrails_pii_params=guardrails_pii_params)
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(prompt,
                                                           params=params,
                                                           guardrails=guardrails,
                                                           guardrails_hap_params=guardrails_hap_params,
                                                           guardrails_pii_params=guardrails_pii_params)

        retries = 0
        while retries < 3:
            response_scoring = requests.post(
                url=generate_url,
                json=payload,
                params=self._client._params(skip_for_create=True),
                headers=self._client._get_headers()
            )
            if response_scoring.status_code in [429, 503, 504, 520]:
                time.sleep(2 ** retries)
                retries += 1
            else:
                break

        return self._handle_response(200, u'generate', response_scoring)

    def _generate_with_url(self,
                           prompt: list | str,
                           params: dict,
                           generate_url: str,
                           guardrails: bool = False,
                           guardrails_hap_params: Optional[dict] = None,
                           guardrails_pii_params: Optional[dict] = None,
                           concurrency_limit: int = 10):
        """
        Helper method which implements multi-threading for with passed generate_url.
        """

        if isinstance(prompt, list):
            generated_responses = []
            if len(prompt) <= concurrency_limit:
                with ThreadPoolExecutor() as executor:
                    response_batch = list(
                        executor.map(self._generate_with_url, prompt, [params] * len(prompt),
                                     [generate_url] * len(prompt), [guardrails] * len(prompt),
                                     [guardrails_hap_params] * len(prompt), [guardrails_pii_params] * len(prompt))
                    )
                generated_responses.extend(response_batch)
            else:
                for i in range(0, len(prompt), concurrency_limit):
                    prompt_batch = prompt[i:i + concurrency_limit]
                    with ThreadPoolExecutor() as executor:
                        response_batch = list(
                            executor.map(self._generate_with_url, prompt_batch,
                                         [params] * len(prompt_batch),
                                         [generate_url] * len(prompt_batch), [guardrails] * len(prompt),
                                         [guardrails_hap_params] * len(prompt), [guardrails_pii_params] * len(prompt))
                        )
                    generated_responses.extend(response_batch)
            return generated_responses

        else:
            return self._send_inference_payload(prompt,
                                                params,
                                                generate_url,
                                                guardrails,
                                                guardrails_hap_params,
                                                guardrails_pii_params)

    def _generate_with_url_async(self,
                                 prompt: list | str,
                                 params: dict,
                                 generate_url: str,
                                 guardrails: bool = False,
                                 guardrails_hap_params: Optional[dict] = None,
                                 guardrails_pii_params: Optional[dict] = None,
                                 concurrency_limit: int = 10):
        async_params = params or {}
        async_params['return_options'] = {"input_text": True}

        if isinstance(prompt, list):
            for i in range(0, len(prompt), concurrency_limit):
                prompt_batch = prompt[i:i + concurrency_limit]
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._send_inference_payload,
                                               single_prompt,
                                               async_params,
                                               generate_url,
                                               guardrails,
                                               guardrails_hap_params,
                                               guardrails_pii_params) for single_prompt in prompt_batch]
                    for future in as_completed(futures):
                        yield future.result()
        else:
            yield self._send_inference_payload(prompt,
                                               async_params,
                                               generate_url,
                                               guardrails,
                                               guardrails_hap_params,
                                               guardrails_pii_params)

    def _generate_stream_with_url(self,
                                  prompt: str,
                                  params: dict,
                                  generate_stream_url: str,
                                  raw_response: bool = False,
                                  guardrails: bool = False,
                                  guardrails_hap_params: Optional[dict] = None,
                                  guardrails_pii_params: Optional[dict] = None):

        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(prompt,
                                                      params=params,
                                                      guardrails=guardrails,
                                                      guardrails_hap_params=guardrails_hap_params,
                                                      guardrails_pii_params=guardrails_pii_params)
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(prompt,
                                                           params=params,
                                                           guardrails=guardrails,
                                                           guardrails_hap_params=guardrails_hap_params,
                                                           guardrails_pii_params=guardrails_pii_params)

        s = requests.Session()
        retries = 0
        while retries < 3:
            with s.post(url=generate_stream_url,
                        json=payload,
                        headers=self._client._get_headers(),
                        params=self._client._params(skip_for_create=True),
                        stream=True) as resp:
                  
                if resp.status_code in [429, 503, 504, 520]:
                    time.sleep(2 ** retries)
                    retries += 1
                elif resp.status_code == 200:
                    for chunk in resp.iter_lines(decode_unicode=False):
                        chunk = chunk.decode('utf-8')
                        if 'generated_text' in chunk:
                            response = chunk.replace('data: ', '')
                            try:
                                parsed_response = json.loads(response)
                            except json.JSONDecodeError:
                                raise Exception(f"Could not parse {response} as json")
                            if raw_response:
                                yield parsed_response
                                continue
                            yield self._return_guardrails_stats(parsed_response)['generated_text']
                    break

        if resp.status_code != 200:
            raise WMLClientError(f'Request failed with: {resp.text} ({resp.status_code})')

    def _tokenize_with_url(self,
                           prompt: str,
                           tokenize_url: str,
                           return_tokens: bool,
                           ):

        payload = self._prepare_inference_payload(prompt)

        parameters = payload.get('parameters', {})
        parameters.update({"return_tokens": return_tokens})
        payload['parameters'] = parameters

        retries = 0
        while retries < 3:
            response_scoring = requests.post(
                url=tokenize_url,
                json=payload,
                params=self._client._params(skip_for_create=True),
                headers=self._client._get_headers()
            )
            if response_scoring.status_code in [429, 503, 504, 520]:
                time.sleep(2 ** retries)
                retries += 1
            elif response_scoring.status_code == 404:
                raise WMLClientError("Tokenize is not supported for this release")
            else:
                break

        return self._handle_response(200, u'generate', response_scoring)

    def _return_guardrails_stats(self, single_response):
        results = single_response['results'][0]
        hap_details = results.get("moderations", {}).get("hap") if self._client._use_fm_ga_api else results.get("moderation", {}).get("hap")  # Remove 'else' on CPD 5.0 release
        if hap_details:
            if hap_details[0].get("input"):
                warnings.warn(next(warning.get('message') for warning in single_response.get('system', {}).get('warnings')
                                if warning.get('id') == 'UNSUITABLE_INPUT'), category=HAPDetectionWarning)
            else:
                warnings.warn(f'Potentially harmful text detected: {hap_details}', category=HAPDetectionWarning)
        pii_details = results.get("moderations", {}).get("pii") if self._client._use_fm_ga_api else results.get("moderation", {}).get("pii")  # Remove 'else' on CPD 5.0 release
        if pii_details:
            if pii_details[0].get("input"):
                warnings.warn(next(warning.get('message') for warning in single_response.get('system', {}).get('warnings')
                                if warning.get('id') == 'UNSUITABLE_INPUT'), category=PIIDetectionWarning)
            else:
                warnings.warn(f'Personally identifiable information detected: {pii_details}', category=PIIDetectionWarning)
        return results

    @staticmethod
    def _update_moderations_params(additional_params: dict) ->dict:

        default_params = {
            'input': {
                'enabled': True
            },
            'output': {
                'enabled': True
            }
        }

        if additional_params:
            for key, value in default_params.items():
                if key in additional_params:
                    if additional_params[key]:
                        if "threshold" in additional_params:
                            default_params[key]["threshold"] = additional_params["threshold"]
                    else:
                        default_params[key]["enabled"] = False
                else:
                    if "threshold" in additional_params:
                        default_params[key]["threshold"] = additional_params["threshold"]
            if "mask" in additional_params:
                default_params.update({"mask": additional_params["mask"]})
        return default_params
