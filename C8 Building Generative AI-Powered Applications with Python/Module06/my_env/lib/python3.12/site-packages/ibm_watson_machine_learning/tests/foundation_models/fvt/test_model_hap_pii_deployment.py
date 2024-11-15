#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import warnings

import pytest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager, PromptTemplate
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods, ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams, GenTextModerationsMetaNames
from ibm_watson_machine_learning.tests.conftest import load_updated_credentials, create_project, load_api_client, \
    load_cos_credentials, load_original_credentials, delete_project
from ibm_watson_machine_learning.tests.utils import is_cp4d

# model_types_list = [model.value for model in ModelTypes]

# Prompt Template cannot be deployed with constricted model available_models = [model_spec['model_id'] for model_spec
# in get_model_specs(wml_credentials.get('url')).get('resources', []) if model_spec['model_id'] in model_types_list
# and not ('constricted' in [el.get('id') for el in model_spec.get('lifecycle', [])])]

# For automatic tests we select only one model
available_models = ['google/flan-ul2']


@pytest.mark.skipif(is_cp4d(), reason="Prompt Template deployment is not supported on CPD 4.8")
class TestModelHapPii:
    """
    This tests covers:
    - Generate text with HAP on
    - generate text with PII on
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.project_id = create_project(load_api_client(load_original_credentials()), load_cos_credentials(), load_original_credentials())
        cls.wml_credentials = load_updated_credentials(cls.project_id)
        cls.client = APIClient(wml_credentials=cls.wml_credentials,project_id=cls.project_id)
        cls.client.set.default_project(cls.project_id)
        cls.text_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 30,
            GenParams.MAX_NEW_TOKENS: 50
        }
        cls.prompt_mgr = PromptTemplateManager(cls.wml_credentials.copy(),project_id=cls.project_id)
        cls.stored_prompt = cls.prompt_mgr.store_prompt(PromptTemplate(name="My test prompt",
                                                                       model_id=ModelTypes.FLAN_UL2,
                                                                       input_text="[ {object} ] ",
                                                                       instruction="Please repeat the words in [], do not trim space.",
                                                                       input_variables=["object"]))
        cls.deployment_id = None

    @classmethod
    def teardown_class(cls):
        TestModelHapPii.prompt_mgr.delete_prompt(TestModelHapPii.stored_prompt.prompt_id, force=True)
        delete_project(cls.project_id, cls.client)

    def teardown_method(self):
        TestModelHapPii.client.deployments.delete(TestModelHapPii.deployment_id)

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_00_generate_hap_output(self, model):
        prompt_variables = {"object": "I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.THRESHOLD: 0.01
        }
        # guardrails_pii_params = {
        #     GenTextModerationsMetaNames.INPUT: False,
        #     GenTextModerationsMetaNames.OUTPUT: False
        #     }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            # guardrails_pii_params=guardrails_pii_params
        )

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('hap'), "No HAP detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_01_generate_pii_output(self, model):
        prompt_variables = {"object": 'foo.bar@ibm.com'}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.OUTPUT: False

        }
        guardrails_pii_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.OUTPUT: True,
            GenTextModerationsMetaNames.THRESHOLD: 0.01
        }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params)

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('pii'), "No PII detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_02_generate_hap_pii_output(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.THRESHOLD: 0.00

        }
        guardrails_pii_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.OUTPUT: True,
            GenTextModerationsMetaNames.THRESHOLD: 0.00
        }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params)

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('pii') or \
               response['results'][0].get('moderations', {}).get('hap'), "No HAP/PII detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_03_generate_hap_input(self, model):
        prompt_variables = {"object": "I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.THRESHOLD: 0.01
        }
        # guardrails_pii_params = {
        #     GenTextModerationsMetaNames.INPUT: False,
        #     GenTextModerationsMetaNames.OUTPUT: False
        #     }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            # guardrails_pii_params=guardrails_pii_params
        )

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('hap'), "No HAP detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_04_generate_pii_input(self, model):
        prompt_variables = {"object": 'foo.bar@ibm.com'}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.OUTPUT: False

        }
        guardrails_pii_params = {
            GenTextModerationsMetaNames.INPUT: True,
            GenTextModerationsMetaNames.OUTPUT: True,
            GenTextModerationsMetaNames.THRESHOLD: 0.01
        }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params)

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('pii'), "No PII detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_05_generate_hap_pii_input(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.THRESHOLD: 0.00

        }
        guardrails_pii_params = {
            GenTextModerationsMetaNames.INPUT: True,
            GenTextModerationsMetaNames.OUTPUT: True,
            GenTextModerationsMetaNames.THRESHOLD: 0.00
        }

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params)

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('pii') or \
               response['results'][0].get('moderations', {}).get('hap'), "No HAP/PII detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_06_generate_stream_hap_pii_output(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id
        guardrails_hap_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.THRESHOLD: 0.00

        }
        guardrails_pii_params = {
            GenTextModerationsMetaNames.INPUT: False,
            GenTextModerationsMetaNames.OUTPUT: True,
            GenTextModerationsMetaNames.THRESHOLD: 0.00
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            text = TestModelHapPii.client.deployments.generate_text_stream(
                deployment_id=deployment_id,
                params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
                guardrails=True,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params
            )

            text_stream = list(text)
            print(text_stream)
            assert len(['Potentially harmful text detected' in str(warning.message) for warning in w]) or \
                   len(['Personally identifiable information detected' in str(warning.message) for warning in w]), \
                "No worning with HAP/PII detection"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_07_generate_guardrails_enabled(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=True)

        print(response['results'][0].get('moderations'))
        assert response['results'][0].get('moderations', {}).get('pii') or \
               response['results'][0].get('moderations', {}).get('hap'), "No HAP/PII detected"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_08_generate_guardrails_enabled(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            text = TestModelHapPii.client.deployments.generate_text_stream(
                deployment_id=deployment_id,
                params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
                guardrails=True
            )

            text_stream = list(text)
            print(text_stream)
            assert len(['Potentially harmful text detected' in str(warning.message) for warning in w]) or \
                   len(['Personally identifiable information detected' in str(warning.message) for warning in w]), \
                "No warning with HAP/PII detection"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_09_generate_guardrails_disabled(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id

        response = TestModelHapPii.client.deployments.generate(
            deployment_id=deployment_id,
            params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
            guardrails=False)

        print(response['results'][0].get('moderations'))
        assert 'pii' not in response['results'][0].get('moderations', {}), "PII detected, detection should be disabled"
        assert 'hap' not in response['results'][0].get('moderations', {}), "HAP detected, detection should be disabled"

    @pytest.mark.parametrize('model', available_models, ids=available_models)
    def test_10_generate_stream_guardrails_disabled(self, model):
        prompt_variables = {"object": "foo.bar@ibm.com  I hate this damn world."}
        meta_props = {
            TestModelHapPii.client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            TestModelHapPii.client.deployments.ConfigurationMetaNames.ONLINE: {},
            TestModelHapPii.client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: model}
        deployment_details = TestModelHapPii.client.deployments.create(
            TestModelHapPii.stored_prompt.prompt_id,
            meta_props=meta_props
        )
        deployment_id = TestModelHapPii.client.deployments.get_uid(deployment_details)
        TestModelHapPii.deployment_id = deployment_id

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            text = TestModelHapPii.client.deployments.generate_text_stream(
                deployment_id=deployment_id,
                params=(TestModelHapPii.text_params | {'prompt_variables': prompt_variables}),
                guardrails=False
            )

            text_stream = list(text)
            print(text_stream)
            assert len(['Potentially harmful text detected' in str(warning.message) for warning in w]) == 0 or \
                   len(['Personally identifiable information detected' in str(warning.message) for warning in
                        w]) == 0, "Warning with HAP/PII detection raised"
