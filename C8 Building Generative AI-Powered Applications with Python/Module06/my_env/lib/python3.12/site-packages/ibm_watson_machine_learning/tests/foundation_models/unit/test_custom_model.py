#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest
import json
import logging

from ibm_watson_machine_learning.foundation_models.utils.utils import get_custom_model_specs
from ibm_watson_machine_learning.utils import NextResourceGenerator
from ibm_watson_machine_learning.foundation_models import ModelInference
from ibm_watson_machine_learning.wml_client_error import WMLClientError, MissingMetaProp, InvalidValue


@pytest.mark.unittest
class TestCustomModel:
    """
    These tests cover:
    - utils method get_custom_model_spec
    - model store
    - hardware specification store
    - model deployment
    - model inference generate
    """
    response_path = './foundation_models/artifacts/byom_api_responses.json'
    model_name = 'TheBloke/Mixtral-8x7B-v0.1-GPTQ'
    model_asset_id = '33527417-5d33-4ecd-9acf-dbb604586a4f'
    deployment_id = '1e09f0d0-6656-4846-9877-04c90a9d25f5'
    hw_spec_name = 'HW SPEC from sdk test'

    @staticmethod
    def mock_data_from_request(mock, data_part='', status_code=200):
        with open(TestCustomModel.response_path) as json_file:
            data = json.load(json_file).get(data_part, {})

        mock.status_code = status_code
        mock.json.return_value = data

    def test_get_all_custom_models_spec(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        model_specs = get_custom_model_specs(api_client=api_client_mock)

        assert isinstance(model_specs, dict), \
            f"Model specifications are not of the `dict` type, actual: {type(model_specs)}"
        logging.info(json.dumps(model_specs, indent=4))

        assert isinstance(model_specs, dict), "get_custom_model_specs() did not return dict"
        assert 'resources' in model_specs, "get_custom_model_specs() did not return `resources` field"

    def test_get_specific_custom_models_spec(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        model_specs = get_custom_model_specs(api_client=api_client_mock, model_id=self.model_name)
        logging.info(json.dumps(model_specs, indent=4))

        assert isinstance(model_specs, dict), \
            f"Model specifications are not of the `dict` type, actual: {type(model_specs)}"
        assert model_specs.get('model_id') == self.model_name, \
            "get_custom_model_specs() did not return proper `model_id`"

    def test_get_invalid_custom_models_spec(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        model_specs = get_custom_model_specs(api_client=api_client_mock, model_id="non_existing_model_name")
        logging.info(json.dumps(model_specs, indent=4))

        assert isinstance(model_specs, dict), \
            f"Model specifications are not of the `dict` type, actual: {type(model_specs)}"
        assert model_specs == {}, "get_custom_model_specs() returned not empty dict"

    def test_get_custom_models_spec_limit_exceeded(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        with pytest.raises(InvalidValue):
            get_custom_model_specs(api_client=api_client_mock, limit=201)

    def test_get_custom_models_spec_limit_below_one(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        with pytest.raises(InvalidValue):
            get_custom_model_specs(api_client=api_client_mock, limit=0)

    def test_get_custom_models_spec_async(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        model_specs = get_custom_model_specs(api_client=api_client_mock, asynchronous=True)

        assert isinstance(model_specs, NextResourceGenerator), \
            "get_custom_model_specs() did not return NextResourceGenerator"

    def test_get_custom_models_spec_get_all(self, api_client_mock, mock_get):
        self.mock_data_from_request(mock_get, data_part='custom_models_spec')

        model_specs = get_custom_model_specs(api_client=api_client_mock, get_all=True)

        assert isinstance(model_specs, dict), "get_custom_model_specs() did not return dict"

    def test_store_model_valid(self, api_client_mock, mock_post, mock_get):
        self.mock_data_from_request(mock_post, data_part='stored_custom_model_details', status_code=201)
        self.mock_data_from_request(mock_get, data_part='stored_custom_model_details')
        api_client_mock.CLOUD_PLATFORM_SPACES = False
        api_client_mock.CPD_version = 4.8

        metadata = {
            api_client_mock.repository.ModelMetaNames.NAME: 'custom fm test',
            api_client_mock.repository.ModelMetaNames.SOFTWARE_SPEC_UID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee",
            api_client_mock.repository.ModelMetaNames.TYPE: 'custom_foundation_model_1.0'
        }

        stored_details = api_client_mock.repository.store_model(self.model_name, meta_props=metadata)
        logging.info(json.dumps(stored_details, indent=4))

        assert stored_details is not None, "`client.repository.store_model()` should not return empty dict"
        assert stored_details['metadata']['id'] == self.model_asset_id, \
            f"Model asset identifier {stored_details['metadata']['id']} is different than expected"
        assert stored_details['metadata']['name'] == 'custom fm test', \
            f"store_model() returned invalid name {stored_details['metadata']['name']}"

    def test_store_model_cloud(self, api_client_mock, mock_post, mock_get):
        self.mock_data_from_request(mock_post, data_part='stored_custom_model_details', status_code=201)
        self.mock_data_from_request(mock_get, data_part='stored_custom_model_details')
        api_client_mock.CLOUD_PLATFORM_SPACES = True

        metadata = {
            api_client_mock.repository.ModelMetaNames.NAME: 'custom fm test',
            api_client_mock.repository.ModelMetaNames.SOFTWARE_SPEC_UID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeee",
            api_client_mock.repository.ModelMetaNames.TYPE: 'custom_foundation_model_1.0'
        }

        with pytest.raises(WMLClientError):
            api_client_mock.repository.store_model(self.model_name, meta_props=metadata)

    def test_store_model_empty_metadata(self, api_client_mock):
        with pytest.raises(WMLClientError):
            api_client_mock.repository.store_model(self.model_name, meta_props={})

    def test_store_model_invalid_cpd_version(self, api_client_mock):
        metadata = {api_client_mock.repository.ModelMetaNames.TYPE: 'custom_foundation_model_1.0'}
        version = api_client_mock.CPD_version
        api_client_mock.CPD_version = 4.7

        with pytest.raises(WMLClientError):
            api_client_mock.repository.store_model(self.model_name, meta_props=metadata)

        api_client_mock.CPD_version = version

    def test_store_model_invalid_metadata(self, api_client_mock):
        metadata = {api_client_mock.repository.ModelMetaNames.TYPE: 'custom_foundation_model_1.0'}
        expected_exception = WMLClientError if api_client_mock.CLOUD_PLATFORM_SPACES else MissingMetaProp

        with pytest.raises(expected_exception):
            api_client_mock.repository.store_model(self.model_name, meta_props=metadata)

    def test_store_hardware_spec_valid(self, api_client_mock, mock_post):
        self.mock_data_from_request(mock_post, data_part='store_hardware_spec_details', status_code=201)
        metadata = {
            api_client_mock.hardware_specifications.ConfigurationMetaNames.NAME: "HW SPEC from sdk",
            api_client_mock.hardware_specifications.ConfigurationMetaNames.NODES: {"cpu": {"units": "2"},
                                                                                   "mem": {"size": "128Gi"},
                                                                                   "gpu": {"num_gpu": 1}}
        }

        hw_spec_details = api_client_mock.hardware_specifications.store(metadata)
        logging.info(json.dumps(hw_spec_details, indent=4))

        assert hw_spec_details is not None, "`hardware_specifications.store()` should not return empty dict"
        assert hw_spec_details['metadata']['name'] == self.hw_spec_name, \
            f"Hardware specification name `{hw_spec_details['metadata']['name']}` is different than expected"

    def test_store_hardware_spec_invalid_metadata(self, api_client_mock):
        metadata = {
            api_client_mock.hardware_specifications.ConfigurationMetaNames.NODES:
                {
                    "cpu": {"units": "2"},
                    "mem": {"size": "128Gi"},
                    "gpu": {"num_gpu": 1}
                }
        }

        with pytest.raises(MissingMetaProp):
            api_client_mock.hardware_specifications.store(metadata)

    def test_deploy_model_valid(self, api_client_mock, mock_post, mock_get):
        self.mock_data_from_request(mock_post, data_part='deployment_custom_model_details', status_code=202)
        self.mock_data_from_request(mock_get, data_part='deployment_custom_model_details')
        api_client_mock.CLOUD_PLATFORM_SPACES = True

        metadata = {
            api_client_mock.deployments.ConfigurationMetaNames.NAME: "Custom FM Deployment",
            api_client_mock.deployments.ConfigurationMetaNames.DESCRIPTION: "Deployment of custom foundation model with SDK",
            api_client_mock.deployments.ConfigurationMetaNames.ONLINE: {},
            api_client_mock.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name": self.hw_spec_name},
            api_client_mock.deployments.ConfigurationMetaNames.FOUNDATION_MODEL: {"max_new_tokens": 4}
        }

        deployment_details = api_client_mock.deployments.create(self.model_asset_id, metadata)
        logging.info(json.dumps(deployment_details, indent=4))

        assert deployment_details is not None, "`client.deployments.create()` should not return empty dict"
        assert deployment_details['metadata']['id'] == self.deployment_id, \
            f"Deployment identifier `{deployment_details['metadata']['id']}` is different than expected"

    def test_deploy_model_empty_metadata(self, api_client_mock):
        with pytest.raises(WMLClientError):
            api_client_mock.deployments.create(self.model_asset_id)

    def test_deploy_model_invalid_metadata(self, api_client_mock, mock_post):
        self.mock_data_from_request(mock_post, data_part='deployment_custom_model_details', status_code=404)
        mock_post.text = "Failure deployment reason"

        with pytest.raises(WMLClientError):
            api_client_mock.deployments.create(self.model_asset_id, meta_props={})

    def test_model_inference_basic(self, api_client_mock, mock_post, mock_get):
        self.mock_data_from_request(mock_get, data_part='deployment_custom_model_details')
        self.mock_data_from_request(mock_post, data_part='generated_text_model_inference', status_code=200)

        deployment_inference = ModelInference(deployment_id=self.deployment_id, api_client=api_client_mock)

        response = deployment_inference.generate_text("What is 2 + 2?",
                                                      params={'max_new_tokens': 4})
        logging.info(json.dumps(response, indent=4))

        assert isinstance(response, str), "Generated text is not `str`"
        assert len(response) > 0, "Generated text is empty"

    def test_model_inference_raw_response(self, api_client_mock, mock_post, mock_get):
        self.mock_data_from_request(mock_get, data_part='deployment_custom_model_details')
        self.mock_data_from_request(mock_post, data_part='generated_text_model_inference', status_code=200)

        deployment_inference = ModelInference(deployment_id=self.deployment_id, api_client=api_client_mock)

        response = deployment_inference.generate_text("What is 2 + 2?",
                                                      params={'max_new_tokens': 4}, raw_response=True)
        logging.info(json.dumps(response, indent=4))

        assert isinstance(response, dict), "Generated response is not `dict`"
        assert 'results' in response, "Generated response does not contain results field"
