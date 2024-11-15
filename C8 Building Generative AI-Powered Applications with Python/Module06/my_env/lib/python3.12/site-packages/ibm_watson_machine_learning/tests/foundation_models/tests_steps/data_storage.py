#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from dataclasses import dataclass
from os import environ

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.tests.utils import get_cos_credentials, get_wml_credentials


@dataclass
class DataStorage:
    SPACE_ONLY = True  # False mean that tests will be working  on `default_project_id`

    bucket_name = environ.get('BUCKET_NAME', "prompt-tuning-sdk-tests")
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    data_location = './foundation_models/data/file_to_tune1.json'
    data_cos_path = 'file_to_tune1.json'
    results_cos_path = 'results_wml_prompt_tuning'

    experiment = None
    prompt_tuner = None
    tuned_details = None
    run_id = None
    asset_id = None

    train_data_connections: list = None
    results_data_connection = None

    stored_model_id = None
    promoted_model_id = None

    connection_id = None
    deployment_id = None

    project_models_to_delete = []
    space_models_to_delete = []

    # Function Representatives
    credentials = get_wml_credentials()
    project_id = None
    cos_credentials = get_cos_credentials()
    cos_endpoint = cos_credentials['endpoint_url']
    cos_resource_instance_id = cos_credentials['resource_instance_id']
    api_client = None
    space_id = None

    # Prompt Template
    prompt_id = None
    stored_prompt_id = None
    base_model_id = ModelTypes.FLAN_T5_XL.value
    prompt_mgr = None

