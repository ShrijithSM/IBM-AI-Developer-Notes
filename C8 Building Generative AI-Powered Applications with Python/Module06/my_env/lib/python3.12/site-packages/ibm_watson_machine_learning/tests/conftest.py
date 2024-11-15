#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from os import environ

import allure
import pytest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning._wrappers import requests
from ibm_watson_machine_learning.tests.foundation_models.tests_steps.data_storage import DataStorage
from ibm_watson_machine_learning.tests.foundation_models.tests_steps.prompt_template_steps import PromptTemplateSteps
from ibm_watson_machine_learning.tests.foundation_models.tests_steps.prompt_tuning_steps import PromptTuningSteps
from ibm_watson_machine_learning.tests.foundation_models.tests_steps.universal_steps import UniversalSteps
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure


def pytest_collection_modifyitems(items):
    """
    Because UnitTest do not like to cooperate with fixtures other than with param `autouse=False`
    there is a need to enumerate test BY MODEL and then ALPHANUMERICAL, which this function does.
    """
    for i, item in enumerate(items):
        if 'foundation_models' in item.nodeid:
            timeout = 35 * 60 if 'prompt_tuning' in item.name else 2 * 60  # 35 minutes for prompt tuning, 2 mins for other tests
            item.add_marker(pytest.mark.timeout(timeout))


class Credentials(dict):
    """
    Wrapper to search thought the credentials `keys` and search for `secret values`
    then replace them with `****` so they will not be shown in console log
    """

    def __repr__(self):
        secret_dict = {'apikey': '****'}
        tmp = dict(self)
        for el in secret_dict:
            if el in self:
                tmp[el] = secret_dict[el]
        return tmp.__repr__()


def load_original_credentials() -> dict:
    env_creds = get_wml_credentials()
    return Credentials(env_creds)


def load_api_client(original_credentials: dict) -> object:
    api_client = APIClient(original_credentials)
    return api_client


def load_cos_credentials() -> dict:
    cos_credentials = get_cos_credentials()
    return Credentials(cos_credentials)


def check_if_project_in_creds(original_credentials: dict) -> bool:
    if "project_id" in original_credentials:
        print("Used project from config.ini file!")
        return True
    else:
        print("New project will be created!")
        return False


def create_project(api_client: object, cos_credentials: dict, original_credentials: dict) -> str:
    PROJECT_ID_IN_CREDS = check_if_project_in_creds(original_credentials)
    if PROJECT_ID_IN_CREDS:
        return original_credentials['project_id']
    else:

        from datetime import datetime
        creation_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if api_client.ICP_PLATFORM_SPACES:
            # CPD
            base_url = original_credentials['url']
            # guid does not matter here, so it is just an id
            data = {
                "name": f"Auto-created Project - QA - {creation_time}",
                "generator": 'wx-registration-sandbox',
                "public": True,
                'storage': {'type': 'assetfiles',
                            'guid': 'ab8290c7-d26b-4664-a992-917d4a1baxxd'}
            }

        else:
            # CLOUD
            base_url = api_client.PLATFORM_URL
            data = {
                "name": f"Auto-created Project - QA - {creation_time}",
                "generator": "wx-registration-sandbox",
                "description": "A project to try things in",
                "storage": {
                    "type": "bmcos_object_storage",
                    "resource_crn": cos_credentials['resource_instance_id'],
                    "guid": cos_credentials['resource_instance_id'].replace("::", "").split(':')[-1]
                },
                "compute": [
                    {
                        "type": "machine_learning",
                        "name": original_credentials["name"],
                        "crn": original_credentials["iam_serviceid_crn"],
                        "guid": original_credentials["iam_serviceid_crn"].replace("::", "").split(':')[-1]
                    }
                ],
                "type": "wx",
                "public": False
            }

        response = requests.post(
            url=f"{base_url}/transactional/v2/projects",
            json=data,
            headers=api_client._get_headers()
        )
        assert response.status_code == 201, (f"Response code received: {response.status_code} it is not 201,"
                                             f"project couldn't be created")

        project_embedded = response.json()['location'].split('projects/')[-1]
        print(f'\nCreated Project ID: {project_embedded}')

        return project_embedded


def delete_project(project_embedded: str, api_client: object):
    creds = load_original_credentials()
    if "project_id" in creds:
        print("Project was not deleted!")
    else:
        base_url = api_client.PLATFORM_URL if api_client.PLATFORM_URL else api_client.wml_credentials['url']

        response = requests.delete(
            url=f"{base_url}/transactional/v2/projects/{project_embedded}",
            headers=api_client._get_headers()
        )

        assert response.status_code == 204, (f"Response code received: {response.status_code} it is not 204,"
                                             f"project could be not deleted")
        print(f'Project deleted: {response} ')


def load_updated_credentials(new_project_id: str) -> dict:
    """
    Fixture responsible for getting credentials from `config.ini` file
        return:
            dict: Credentials for WML
    """
    env_creds = get_wml_credentials()

    # because we want to be backward compatible if there is no project_id field in credentials we will add it
    env_creds['project_id'] = new_project_id

    return Credentials(env_creds)


@pytest.fixture(scope="session", name="org_creds")
def fixture_org_credentials() -> dict:
    env_creds = load_original_credentials()
    return Credentials(env_creds)


@pytest.fixture(scope="session", name="credentials")
def fixture_credentials_updated(create_project: str) -> dict:
    creds = load_updated_credentials(create_project)
    return creds


@pytest.fixture(scope="session", name="is_project_in_creds")
def fixture_is_project_in_creds(org_creds: dict) -> bool:
    return check_if_project_in_creds(org_creds)


@pytest.fixture(scope="session", name="create_project")
def fixture_create_project(api_client: object, cos_credentials: dict, org_creds: dict, is_project_in_creds: bool) -> str:
    if is_project_in_creds:
        yield org_creds['project_id']
    else:
        new_project_id = create_project(api_client, cos_credentials, org_creds)
        yield new_project_id

        delete_project(new_project_id, api_client)


@pytest.fixture(scope="session", name="project_id")
def fixture_project_id(credentials: dict, create_project: str) -> str:
    project_id = credentials.get("project_id", create_project)
    return project_id


@pytest.fixture(scope="session", name="space_id")
def fixture_space_id(api_client: object, cos_resource_instance_id: str) -> str:
    """
    Fixture responsible for returning space ID
        return:
            str: Space ID
    """
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')
    space_id = get_space_id(api_client, space_name,
                            cos_resource_instance_id=cos_resource_instance_id)
    return space_id


@pytest.fixture(scope="session", name="api_client")
def fixture_api_client(org_creds: dict) -> object:
    """
    Fixture responsible for setup API Client with given credentials.
        return:
            APIClient Object:
    """
    api_client = load_api_client(org_creds)
    return api_client


@pytest.fixture(scope="session", name="cos_credentials")
def fixture_cos_credentials() -> dict:
    """
    Fixture responsible for getting COS credentials
        return:
            dict: COS Credentials
    """
    cos_credentials = load_cos_credentials()
    return Credentials(cos_credentials)


@pytest.fixture(scope="session", name="cos_endpoint")
def fixture_cos_endpoint(cos_credentials: dict) -> str:
    """
    Fixture responsible for getting COS endpoint.
        return:
            str: COS Endpoint
    """
    cos_endpoint = cos_credentials['endpoint_url']
    return cos_endpoint


@pytest.fixture(scope="session", name="cos_resource_instance_id")
def fixture_cos_resource_instance_id(cos_credentials: dict) -> str:
    """
    Fixture responsible for getting COS Instance ID from cos_credentials part of config.ini file
        return:
            str: COS resource instance ID
    """
    cos_resource_instance_id = cos_credentials['resource_instance_id']
    return cos_resource_instance_id


@pytest.fixture(name="space_cleanup")
def fixture_space_clean_up(data_storage, request):
    space_checked = False
    while not space_checked:
        space_cleanup(data_storage.api_client,
                      get_space_id(data_storage.api_client, data_storage.space_name,
                                   cos_resource_instance_id=data_storage.cos_resource_instance_id),
                      days_old=7)
        space_id = get_space_id(data_storage.api_client, data_storage.space_name,
                                cos_resource_instance_id=data_storage.cos_resource_instance_id)
        try:
            assert space_id is not None, "space_id is None"
            data_storage.api_client.spaces.get_details(space_id)

            space_checked = True
        except AssertionError or ApiRequestFailure:
            space_checked = False

    data_storage.space_id = space_id

    if data_storage.SPACE_ONLY:
        data_storage.api_client.set.default_space(data_storage.space_id)
    else:
        data_storage.api_client.set.default_project(data_storage.project_id)


@allure.title("Data Storage Class - initialization")
@pytest.fixture(scope="function", name="data_storage")
def fixture_data_storage_init(api_client, prompt_mgr):
    """
    Every step will be using the same object of DataStorage
    """
    data_storage = DataStorage()
    data_storage.api_client = api_client
    data_storage.prompt_mgr = prompt_mgr
    return data_storage


@allure.title("Universal Steps - initialization")
@pytest.fixture(scope="function", name="universal_step")
def fixture_universal_step_init(data_storage):
    return UniversalSteps(data_storage)


@allure.title("Prompt Tuning Steps - initialization")
@pytest.fixture(scope="function", name="prompt_tuning_step")
def fixture_prompt_tuning_step_init(data_storage):
    return PromptTuningSteps(data_storage)


@allure.title("Prompt Template Steps - initialization")
@pytest.fixture(scope="function", name="prompt_template_step")
def fixture_prompt_template_step_init(data_storage):
    return PromptTemplateSteps(data_storage)
