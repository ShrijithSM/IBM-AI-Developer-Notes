#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest
from pytest_mock import MockerFixture
from datetime import datetime, timedelta

from ibm_watson_machine_learning import APIClient


@pytest.fixture(scope="function", name="mock_get")
def fixture_mock_get(mocker: MockerFixture):
    mock = mocker.Mock()
    mocker.patch("requests.get", return_value=mock)
    yield mock


@pytest.fixture(scope="function", name="mock_post")
def fixture_mock_post(mocker: MockerFixture):
    mock = mocker.Mock()
    mocker.patch("requests.post", return_value=mock)
    yield mock


@pytest.fixture(scope="function", name="api_client_mock")
def fixture_api_client_mock(mock_get, mock_post, credentials):
    """
    Fixture responsible for setup API Client with given credentials.
        Args:
            mocker
        return:
            APIClient Object:
    """
    from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan

    mock_get.status_code = 200
    mock_post.status_code = 200

    mock_post.json.return_value = {"access_token": "token", "expires_in": 900}
    mock_get.json.return_value = {"accessToken": "token"}

    def mocked_get_token(cls, *args, **kwargs):
        return "token"

    def mocked_get_expiration_datetime(cls, *args, **kwargs):
        return datetime.now() + timedelta(minutes=20)

    plan = ServiceInstanceNewPlan

    valid_get_token = plan._get_token
    plan._get_token = mocked_get_token

    valid_get_expiration_datetime = plan._get_expiration_datetime
    plan._get_expiration_datetime = mocked_get_expiration_datetime

    api_client = APIClient(credentials)

    mock_get.json.return_value = {"entity": {"storage": {"type": "project_type"}}}
    api_client.set.default_project("project_id")

    yield api_client

    # Undo changes
    plan._get_token = valid_get_token
    plan._get_expiration_datetime = valid_get_expiration_datetime
