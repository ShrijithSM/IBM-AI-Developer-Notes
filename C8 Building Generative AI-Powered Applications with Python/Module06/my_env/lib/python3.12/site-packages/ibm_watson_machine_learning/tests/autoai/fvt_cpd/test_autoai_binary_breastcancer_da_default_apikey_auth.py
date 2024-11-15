#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync
from ibm_watson_machine_learning.tests.conftest import load_updated_credentials, create_project, load_api_client, \
    load_cos_credentials, load_original_credentials, delete_project
from ibm_watson_machine_learning.tests.utils import is_cp4d
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError


@unittest.skipIf(not is_cp4d(), "Not supported on cloud")
class TestAutoAIRemote(AbstractTestAutoAIAsync, unittest.TestCase):
    """
    The test can be run on CPD only
    """

    cos_resource = None
    data_location = './autoai/data/breast_cancer.csv'

    data_cos_path = 'data/breast_cancer.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "breast_cancer test sdk"

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.BINARY,
        prediction_column='diagnosis',
        positive_label='M',
        scoring=Metrics.AVERAGE_PRECISION_SCORE,
        max_number_of_estimators=1,
        include_only_estimators=[ClassificationAlgorithms.LR]

    )

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.project_id = create_project(load_api_client(load_original_credentials()), load_cos_credentials(),
                                        load_original_credentials())
        cls.wml_credentials = load_updated_credentials(cls.project_id)
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials, project_id=cls.project_id)

        try:
            del cls.wml_credentials['password']
        except:
            pass
        try:
            del cls.wml_credentials['bedrock_url']
        except:
            pass

        assert (cls.wml_credentials.get('username') is not None)
        assert (cls.wml_credentials.get('password') is None)
        assert (cls.wml_credentials.get('bedrock_url') is None)
        assert (cls.wml_credentials.get('apikey') is not None)

    @classmethod
    def teardown_class(cls):
        delete_project(cls.project_id, cls.wml_client)

    def test_00d_prepare_data_asset(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.data_location.split('/')[-1],
            file_path=self.data_location)

        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_99_delete_data_asset(self):
        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)


if __name__ == '__main__':
    unittest.main()
