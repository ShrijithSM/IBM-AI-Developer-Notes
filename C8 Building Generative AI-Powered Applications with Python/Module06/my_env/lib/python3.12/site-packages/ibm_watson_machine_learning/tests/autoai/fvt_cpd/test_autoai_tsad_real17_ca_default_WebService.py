#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import pytest

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestTSADAsync
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType


class TestAutoAIRemote(AbstractTestTSADAsync, unittest.TestCase):
    """
    The test can be run on Cloud only
    """

    cos_resource = None
    data_location = './autoai/data/real_17.csv'

    data_cos_path = 'real_17.csv'

    batch_payload_location = data_location
    batch_payload_cos_location = data_cos_path

    SPACE_ONLY = False
    ONLINE_DEPLOYMENT = True
    BATCH_DEPLOYMENT = False
    NOTEBOOK_CHECK = False
    LALE_CHECK = False

    OPTIMIZER_NAME = "Timeseries anomaly prediction real_17"
    DEPLOYMENT_NAME = OPTIMIZER_NAME + "Deployment"

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.TIMESERIES_ANOMALY_PREDICTION,
        feature_columns=['value'],
        timestamp_column_name='timestamp',
        lookback_window=10,
        max_number_of_estimators=2,
        retrain_on_holdout=False,
        notebooks=True
    )

    def test_00b_prepare_COS_instance_and_connection(self):
        TestAutoAIRemote.connection_id, TestAutoAIRemote.bucket_name = create_connection_to_cos(
            wml_client=self.wml_client,
            cos_credentials=self.cos_credentials,
            cos_endpoint=self.cos_endpoint,
            bucket_name=self.bucket_name,
            save_data=True,
            data_path=self.data_location,
            data_cos_path=self.data_cos_path)

        assert isinstance(self.connection_id, str), "connection_id is not string"

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
            )
        )

        assert TestAutoAIRemote.data_connection is not None, f"data_connection is None: {TestAutoAIRemote.data_connection}"

    def test_05a_param_in_configuration(self):
        optimizer_id = self.remote_auto_pipelines._workspace.wml_client.training.get_details(
            training_uid=self.run_id
        ).get('entity')['pipeline']['id']
        optimizer_config = self.remote_auto_pipelines._workspace.wml_client.pipelines.get_details(pipeline_uid=optimizer_id)
        ts_parameters = optimizer_config['entity']['document']['pipelines'][0]['nodes'][0]['parameters']['optimization']
        print(ts_parameters)

        assert 'retrain_on_holdout' in ts_parameters, f"retrain_on_holdout missing in optimization parameters: {ts_parameters}"

    def test_99_delete_connection(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.connections.delete(self.connection_id)

        with pytest.raises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
