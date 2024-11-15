#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import pandas as pd
import pytest

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import save_data_to_cos_bucket
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAISync, \
    AbstractTestWebservice
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, RegressionAlgorithms


class TestAutoAIRemote(AbstractTestAutoAISync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = "./autoai/data/PM25.csv"
    data_cos_path = "data/PM25.csv"
    SPACE_ONLY = True
    OPTIMIZER_NAME = "PM25 - time ordered data - test sdk"
    HISTORICAL_RUNS_CHECK = False
    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc="test description",
        prediction_type=PredictionType.REGRESSION,
        prediction_column="pollution",
        scoring=Metrics.MEAN_SQUARED_ERROR,
        include_only_estimators= [RegressionAlgorithms.RF, RegressionAlgorithms.SnapDT],
        max_number_of_estimators=4,
        text_processing=False,
        drop_duplicates=False,
        time_ordered_data=True,
        feature_selector_mode="on"
    )

    def test_00b_prepare_COS_instance(self):
        TestAutoAIRemote.bucket_name = save_data_to_cos_bucket(self.data_location,
                                                               self.data_cos_path,
                                                               access_key_id=self.cos_credentials["cos_hmac_keys"][
                                                                   "access_key_id"],
                                                               secret_access_key=self.cos_credentials["cos_hmac_keys"][
                                                                   "secret_access_key"],
                                                               cos_endpoint=self.cos_endpoint,
                                                               bucket_name=self.bucket_name)

    def test_00c_prepare_connection_to_COS(self):
        connection_details = self.wml_client.connections.create({
            "datasource_type": self.wml_client.connections.get_datasource_type_uid_by_name("bluemixcloudobjectstorage"),
            "name": "Connection to COS for tests",
            "properties": {
                "bucket": self.bucket_name,
                "access_key": self.cos_credentials["cos_hmac_keys"]["access_key_id"],
                "secret_key": self.cos_credentials["cos_hmac_keys"]["secret_access_key"],
                "iam_url": self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                "url": self.cos_endpoint
            }
        })

        TestAutoAIRemote.connection_id = self.wml_client.connections.get_uid(connection_details)

        assert isinstance(self.connection_id,
                          str), f"connection_id={self.connection_id} is not str, but {type(self.connection_id)}"

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
            )
        )
        TestAutoAIRemote.results_connection = None

        assert TestAutoAIRemote.data_connection is not None, f"Incorrect DataConnection object: {TestAutoAIRemote.data_connection}"

    def test_05a_validate_if_feature_selector_params_passed_in_payload(self):
        configuration_pipeline_details = self.wml_client.pipelines.get_details(
            self.remote_auto_pipelines.get_run_details()["entity"]["pipeline"]["id"])
        optimization_params = \
            configuration_pipeline_details["entity"]["document"]["pipelines"][0]["nodes"][0]["parameters"][
                "optimization"]
        params_to_be_checked = ["time_ordered_data", "feature_selector_mode"]
        missing_params = []
        for param in params_to_be_checked:
            try:
                assert optimization_params.get(param) == self.experiment_info.get(param)
            except AssertionError:
                missing_params.append(param)

        assert len(
            missing_params) == 0, f"Missing parameters {missing_params} in training configuration: {optimization_params}"

        print(optimization_params)

    def test_08_get_train_data(self):

        print(self.remote_auto_pipelines.get_run_details())
        X_train, X_holdout, y_train, y_holdout = self.remote_auto_pipelines.get_data_connections()[0].read(
            with_holdout_split=True)

        print("train data sample:")
        print(X_train)
        print(y_train)
        print("holdout data sample:")
        print(X_holdout)
        print(y_holdout)

        AbstractTestAutoAISync.X_df = X_holdout
        AbstractTestAutoAISync.X_values = AbstractTestAutoAISync.X_df.values
        AbstractTestAutoAISync.y_values = y_holdout

        assert len(X_train) > 0, f"Empty X_train {X_train}"
        assert len(X_holdout) > 0, f"Empty X_holdout {X_holdout}"
        assert len(X_train) == len(y_train), f"X and y train have different number of rows"
        assert len(X_holdout) == len(X_holdout), f"X and y holdout have different number of rows"

        # validate if data are time ordered, holdout should be the last values.
        original_df = pd.read_csv(self.data_location)
        X_orig = original_df.drop(self.experiment_info["prediction_column"], axis=1)
        y_orig = original_df[[self.experiment_info["prediction_column"]]]

        assert all(X_orig["date"][-len(y_holdout):].values == X_holdout[
            "date"].values), f"Invalid holdout split for time ordered data. X_holdout: {X_holdout}"
        assert all(y_orig["pollution"][-len(y_holdout):].values == y_holdout[
            "pollution"].values), f"Invalid holdout split for time ordered data. X_holdout: {y_holdout}"
        assert all(X_orig["date"][-len(y_holdout):] == X_holdout[
            "date"]), f"Invalid holdout split for time ordered data. X_holdout: {X_holdout}"

    def test_99_delete_connection_and_connected_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)
        self.wml_client.connections.delete(self.connection_id)

        with pytest.raises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == "__main__":
    unittest.main()
