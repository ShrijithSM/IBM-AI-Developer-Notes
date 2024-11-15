#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import os
import pytest

from os.path import join
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestTSAsync, \
    BaseTestStoreModel
from ibm_watson_machine_learning.helpers.connections import DeploymentOutputAssetLocation
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, ImputationStrategy, ForecastingPipelineTypes


class TestAutoAIRemote(AbstractTestTSAsync, BaseTestStoreModel, unittest.TestCase):
    """
    The test can be run on Cloud only
    """

    cos_resource = None
    data_location = './autoai/data/FoodDemand_meal1885_center55.csv'

    data_cos_path = 'FoodDemand_meal1885_center55.csv'

    batch_payload_location = data_location
    batch_payload_cos_location = data_cos_path

    SPACE_ONLY = True
    HISTORICAL_RUNS_CHECK = False
    BATCH_DEPLOYMENT = True
    BATCH_DEPLOYMENT_WITH_DA = True
    BATCH_DEPLOYMENT_WITH_DF = True  # batch input passed as Pandas.DataFrame
    BATCH_DEPLOYMENT_WITH_CDA = False  # batch input passed as DataConnection type data-assets with connection_id(COS)
    BATCH_DEPLOYMENT_WITH_CA = False  # batch input passed as DataConnection type connected assets (COS)

    OPTIMIZER_NAME = "Food Demand Nans test sdk"
    DEPLOYMENT_NAME = OPTIMIZER_NAME + "Deployment"

    target_space_id = None
    summary_exogenous = None
    summary_non_exogenous = None

    observations = None
    supporting_features = None

    pipeline_to_deploy = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.FORECASTING,
        prediction_columns=['num_orders'],
        feature_columns=['checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders'],
        pipeline_types=[ForecastingPipelineTypes.FlattenEnsembler] + ForecastingPipelineTypes.get_exogenous(),
        supporting_features_at_forecast=True,
        timestamp_column_name='week',
        numerical_imputation_strategy=ImputationStrategy.MEDIAN,
        max_number_of_estimators=7,
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

        assert isinstance(self.connection_id,
                          str), f"connection_id={self.connection_id} is not str, but {type(self.connection_id)}"

    def test_00d_prepare_connected_data_asset(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: f"{self.data_cos_path} - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.bucket_name,
                                                                                       self.data_cos_path)
        })
        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)

        assert isinstance(self.asset_id,
                          str), f"asset_id={self.asset_id} is not str, but {type(self.asset_id)}"

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.results_connection = None

        assert TestAutoAIRemote.data_connection is not None, "data_connection is empty, None"

    def test_06_get_train_data(self):
        AbstractTestTSAsync.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())

        assert len(self.train_data) > 0, "train_data DataFrame is empty"

        AbstractTestTSAsync.train_X, AbstractTestTSAsync.test_X, AbstractTestTSAsync.train_y, AbstractTestTSAsync.test_y = \
            self.remote_auto_pipelines.get_data_connections()[0].read(with_holdout_split=True)

        AbstractTestTSAsync.scoring_df = self.train_X[:10]

        print("train data sample:")
        print(self.train_X.head())

        assert len(self.train_X) > 0, "train_X DataFrame is empty"

        features_only = [feature for feature in self.experiment_info['feature_columns'] if
                         feature not in self.experiment_info['prediction_columns']]

        AbstractTestTSAsync.observations = self.train_data[self.experiment_info['feature_columns']][-10:-5]
        AbstractTestTSAsync.supporting_features = self.train_data[features_only][-5:]

    def test_09a_predict_exogenous_pipelines_as_sklearn(self):
        pipelines_summary = self.remote_auto_pipelines.summary()

        AbstractTestTSAsync.winning_pipelines_summary = pipelines_summary[pipelines_summary['Winner']]
        TestAutoAIRemote.summary_exogenous = self.winning_pipelines_summary[
            pipelines_summary['Enhancements'].str.contains('SUP')]
        TestAutoAIRemote.summary_non_exogenous = self.winning_pipelines_summary.loc[
                                                 set(self.winning_pipelines_summary.index) - set(
                                                     TestAutoAIRemote.summary_exogenous.index), :]

        failed_pipelines = []
        for pipeline_name in self.summary_exogenous.index:
            print(pipeline_name)
            try:
                pipeline_model = self.remote_auto_pipelines.get_pipeline(pipeline_name, astype='sklearn')
                predictions = pipeline_model.predict(self.observations, supporting_features=self.supporting_features)
                print(predictions)
                assert len(predictions) > 0
            except Exception as e:
                failed_pipelines.append(pipeline_name)
                print(e)

        TestAutoAIRemote.pipeline_to_deploy = pipeline_name

        assert len(failed_pipelines) == 0, f"Some Pipelines failed: {failed_pipelines}"

    def test_09b_predict_exogenous_pipelines_as_lale(self):
        failed_pipelines = []
        for pipeline_name in self.summary_exogenous.index:
            print(pipeline_name)
            try:
                pipeline_model = self.remote_auto_pipelines.get_pipeline(pipeline_name, astype='lale')
                predictions = pipeline_model.predict(self.observations, supporting_features=self.supporting_features)
                print(predictions)
                assert len(predictions) > 0
            except Exception as e:
                failed_pipelines.append(pipeline_name)
                print(e)

        assert len(failed_pipelines) == 0, f"Some Pipelines failed: {failed_pipelines}"

    def test_09c_predict_non_exogenous_pipelines_as_lale(self):
        failed_pipelines = []
        for pipeline_name in self.summary_non_exogenous.index:
            print(pipeline_name)
            try:
                pipeline_model = self.remote_auto_pipelines.get_pipeline(pipeline_name, astype='lale')
                predictions = pipeline_model.predict(self.scoring_df[self.experiment_info['prediction_columns']].values)
                print(predictions)
                assert len(predictions) > 0
            except Exception as e:
                failed_pipelines.append(pipeline_name)
                print(e)

        assert len(failed_pipelines) == 0, f"Some Pipelines failed: {failed_pipelines}"

    def test_09d_predict_using_all_fitted_pipelines_no_payload(self):
        failed_pipelines = []
        for pipeline_name in self.winning_pipelines_summary.index:
            print(pipeline_name)
            try:
                pipeline_model = self.remote_auto_pipelines.get_pipeline(pipeline_name, astype='lale')
                predictions = pipeline_model.predict()
                print(predictions)
                assert len(predictions) > 0
            except Exception as e:
                failed_pipelines.append(pipeline_name)
                print(e)

        assert len(failed_pipelines) == 0, f"Some Pipelines failed: {failed_pipelines}"

    def test_34a_score_deployed_model_with_df_observations_and_supporting_features(self):
        predictions = AbstractTestTSAsync.service.score(payload={'observations': AbstractTestTSAsync.observations,
                                                                 'supporting_features': AbstractTestTSAsync.supporting_features})
        print(predictions)

        assert predictions is not None, "Scoring predictions are empty, None"

        forecast_window = self.experiment_info.get('forecast_window', 1)

        assert len(predictions['predictions'][0][
                       'values']) == forecast_window, "Scoring prediction values are not equal forecast_window"
        assert len(predictions['predictions'][0]['values'][0]) == len(
            self.experiment_info.get('prediction_columns',
                                     [])), "Prediction values does not have prediction for all columns"

    def test_44a_run_job__batch_deployed_model_with_data_frame_observations_and_supporting_features(self):
        if not self.BATCH_DEPLOYMENT or not self.BATCH_DEPLOYMENT_WITH_DF:
            self.skipTest("Skip batch deployment")

        scoring_params = AbstractTestTSAsync.service_batch.run_job(
            payload={'observations': AbstractTestTSAsync.observations,
                     'supporting_features': AbstractTestTSAsync.supporting_features},
            background_mode=False)
        print(scoring_params)
        print(AbstractTestTSAsync.service_batch.get_job_result(scoring_params['metadata']['id']))

        assert scoring_params is not None
        assert 'predictions' in str(scoring_params).lower()

    def test_45_run_job_batch_deployed_model_with_data_assets(self):
        if not self.BATCH_DEPLOYMENT or not self.BATCH_DEPLOYMENT_WITH_DA:
            self.skipTest("Skip batch deployment")

        data_connections_space_only = []

        test_case_batch_output_filename = "da_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        assert self.batch_payload_location is not None, "Test configuration failure: Batch payload location is missing"

        tmp_filename = 'tmp_scoring.csv'
        data_connections_space_only = []
        for data_df in [AbstractTestTSAsync.observations, AbstractTestTSAsync.supporting_features]:
            data_df.to_csv(tmp_filename, index=False)
            asset_details = self.wml_client.data_assets.create(
                name=self.batch_payload_location.split('/')[-1],
                file_path=tmp_filename)
            asset_id = self.wml_client.data_assets.get_id(asset_details)
            AbstractTestTSAsync.data_assets_to_delete.append(asset_id)
            data_connections_space_only.append(DataConnection(data_asset_id=asset_id))
            os.remove(tmp_filename)

        scoring_params = AbstractTestTSAsync.service_batch.run_job(
            payload={'observations': data_connections_space_only[0],
                     'supporting_features': data_connections_space_only[1]},
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        assert scoring_params is not None

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        data_asset_details = self.wml_client.data_assets.get_details()
        predictions = AbstractTestTSAsync.service_batch.get_job_result(deployment_job_id)
        print(predictions)

        assert test_case_batch_output_filename in str(
            data_asset_details), f"Batch output file: {test_case_batch_output_filename} missed in data assets"
        assert predictions is not None

    def test_99_delete_connection_and_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.connections.delete(self.connection_id)
        self.wml_client.data_assets.delete(self.asset_id)

        with pytest.raises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)

        with pytest.raises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)


if __name__ == '__main__':
    unittest.main()
