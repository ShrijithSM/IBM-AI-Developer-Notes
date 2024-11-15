#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc

from ibm_watson_machine_learning.helpers.connections import *
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes, PredictionType
from ibm_watson_machine_learning.deployment import WebService


class BaseTestStoreModel(abc.ABC):
    """
    Note: The tests class need to be run after one of AbstractTestAutoAIAsync or AbstractTestAutoAISync class
    The tests which covers:
    - store model with experiment metadata
    - store model with training_id
    - deployment with lale pipeline
    - deployment deletion
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """
    pipeline_model = None
    remote_auto_pipelines = None
    experiment_metadata: dict = None

    project_models_ids = []
    space_models_ids = []
    online_deployments_ids = []
    batch_deployments_ids = []
    pipeline_model = None

    scoring_df = None

    SKIP_BATCH_DEPLOYMENT = False

    def set_project_space(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_30a_setup_experiment_metadata(self):
        run_details = self.wml_client.training.get_details(self.run_id)
        BaseTestStoreModel.experiment_metadata = self.experiment_info

        if self.experiment_info.get("prediction_type") == PredictionType.FORECASTING:
            model_path = "/data/autoai-ts/model.zip"
        else:
            model_path = "/data/automl/model.zip"

        match run_details['entity']['results_reference']['type']:
            case DataConnectionTypes.CA:
                training_result_reference = DataConnection(
                    connection_asset_id=run_details['entity']['results_reference']['connection']['id'],
                    location=S3Location(
                        bucket=run_details['entity']['results_reference']['location']['bucket'],
                        path=self.data_cos_path,
                        model_location=f"{run_details['entity']['results_reference']['location']['training']}" +
                                       model_path,
                        training_status=run_details['entity']['results_reference']['location']['training_status']
                    )
                )
            case DataConnectionTypes.CN:
                training_result_reference = DataConnection(
                    location=ContainerLocation(
                        path=f"{run_details['entity']['results_reference']['location']['training']}/data/automl",
                        model_location=f"{run_details['entity']['results_reference']['location']['training']}" +
                                       model_path,
                        training_status=run_details['entity']['results_reference']['location']['training_status']
                    )
                )
            case DataConnectionTypes.FS:
                training_result_reference = DataConnection(
                    location=FSLocation(
                        path=f"{run_details['entity']['results_reference']['location']['path']}/{self.run_id}/data/automl"
                    )
                )

        BaseTestStoreModel.experiment_metadata.update({"training_data_references": [self.data_connection],
                                                       "training_result_reference": training_result_reference})

    def test_30b_store_model_from_object_with_experiment_metadata(self):
        self.set_project_space()
        BaseTestStoreModel.pipeline_model = self.remote_auto_pipelines.get_pipeline(self.pipeline_to_deploy)
        model_metadata = {
            self.wml_client.repository.ModelMetaNames.NAME: 'AutoAI wml_client.repository.store_model() with experiment_metadata'
        }

        stored_model_details = self.wml_client.repository.store_model(model=BaseTestStoreModel.pipeline_model,
                                                                      meta_props=model_metadata,
                                                                      experiment_metadata=BaseTestStoreModel.experiment_metadata)

        self.assertIsNotNone(stored_model_details)
        print(stored_model_details)

        model_id = self.wml_client.repository.get_model_id(stored_model_details)
        model_details = self.wml_client.repository.get_model_details(model_id)
        self.assertIn(model_id, str(model_details))
        BaseTestStoreModel.project_models_ids.append(model_id)

    def test_30c_store_model_from_pipeline_name_with_experiment_metadata(self):
        self.set_project_space()

        model_metadata = {
            self.wml_client.repository.ModelMetaNames.NAME: 'AutoAI wml_client.repository.store_model() with experiment_metadata'
        }

        stored_model_details = self.wml_client.repository.store_model(model=self.pipeline_to_deploy,
                                                                      meta_props=model_metadata,
                                                                      experiment_metadata=BaseTestStoreModel.experiment_metadata)

        self.assertIsNotNone(stored_model_details)
        print(stored_model_details)
        model_id = self.wml_client.repository.get_model_id(stored_model_details)
        model_details = self.wml_client.repository.get_model_details(model_id)
        self.assertIn(model_id, str(model_details))
        BaseTestStoreModel.project_models_ids.append(model_id)

    def test_31a_store_model_from_object__with_training_id(self):
        self.set_project_space()

        if BaseTestStoreModel.pipeline_model is None:
            BaseTestStoreModel.pipeline_model = self.remote_auto_pipelines.get_pipeline(self.pipeline_to_deploy)

        model_metadata = {
            self.wml_client.repository.ModelMetaNames.NAME: 'AutoAI wml_client.repository.store_model() with training_id'
        }

        stored_model_details = self.wml_client.repository.store_model(model=BaseTestStoreModel.pipeline_model,
                                                                      meta_props=model_metadata,
                                                                      training_id=self.run_id)

        self.assertIsNotNone(stored_model_details)
        print(stored_model_details)

        model_id = self.wml_client.repository.get_model_id(stored_model_details)
        model_details = self.wml_client.repository.get_model_details(model_id)
        self.assertIn(model_id, str(model_details))
        BaseTestStoreModel.project_models_ids.append(model_id)

    def test_31b_store_model_from_pipeline_name_with_training_id(self):
        self.set_project_space()
        model_metadata = {
            self.wml_client.repository.ModelMetaNames.NAME: 'AutoAI wml_client.repository.store_model() with training_id'
        }

        stored_model_details = self.wml_client.repository.store_model(model=self.pipeline_to_deploy,
                                                                      meta_props=model_metadata,
                                                                      training_id=self.run_id)

        self.assertIsNotNone(stored_model_details)
        print(stored_model_details)

        model_id = self.wml_client.repository.get_model_id(stored_model_details)
        model_details = self.wml_client.repository.get_model_details(model_id)
        self.assertIn(model_id, str(model_details))
        BaseTestStoreModel.project_models_ids.append(model_id)

    def test_32_promote_models_to_space(self):
        if self.SPACE_ONLY:
            BaseTestStoreModel.space_models_ids = self.project_models_ids
        else:
            for model_id in BaseTestStoreModel.project_models_ids:
                model_details = self.wml_client.repository.get_model_details(model_id)
                self.assertIn(model_id, str(model_details))
                BaseTestStoreModel.space_models_ids.append(
                    self.wml_client.spaces.promote(asset_id=model_id,
                                                   source_project_id=self.project_id,
                                                   target_space_id=self.space_id))
        self.wml_client.set.default_space(self.space_id)
        for model_id in BaseTestStoreModel.space_models_ids:
            promoted_model_details = self.wml_client.repository.get_model_details(model_id)
            self.assertIn(model_id, str(promoted_model_details))
            self.assertIsInstance(promoted_model_details, dict)

    def test_33a_deploy_model_web_service_with_experiment_metadata(self):
        deployment_name = 'Deployment_webservice_test'

        if self.SPACE_ONLY:
            service = WebService(
                source_wml_credentials=self.wml_credentials,
                target_wml_credentials=self.wml_credentials,
                source_space_id=self.space_id,
                target_space_id=self.space_id
            )
        else:
            service = WebService(
                source_wml_credentials=self.wml_credentials,
                target_wml_credentials=self.wml_credentials,
                source_project_id=self.project_id,
                target_space_id=self.space_id
            )
        service.create(
            model=self.pipeline_to_deploy,
            metadata=self.experiment_metadata,
            deployment_name=deployment_name
        )

        params = service.get_params()
        print(params)
        self.assertIsInstance(params, dict, msg="service.get_params() did not return dict")
        deployment_id = params['metadata']['id']
        self.assertEqual(service.id, deployment_id, msg="service.id is different than deployment_id")
        self.assertEqual(params['entity']['name'], deployment_name,
                         msg=f"Name expected: {deployment_name}, actual: {params['entity']['name']}")

        service.delete(deployment_id)
        with self.assertRaises(WMLClientError, msg="It should throw an exception due to missing deployment ID"):
            self.wml_client.deployments.get_details(deployment_id)

    def test_33b_deploy_online_models(self):
        for model_id in self.space_models_ids:
            model_details = self.wml_client.repository.get_model_details(model_id)
            self.assertIn(model_id, str(model_details))
            deploy_meta = {
                self.wml_client.deployments.ConfigurationMetaNames.NAME: "Tests: AutoAI model stored with repository.store_model()",
                self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
            }

            deployment_details = self.wml_client.deployments.create(artifact_uid=model_id, meta_props=deploy_meta)
            BaseTestStoreModel.online_deployments_ids.append(self.wml_client.deployments.get_id(deployment_details))

    def test_34_score_online_deployments(self):
        if hasattr(self, 'X_df'):
            self.scoring_df = self.X_df[:5]
        import pandas as pd

        scoring_payload = {
            "input_data": [{
                'values': pd.DataFrame(self.scoring_df)
            }]
        }
        for deployment_id in self.online_deployments_ids:
            print(f"Scoring deployment: {deployment_id}...")
            predictions = self.wml_client.deployments.score(deployment_id, scoring_payload)
            print(predictions)
            self.assertIn('predictions', predictions)

    def test_35_deploy_batch_models(self):
        if self.SKIP_BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployments.")

        for model_id in self.space_models_ids:
            deployment_props = {
                self.wml_client.deployments.ConfigurationMetaNames.NAME: "Tests: AutoAI model stored with repository.store_model()",
                self.wml_client.deployments.ConfigurationMetaNames.BATCH: {},
                self.wml_client.deployments.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS: [
                    {
                        'node_runtime_id': 'auto_ai.kb',
                        'hardware_spec': {
                            'name': 'M'
                        }
                    }
                ]
            }
            deployment_details = self.wml_client.deployments.create(
                artifact_uid=model_id,
                meta_props=deployment_props)
            BaseTestStoreModel.batch_deployments_ids.append(self.wml_client.deployments.get_id(deployment_details))

    def test_36_score_batch_deployments(self):
        if self.SKIP_BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployments.")

        if hasattr(self, 'X_df'):
            self.scoring_df = self.X_df[:5]

        import time
        for deployment_id in self.batch_deployments_ids:
            print(f"Scoring deployment: {deployment_id}...")
            scoring_payload = {
                self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{'values': self.scoring_df}]
            }
            job_details = self.wml_client.deployments.create_job(deployment_id, scoring_payload)
            deployment_job_id = self.wml_client.deployments.get_job_uid(job_details)
            print("Batch deployment job created, waining for job to complete...")
            status_state = job_details['entity']['scoring']['status']['state']
            while status_state not in ['failed', 'error', 'completed', 'canceled']:
                time.sleep(10)
                print(".", end=" ")
                status_state = \
                    self.wml_client.deployments.get_job_details(deployment_job_id)['entity']['scoring']['status'][
                        'state']
            print(f"Batch deployment job status state: {status_state}")
            self.assertEqual(status_state, 'completed')
            predictions = self.wml_client.deployments.get_job_details(deployment_job_id)
            print(predictions)
            self.assertIsNotNone(predictions)

    def test_91_delete_deployments(self):
        self.batch_deployments_ids.extend(self.online_deployments_ids)
        for deployment_id in self.batch_deployments_ids:
            self.wml_client.deployments.delete(deployment_id)
            with self.assertRaises(WMLClientError):
                self.wml_client.deployments.get_details(deployment_id)

    def test_92_delete_models(self):
        def delete_models(models_ids):
            for model_id in models_ids:
                self.wml_client.repository.delete(model_id)
                with self.assertRaises(WMLClientError):
                    self.wml_client.repository.get_model_details(model_id)

        if not self.wml_client.default_space_id:
            self.wml_client.set.default_space(self.space_id)
            delete_models(self.space_models_ids)

        if self.space_models_ids != self.project_models_ids:
            self.wml_client.set.default_project(self.project_id)
            delete_models(self.project_models_ids)

        BaseTestStoreModel.project_models_ids = []
        BaseTestStoreModel.space_models_ids = []
        BaseTestStoreModel.online_deployments_ids = []
        BaseTestStoreModel.batch_deployments_ids = []