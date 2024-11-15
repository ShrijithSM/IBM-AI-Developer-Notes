#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil
import unittest
import datetime

import joblib
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import svm

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestScikitLearnDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying scikit-learn
    using directory.
    """
    deployment_type = "scikit-learn_1.1"
    software_specification_name = "runtime-23.1-py3.10"
    model_name = deployment_name = "sklearn_model_from_directory"
    file_name = 'scikit_model_' + datetime.datetime.now().isoformat()
    IS_MODEL = True

    def get_model(self):
        model_data = datasets.load_digits()
        scaler = preprocessing.StandardScaler()
        clf = svm.SVC(kernel='rbf')
        pipeline = Pipeline([('scaler', scaler), ('svc', clf)])
        model = pipeline.fit(model_data.data, model_data.target)

        TestScikitLearnDeployment.training_data = pd.DataFrame(model_data.data, columns=[str(i) for i in range(model_data.data.shape[-1])])  # columns names needs to be string
        TestScikitLearnDeployment.training_target = pd.Series(model_data.target)

        TestScikitLearnDeployment.full_path = os.path.join(os.getcwd(), 'base', 'artifacts', self.file_name)
        filename = self.file_name + '.pkl'
        os.makedirs(self.full_path, exist_ok=True)
        joblib.dump(model, os.path.join(self.full_path, filename))

        return self.full_path

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                'values': [
                    [0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0,
                     16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0,
                     0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0,
                     16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
            }]
        }

    def test_01_store_model(self):
        model_props = self.create_model_props()
        self.model = self.get_model()

        model_details = self.wml_client.repository.store_model(
            meta_props=model_props,
            model=self.model,
            training_data=self.training_data,
            training_target=self.training_target
        )
        TestScikitLearnDeployment.model_id = self.wml_client.repository.get_model_id(model_details)
        TestScikitLearnDeployment.model_href = self.wml_client.repository.get_model_href(model_details)

        self.assertIsNotNone(self.model_href)
        self.assertIsNotNone(self.model_id)

    def test_17_delete_directory(self):
        shutil.rmtree(self.full_path)


if __name__ == "__main__":
    unittest.main()
