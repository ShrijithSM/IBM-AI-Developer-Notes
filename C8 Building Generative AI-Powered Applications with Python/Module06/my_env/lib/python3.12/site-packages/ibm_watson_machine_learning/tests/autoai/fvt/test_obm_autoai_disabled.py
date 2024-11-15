#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from copy import copy

import pytest
from ibm_watson_machine_learning.tests.conftest import create_project, load_api_client, load_original_credentials, load_cos_credentials, \
    load_updated_credentials, delete_project

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.preprocessing import DataJoinGraph
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType
from ibm_watson_machine_learning.utils.utils import WMLClientError


class TestAutoAIRemote(unittest.TestCase):
    data_join_graph = None
    project_id = None
    wml_credentials = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.project_id = create_project(load_api_client(load_original_credentials()), load_cos_credentials(),
                                        load_original_credentials())
        cls.wml_credentials = load_updated_credentials(cls.project_id)
        cls.wml_client = APIClient(wml_credentials=copy(cls.wml_credentials))

    @classmethod
    def tearDownClass(cls):
        delete_project(cls.project_id, cls.wml_client)

    def test_01_create_data_join_graph__graph_created(self):
        data_join_graph = DataJoinGraph()
        data_join_graph.node(name="node_daily_sales")
        data_join_graph.node(name="node_methods")
        data_join_graph.node(name="node_retailers")
        data_join_graph.node(name="node_products")
        data_join_graph.node(name="node_1k", main=True)

        data_join_graph.edge(from_node="node_products", to_node="node_1k",
                             from_column=["Product number"], to_column=["Product number"])
        data_join_graph.edge(from_node="node_retailers", to_node="node_1k",
                             from_column=["Retailer code"], to_column=["Retailer code"])
        data_join_graph.edge(from_node="node_methods", to_node="node_daily_sales",
                             from_column=["Order method code"], to_column=["Order method code"])

        TestAutoAIRemote.data_join_graph = data_join_graph

        print(f"data_join_graph: {data_join_graph}")

    def test_02_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                             project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_03_initialize_optimizer_error_received(self):
        experiment_info = dict(
            name="N",
            prediction_type=PredictionType.REGRESSION,
            prediction_column='Quantity',
            data_join_graph=self.data_join_graph
        )
        self.assertIsNotNone(experiment_info.get("data_join_graph"))

        with pytest.raises(WMLClientError) as wml_err:
            _ = self.experiment.optimizer(**experiment_info)

        self.assertIn("joining multiple data sources into a single training data set is not supported",
                      str(wml_err.value))


if __name__ == '__main__':
    unittest.main()
