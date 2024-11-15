#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest

from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.helpers.connections.connections import S3Location
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos, save_data_to_container, delete_container
from ibm_watson_machine_learning.wml_client_error import WMLClientError


class PromptTuningSteps:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def write_data_to_container(self):
        save_data_to_container(self.data_storage.data_location, self.data_storage.data_cos_path,
                               self.data_storage.api_client)

    def prepare_COS_instance_and_connection(self):
        self.data_storage.connection_id, self.data_storage.bucket_name = create_connection_to_cos(
            wml_client=self.data_storage.api_client,
            cos_credentials=self.data_storage.cos_credentials,
            cos_endpoint=self.data_storage.cos_endpoint,
            bucket_name=self.data_storage.bucket_name,
            save_data=True,
            data_path=self.data_storage.data_location,
            data_cos_path=self.data_storage.data_cos_path
        )

        assert isinstance(self.data_storage.connection_id, str), "connection_id it is not String"

    def data_reference_setup_c(self):
        self.data_storage.train_data_connections = [DataConnection(
            location=ContainerLocation(path=self.data_storage.data_cos_path
                                       ))]
        assert len(
            self.data_storage.train_data_connections) == 1, "train_data_connection length it is different than '1'"

    def data_reference_setup_ca_default(self):
        self.data_storage.train_data_connections = [DataConnection(
            connection_asset_id=self.data_storage.connection_id,
            location=S3Location(
                bucket=self.data_storage.bucket_name,
                path=self.data_storage.data_cos_path
            )
        )]
        self.data_storage.results_data_connection = DataConnection(
            connection_asset_id=self.data_storage.connection_id,
            location=S3Location(
                bucket=self.data_storage.bucket_name,
                path=self.data_storage.results_cos_path
            )
        )

        assert len(
            self.data_storage.train_data_connections) == 1, "train_data_connection length it is different than '1'"
        assert self.data_storage.results_data_connection is not None, "train_data_connection cannot be None"

    def data_reference_setup_da(self):
        self.data_storage.train_data_connections = [DataConnection(data_asset_id=self.data_storage.asset_id)]

        assert len(
            self.data_storage.train_data_connections) == 1, "train_data_connection length it is different than '1'"

    def read_saved_remote_data_before_fit(self):
        self.data_storage.train_data_connections[0].set_client(self.data_storage.api_client)
        data = self.data_storage.train_data_connections[0].read(raw=True, binary=True)

        assert isinstance(data, bytes), "data it is not Bytes type"

    def prepare_data_asset(self):
        asset_details = self.data_storage.api_client.data_assets.create(
            name=self.data_storage.data_location.split('/')[-1],
            file_path=self.data_storage.data_location)

        self.data_storage.asset_id = self.data_storage.api_client.data_assets.get_id(asset_details)
        assert isinstance(self.data_storage.asset_id, str), "assert_id it is not a String type"

    def read_results_reference_filename(self):
        parameters = self.data_storage.prompt_tuner.get_run_details()
        print(parameters)

        assert parameters is not None, "parameters cannot be None"
        assert parameters['entity']['results_reference']['location'][
                   'file_name'] == self.data_storage.results_cos_path, "parameters are not equal to data_storage.results_cos_path"

    def delete_connection_and_connected_data_asset(self):
        self.data_storage.api_client.connections.delete(self.data_storage.connection_id)

        with pytest.raises(WMLClientError):
            self.data_storage.api_client.connections.get_details(self.data_storage.connection_id)

    def delete_data_asset(self):
        self.data_storage.api_client.data_assets.delete(self.data_storage.asset_id)

        with pytest.raises(WMLClientError):
            self.data_storage.api_client.data_assets.get_details(self.data_storage.asset_id)

    def delete_container(self):
        status = delete_container(self.data_storage.api_client, self.data_storage.data_cos_path)
        print(status)

        for inner_dict in status.values():
            if 204 in inner_dict.values():
                response_code = True

        if not response_code:
            f"Code:'204' it is not in a {status}"
