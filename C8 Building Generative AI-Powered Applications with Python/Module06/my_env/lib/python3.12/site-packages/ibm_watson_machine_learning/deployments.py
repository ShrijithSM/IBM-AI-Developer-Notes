#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
from typing import Optional, Literal
import numpy as np
import json
import inspect

import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import DEPLOYMENT_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, print_text_header_h1, print_text_header_h2, \
    StatusLogger
from ibm_watson_machine_learning.utils.utils import _check_if_import_from_watsonx_ai
from ibm_watson_machine_learning.wml_client_error import (WMLClientError, MissingValue, 
                                                          NoVirtualDeploymentSupportedForICP,  
                                                          InvalidValue, MissingArgument, PromptVariablesError)
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.messages.messages import Messages
from ibm_watson_machine_learning.metanames import DeploymentMetaNames, ScoringMetaNames, DecisionOptimizationMetaNames, \
    DeploymentNewMetaNames, _check_spec_uids
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.utils.autoai.utils import all_logging_disabled
#from ibm_watson_machine_learning.metanames import DeploymentMetaNames, ScoreMetaNames, EnvironmentMetaNames

lib_checker = LibraryChecker()


class Deployments(WMLResource):
    """Deploy and score published artifacts (models and functions).
    """
    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.CLOUD_PLATFORM_SPACES and not client.ICP_PLATFORM_SPACES:
            Deployments._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Deployments._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        if client.CLOUD_PLATFORM_SPACES or client.ICP_PLATFORM_SPACES:
            self.ConfigurationMetaNames = DeploymentNewMetaNames()
        else:
            self.ConfigurationMetaNames = DeploymentMetaNames()

        if client.CLOUD_PLATFORM_SPACES:
            Deployments.cloud_platform_spaces = True

        if client.ICP_PLATFORM_SPACES:
            Deployments.icp_platform_spaces = True

        self.ScoringMetaNames = ScoringMetaNames()
        self.DecisionOptimizationMetaNames = DecisionOptimizationMetaNames()

    def _deployment_status_errors_handling(self, deployment_details, operation_name, deployment_id):
        try:
            if 'failure' in deployment_details['entity']['status']:
                errors = deployment_details[u'entity'][u'status'][u'failure'][u'errors']
                for error in errors:
                    if type(error) == str:
                        try:
                            error_obj = json.loads(error)
                            print(error_obj[u'message'])
                        except:
                            print(error)
                    elif type(error) == dict:
                        print(error['message'])
                    else:
                        print(error)
                raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id +'. Errors: ' + str(errors))
            else:
                print(deployment_details['entity']['status'])
                raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id +'. Error: ' + str(deployment_details['entity']['status']['state']))
        except WMLClientError as e:
            raise e
        except Exception as e:
            self._logger.debug('Deployment ' + operation_name + ' failed: ' + str(e))
            print(deployment_details['entity']['status']['failure'])
            raise WMLClientError('Deployment ' + operation_name + ' failed for deployment id: ' + deployment_id + '.')

    # TODO model_uid and artifact_uid should be changed to artifact_uid only
    def create(self, artifact_uid=None, meta_props=None, rev_id=None, **kwargs):
        """Create a deployment from an artifact. As artifact, we understand model or function which may be deployed.

        :param artifact_uid:  published artifact UID (model or function uid)
        :type artifact_uid: str

        :param meta_props: metaprops, to see the available list of metanames use:

            .. code-block:: python

                client.deployments.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the created deployment
        :rtype: dict

        **Example**

        .. code-block:: python

            meta_props = {
                wml_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
                wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
                wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : { "id":  "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"},
                wml_client.deployments.ConfigurationMetaNames.SERVING_NAME : 'sample_deployment'
            }
            deployment_details = client.deployments.create(artifact_uid, meta_props)

        """
        ##To be removed once deployments adds support for projects
        WMLResource._chk_and_block_create_update_for_python36(self)

        Deployments._validate_type(artifact_uid, u'artifact_uid', str, True)

        _check_spec_uids(meta_props)

        if self._ICP:
            predictionUrl = self._wml_credentials[u'url']

        if meta_props is None:
            raise WMLClientError("Invalid input. meta_props can not be empty.")

        if self._client.CLOUD_PLATFORM_SPACES and 'r_shiny' in meta_props:
            raise WMLClientError('Shiny is not supported in this release')

        if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
            base_model_id = meta_props.get(self.ConfigurationMetaNames.BASE_MODEL_ID)

            ModelTypes = _check_if_import_from_watsonx_ai(self.__module__,
                                                          'ibm_watson_machine_learning.foundation_models.utils.enums',
                                                          'ModelTypes')

            if isinstance(base_model_id, ModelTypes):
                meta_props[self.ConfigurationMetaNames.BASE_MODEL_ID] = base_model_id.value

        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)

        if 'serving_name' in str(metaProps) and meta_props.get('serving_name', False) and 'r_shiny' in str(metaProps):
            if 'parameters' in metaProps['r_shiny']:
                metaProps['r_shiny']['parameters']['serving_name'] = meta_props['serving_name']
            else:
                metaProps['r_shiny']['parameters'] = {'serving_name': meta_props['serving_name']}
            if 'online' in metaProps:
                del metaProps['online']

        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'wml_instance_id' in meta_props:
            metaProps.update({'wml_instance_id': meta_props['wml_instance_id']})

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            ## Remove new meataprop from older one, just in case if user provides.
            if self.ConfigurationMetaNames.HARDWARE_SPEC in metaProps:
                metaProps.pop(self.ConfigurationMetaNames.HARDWARE_SPEC)
            if self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS in metaProps:
                metaProps.pop(self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS)
            if self.ConfigurationMetaNames.R_SHINY in metaProps:
                metaProps.pop(self.ConfigurationMetaNames.R_SHINY)

            artifact_details = self._client.repository.get_details(artifact_uid)
            artifact_href = artifact_details['metadata']['href']
            artifact_revision_id = artifact_href.split('=')[1]

            if 'model' in artifact_href:
                if self._client.CAMS:
                    details = "/v4/models/"+artifact_uid + "?space_id=" + self._client.default_space_id
                else:
                    details = "/v4/models/" + artifact_uid  + "?rev=" + artifact_revision_id
            elif 'function' in artifact_href:
                if self._client.CAMS:
                    details = "/v4/functions/" + artifact_uid + "?space_id=" + self._client.default_space_id
                else:
                    details = "/v4/functions/" + artifact_uid  + "?rev=" + artifact_revision_id
            else:
                raise WMLClientError('Unexpected artifact type: \'{}\'.'.format(artifact_uid))
            if 'model' in artifact_href:
                if "tensorflow_1.11" in artifact_details["entity"]["type"] or "tensorflow_1.5" in \
                        artifact_details["entity"]["type"]:
                    print("Note: Model of framework tensorflow and versions 1.5/1.11 has been deprecated. These versions will not be supported after 26th Nov 2019.")

            metaProps['asset'] = {'href': details}
            if self._client.CAMS:
                if self._client.default_space_id is not None:
                    metaProps['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
                else:
                    raise WMLClientError(
                        "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

        ##Check if default space is set
        else:
            metaProps['asset'] = metaProps.get('asset') if metaProps.get('asset') else {'id': artifact_uid}
            if rev_id is not None:
                metaProps['asset'].update({'rev': rev_id})

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                if self._client.default_project_id:
                    metaProps['project_id'] = self._client.default_project_id
                else:
                    metaProps['space_id'] = self._client.default_space_id
            else:
                if 'space' not in metaProps:
                    if self._client.default_space_id is not None:
                        metaProps['space_id'] = self._client.default_space_id
                    else:
                        raise WMLClientError(
                            "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")
            # note: checking if artifact_uid points to prompt_template
            if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
                with all_logging_disabled():
                    try:
                        PromptTemplateManager = _check_if_import_from_watsonx_ai(self.__module__,
                                                                                 'ibm_watson_machine_learning.foundation_models.prompts',
                                                                                 'PromptTemplateManager')
                        model_id = PromptTemplateManager(api_client=self._client).load_prompt(artifact_uid).model_id
                    except Exception:
                        pass  # Foundation models scenario should not impact other ML models' deployment scenario.
                    else:
                        metaProps.pop("asset")
                        metaProps['prompt_template'] = {"id": artifact_uid}
                        if DeploymentNewMetaNames.BASE_MODEL_ID not in metaProps and \
                                DeploymentNewMetaNames.BASE_DEPLOYMENT_ID not in metaProps:
                            metaProps.update({DeploymentNewMetaNames.BASE_MODEL_ID: model_id})
            # --- end note

        url = self._client.service_instance._href_definitions.get_deployments_href()

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            response = requests.post(
                url,
                json=metaProps,
                params=self._client._params(),  # version is mandatory
                headers=self._client._get_headers())
        else:
            response = requests.post(
                url,
                json=metaProps,
                headers=self._client._get_headers())

        ## Post Deployment call executed
        if response.status_code == 202:
            deployment_details = response.json()
            #print(deployment_details)

            if self._ICP:
                if 'online_url' in deployment_details["entity"]["status"]:
                 scoringUrl = deployment_details.get(u'entity').get(u'status').get('online_url').get('url').replace('https://ibm-nginx-svc:443', predictionUrl)
                 deployment_details[u'entity'][u'status']['online_url']['url'] = scoringUrl

            deployment_uid = self.get_uid(deployment_details)

            import time
            print_text_header_h1(u'Synchronous deployment creation for uid: \'{}\' started'.format(artifact_uid))

            status = deployment_details[u'entity'][u'status']['state']

            notifications = []

            with StatusLogger(status) as status_logger:
                while True:
                    time.sleep(5)
                    deployment_details = self._client.deployments.get_details(deployment_uid, _silent=True)
                    #this is wrong , needs to update for ICP
                    if "system" in deployment_details:
                        notification = deployment_details['system']['warnings'][0]['message']
                        if notification not in notifications:
                            print("\nNote: " + notification)
                            notifications.append(notification)
                    if self._ICP and not self._client.ICP_PLATFORM_SPACES:
                        scoringUrl = deployment_details.get(u'entity').get(u'asset').get('href').replace('https://wml-os-envoy:16600', predictionUrl)
                        deployment_details[u'entity'][u'asset']['href'] = scoringUrl
                    status = deployment_details[u'entity'][u'status'][u'state']
                    status_logger.log_state(status)
                    if status != u'DEPLOY_IN_PROGRESS' and status != "initializing":
                        break
            if status == u'DEPLOY_SUCCESS' or status == u'ready':
                print(u'')
                print_text_header_h2(
                    u'Successfully finished deployment creation, deployment_uid=\'{}\''.format(deployment_uid))
                return deployment_details
            else:
                print_text_header_h2(u'Deployment creation failed')
                self._deployment_status_errors_handling(deployment_details, 'creation', deployment_uid)
        else:
            error_msg = u'Deployment creation failed'
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(error_msg + '. Error: ' + str(response.status_code) + '. ' + reason)

            #return self._handle_response(202, u'created deployment', response)

    @staticmethod
    def get_uid(deployment_details):
        """Get deployment_uid from deployment details.

        *Deprecated:* Use ``get_id(deployment_details)`` instead.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment UID that is used to manage the deployment
        :rtype: str

        **Example**

        .. code-block:: python

            deployment_uid = client.deployments.get_uid(deployment)

        """
        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)

        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)

        try:
            if 'id' in deployment_details[u'metadata']:
                uid = deployment_details.get(u'metadata').get(u'id')
            else:
                uid = deployment_details.get(u'metadata').get(u'guid')
        except Exception as e:
            raise WMLClientError(u'Getting deployment UID from deployment details failed.', e)

        if uid is None:
            raise MissingValue(u'deployment_details.metadata.guid')

        return uid

    @staticmethod
    def get_id(deployment_details):
        """Get deployment id from deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment ID that is used to manage the deployment
        :rtype: str

        **Example**

        .. code-block:: python

            deployment_id = client.deployments.get_id(deployment)

        """

        return Deployments.get_uid(deployment_details)

    @staticmethod
    def get_href(deployment_details):
        """Get deployment_href from deployment details.

        :param deployment_details: metadata of the deployment.
        :type deployment_details: dict

        :return: deployment href that is used to manage the deployment
        :rtype: str

        **Example**

        .. code-block:: python

            deployment_href = client.deployments.get_href(deployment)

        """
        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)
        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)

        try:
            if 'href' in deployment_details[u'metadata']:
                url = deployment_details.get(u'metadata').get(u'href')
            else:
                url = "/ml/v4/deployments/{}".format(deployment_details[u'metadata'][u'id'])
        except Exception as e:
            raise WMLClientError(u'Getting deployment url from deployment details failed.', e)

        if url is None:
            raise MissingValue(u'deployment_details.metadata.href')

        return url

    def _get_serving_name_info(self, serving_name: str) -> tuple:
        """Get info about serving name.

        :param serving_name: serving name to filter deployments
        :type serving_name: str

        :return: information about serving name: (<status_code>, <response json if any>)
        :rtype: tuple

        **Example**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available('test')

        """
        params = {
            'serving_name': serving_name,
            'conflict': 'true',
            'version': self._client.version_param
        }

        url = self._client.service_instance._href_definitions.get_deployments_href()
        res = requests.get(url,
                           headers=self._client._get_headers(),
                           params=params)

        if res.status_code == 409:
            response = res.json()
        else:
            response = None

        return (res.status_code, response)

    def is_serving_name_available(self, serving_name: str) -> bool:
        """Check if serving name is available for usage.

        :param serving_name: serving name to filter deployments
        :type serving_name: str

        :return: information if serving name is available
        :rtype: bool

        **Example**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available('test')

        """
        status_code, _ = self._get_serving_name_info(serving_name)

        return status_code != 409

    def get_details(self, deployment_uid=None, serving_name=None, limit=None, asynchronous=False, get_all=False,
                    spec_state=None, _silent=False):
        """Get information about deployment(s).
        If deployment_uid is not passed, all deployment details are fetched.

        :param deployment_uid: Unique Id of Deployment
        :type deployment_uid: str, optional

        :param serving_name: serving name to filter deployments
        :type serving_name: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param asynchronous:  if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `deployment_uid` is None
        :type spec_state: SpecStates, optional

        :return: metadata of deployment(s)
        :rtype: dict (if deployment_uid is not None) or {"resources": [dict]} (if deployment_uid is None)

        **Example**

        .. code-block:: python

            deployment_details = client.deployments.get_details(deployment_uid)
            deployment_details = client.deployments.get_details(deployment_uid=deployment_uid)
            deployments_details = client.deployments.get_details()
            deployments_details = client.deployments.get_details(limit=100)
            deployments_details = client.deployments.get_details(limit=100, get_all=True)
            deployments_details = []
            for entry in client.deployments.get_details(limit=100, asynchronous=True, get_all=True):
                deployments_details.extend(entry)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_uid, u'deployment_uid', str, False)

        if deployment_uid is not None and not is_uid(deployment_uid):
            raise WMLClientError(u'\'deployment_uid\' is not an uid: \'{}\''.format(deployment_uid))

        url = self._client.service_instance._href_definitions.get_deployments_href()

        query_params = self._client._params()

        if serving_name:
            query_params['serving_name'] = serving_name

        if deployment_uid is None:
            filter_func = self._get_filter_func_by_spec_state(spec_state) if spec_state else None

            deployment_details = self._get_artifact_details(url, deployment_uid, limit, 'deployments',
                                                            query_params=query_params, _async=asynchronous,
                                                            _all=get_all,
                                                            _filter_func=filter_func)
        else:
            deployment_details = self._get_artifact_details(url, deployment_uid, limit, 'deployments',
                                                            query_params=query_params)

        if type(deployment_details).__name__ != 'NextResourceGenerator' and "system" in deployment_details \
                and not _silent:
            print("Note: " + deployment_details['system']['warnings'][0]['message'])

        return deployment_details

    @staticmethod
    def get_scoring_href(deployment_details):
        """Get scoring url from deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: scoring endpoint url that is used for making scoring requests
        :rtype: str

        **Example**

        .. code-block:: python

            scoring_href = client.deployments.get_scoring_href(deployment)

        """

        Deployments._validate_type(deployment_details, u'deployment', dict, True)

        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)
        scoring_url = None
        try:
            url = deployment_details['entity']['status'].get('online_url')
            if url is not None:
                scoring_url = deployment_details['entity']['status']['online_url']['url']
            else:
                raise MissingValue(u'Getting scoring url for deployment failed. This functionality is  available only for sync deployments')

        except Exception as e:
            raise WMLClientError(u'Getting scoring url for deployment failed. This functionality is  available only for sync deployments', e)

        if scoring_url is None:
            raise MissingValue(u'scoring_url missing in online_predictions')
        return scoring_url

    @staticmethod
    def get_serving_href(deployment_details):
        """Get serving url from deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: serving endpoint url that is used for making scoring requests
        :rtype: str

        **Example**

        .. code-block:: python

            scoring_href = client.deployments.get_serving_href(deployment)

        """

        Deployments._validate_type(deployment_details, u'deployment', dict, True)

        if not Deployments.cloud_platform_spaces and not Deployments.icp_platform_spaces:
            Deployments._validate_type_of_details(deployment_details, DEPLOYMENT_DETAILS_TYPE)

        try:
            serving_name = deployment_details['entity']['online'].get('parameters').get('serving_name')
            serving_url = [url for url in deployment_details['entity'].get('status').get('serving_urls') if serving_name == url.split('/')[-2]][0]

            if serving_url:
                return serving_url
            else:
                raise MissingValue(
                    u'Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name.')

        except Exception as e:
            raise WMLClientError(
                u'Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name.',
                e)

    def delete(self, deployment_uid):
        """Delete deployment.

        :param deployment_uid: Unique Id of Deployment
        :type deployment_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.deployments.delete(deployment_uid)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_uid, u'deployment_uid', str, True)

        if deployment_uid is not None and not is_uid(deployment_uid):
            raise WMLClientError(u'\'deployment_uid\' is not an uid: \'{}\''.format(deployment_uid))

        deployment_url = self._client.service_instance._href_definitions.get_deployment_href(deployment_uid)

        if not self._ICP or self._client.ICP_PLATFORM_SPACES:
            response_delete = requests.delete(
                deployment_url,
                params=self._client._params(),
                headers=self._client._get_headers())
        else:
            response_delete = requests.delete(
                deployment_url,
                #TODO:  The below line needs to uncommeted after /v4/deployment accespts query param
               # params=self._client._params(),
                headers=self._client._get_headers())

        return self._handle_response(204, u'deployment deletion', response_delete, False)

    def score(self, deployment_id, meta_props, transaction_id=None):
        """Make scoring requests against deployed artifact.

        :param deployment_id: Unique Id of the deployment to be scored
        :type deployment_id: str

        :param meta_props: meta props for scoring, use ``client.deployments.ScoringMetaNames.show()`` to view the list of ScoringMetaNames
        :type meta_props: dict

        :param transaction_id: transaction id to be passed with records during payload logging
        :type transaction_id: str, optional

        :return: scoring result containing prediction and probability
        :rtype: dict

        .. note::

                * *client.deployments.ScoringMetaNames.INPUT_DATA* is the only metaname valid for sync scoring.
                * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example**

        .. code-block:: python

            scoring_payload = {wml_client.deployments.ScoringMetaNames.INPUT_DATA:
                [{'fields':
                    ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                    'values': [
                        ['M',23,'Single','Student'],
                        ['M',55,'Single','Executive']
                    ]
                }]
            }
            predictions = client.deployments.score(deployment_id, scoring_payload)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, u'deployment_id', str, True)
        Deployments._validate_type(meta_props, u'meta_props', dict, True)

        _check_spec_uids(meta_props)

        if meta_props.get(self.ScoringMetaNames.INPUT_DATA) is None:
            raise WMLClientError("Scoring data input 'ScoringMetaNames.INPUT_DATA' is mandatory for synchronous "
                                 "scoring")

        scoring_data = meta_props[self.ScoringMetaNames.INPUT_DATA]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name='pandas')
                import pandas as pd

                scoring_values = each_score_request["values"]
                # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                if (isinstance(scoring_values, pd.DataFrame)):
                    scoring_values = scoring_values.where(pd.notnull(scoring_values), None)
                    fields_names = scoring_values.columns.values.tolist()
                    values = scoring_values.values.tolist()

                    try:
                        values[pd.isnull(values)] = None

                        # note: above code fails when there is no null values in a dataframe
                    except TypeError:
                        pass

                    each_score_request["values"] = values
                    if fields_names is not None:
                        each_score_request["fields"] = fields_names


                ## If payload is a numpy dataframe

                elif (isinstance(scoring_values, np.ndarray)):

                    values = scoring_values.tolist()
                    each_score_request["values"] = values


                # else:
                #     payload_score["values"] = each_score_request["values"]
                #     if "fields" in each_score_request:
                #         payload_score["fields"] = each_score_request["fields"]
                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

        payload = {}

        payload["input_data"] = score_payload

        headers = self._client._get_headers()

        if transaction_id is not None:
            headers.update({'x-global-transaction-id': transaction_id})

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            scoring_url = self._wml_credentials["url"] + "/ml/v4/deployments/" + deployment_id + "/predictions"
        else:
            scoring_url = self._wml_credentials["url"] + "/v4/deployments/" + deployment_id + "/predictions"

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
            del params[u'space_id']
            response_scoring = requests.post(
                scoring_url,
                json=payload,
                params=params,  # version parameter is mandatory
                headers=headers)
        else:
            response_scoring = requests.post(
                scoring_url,
                json=payload,
                headers=headers)

        return self._handle_response(200, u'scoring', response_scoring)

        #########################################

    def get_download_url(self, deployment_details):
        """Get deployment_download_url from deployment details.

        :param deployment_details: created deployment details
        :type deployment_details: dict

        :return: deployment download URL that is used to get file deployment (for example: Core ML)
        :rtype: str

        **Example**

        .. code-block:: python

            deployment_url = client.deployments.get_download_url(deployment)

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Downloading virtual deployment is no longer supported in Cloud Pak for Data, versions 3.5 and later.')

        if self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(u'Downloading virtual deployment is no longer supported in Cloud Pak for Data as a Service.')

        Deployments._validate_type(deployment_details, u'deployment_details', dict, True)
        try:
            virtual_deployment_detaails = deployment_details.get(u'entity').get(u'status').get(u'virtual_deployment_downloads')
            if virtual_deployment_detaails is not None:
                url = virtual_deployment_detaails[0].get(u'url')
            else:
                url = None
        except Exception as e:
            raise WMLClientError(u'Getting download url from deployment details failed.', e)

        if url is None:
            raise MissingValue(u'deployment_details.entity.virtual_deployment_downloads.url')

        return url

    def list(self, limit=None, return_as_df=True, artifact_type=None):
        """Print deployments in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :param artifact_type: return only deployments with the specified artifact_type
        :type artifact_type: str, optional

        :return: pandas.DataFrame with listed deployments or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.deployments.list()

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        details = self.get_details(limit=limit)
        resources = details[u'resources']

        values = []
        index = 0

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            sw_spec_info = {s['id']: s
                            for s in self._client.software_specifications.get_details(state_info=True)['resources']}

            def enrich_asset_with_type(asset_details, asset_type):
                if asset_type:
                    asset_details['metadata']['asset_type'] = asset_type

                return asset_details

            asset_info = {el['metadata']['id']: enrich_asset_with_type(el, asset_type)
                          for asset_type, resources in {'model': self._client._models.get_details(get_all=True),
                                                        'function': self._client._functions.get_details(get_all=True)
                                                        }.items()
                          for el in resources['resources']}

            def get_spec_info(spec_id, prop):
                if spec_id and spec_id in sw_spec_info:
                    return sw_spec_info[spec_id].get(prop, '')
                else:
                    return ''

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            # Ideally this needs to be on the server side
            if limit is not None and index == limit:
                break

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                if 'guid' in m[u'metadata']:
                    values.append((m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at'], self._get_deployable_asset_type(m)))
                else:
                    values.append((m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'entity'][u'status']['state'],
                                  m[u'metadata'][u'created_at'], self._get_deployable_asset_type(m)))
            else:
                spec_id = asset_info.get(m['entity'].get('asset', m['entity'].get('prompt_template'))['id'], {})\
                    .get('entity', {}).get('software_spec', {}).get('id')

                if artifact_type and m['entity'].get('deployed_asset_type', 'unknown') != artifact_type:
                    pass # filter by artifact_type
                else:
                    values.append(
                        (m['metadata']['guid'] if 'guid' in m['metadata'] else m['metadata']['id'],
                         m['entity']['name'],
                         m['entity']['status']['state'],
                         m['metadata']['created_at'],
                         m['entity'].get('deployed_asset_type', 'unknown'),
                         get_spec_info(spec_id, 'state'),
                         get_spec_info(spec_id, 'replacement')))

            index = index + 1

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            table = self._list(values, [u'GUID', u'NAME', u'STATE', u'CREATED', u'ARTIFACT_TYPE'], limit, 50)
        else:
            table = self._list(values,
                               ['GUID', 'NAME', 'STATE', 'CREATED', 'ARTIFACT_TYPE', 'SPEC_STATE', 'SPEC_REPLACEMENT'],
                               limit, 50)

        if return_as_df:
            return table

    def list_jobs(self, limit=None,  return_as_df=True):
        """Print the async deployment jobs in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed deployment jobs or None
        :rtype: pandas.DataFrame or None if return_as_df is False

        .. note::

            This method list only async deployment jobs created for WML deployment.

        **Example**

        .. code-block:: python

            client.deployments.list_jobs()

        """

        details = self.get_job_details(limit=limit)
        resources = details[u'resources']
        values = []
        index = 0

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            if limit is not None and index == limit:
                break

            if 'scoring' in m['entity']:
                state = m['entity']['scoring']['status']['state']
            else:
                state = m['entity']['decision_optimization']['status']['state']

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                deploy_id = m['entity']['deployment']['id']
                values.append((m[u'metadata'][u'id'], state, m[u'metadata'][u'created_at'], deploy_id))
            else:
                deploy_id = m['entity']['deployment']['href'].split("/")[3]
                values.append((m[u'metadata'][u'guid'], state, m[u'metadata'][u'created_at'], deploy_id))

            index = index + 1

        table = self._list(values, ['JOB-UID', 'STATE', 'CREATED', 'DEPLOYMENT-ID'], limit, 50)
        if return_as_df:
            return table

    def _get_deployable_asset_type(self, details):
        url = details[u'entity'][u'asset']['id']
        if 'model' in url:
            return 'model'
        elif 'function' in url:
            return 'function'
        else:
            return 'unknown'

    def update(self, deployment_uid, changes):
        """Updates existing deployment metadata. If ASSET is patched, then 'id' field is mandatory
        and it starts a deployment with the provided asset id/rev. Deployment id remains the same.

        :param deployment_uid: Unique Id of deployment which  should be updated
        :type deployment_uid: str

        :param changes:  elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated deployment
        :rtype: dict

        **Examples**

        .. code-block:: python

            metadata = {client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment"}
            updated_deployment_details = client.deployments.update(deployment_uid, changes=metadata)

            metadata = {client.deployments.ConfigurationMetaNames.ASSET: {  "id": "ca0cd864-4582-4732-b365-3165598dc945",
                                                                            "rev":"2" }}
            deployment_details = client.deployments.update(deployment_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        ret202 = False

        ## In case of passing 'AUTO_ONLINE_DEPLOYMENT' as true, we need to poll for deployment to be either 'deploy_success' or 'update_success'.

        Deployments._validate_type(deployment_uid, 'deployment_uid', str, True)

        if ('asset' in changes and not changes[u'asset']) and \
            ('prompt_template' in changes and not changes[u'prompt_template']):
            msg = "ASSET/PROMPT_TEMPLATE cannot be empty. 'id' and 'rev' (only ASSET) fields are supported. 'id' is mandatory"
            print(msg)
            raise WMLClientError(msg)
        

        # if changes.get('asset') is not None and (changes.get('name') is not None or changes.get('description') is not None):
        patch_job = changes.get('asset') is not None or self.ConfigurationMetaNames.PROMPT_TEMPLATE in changes or \
                    self.ConfigurationMetaNames.SERVING_NAME in changes or self.ConfigurationMetaNames.OWNER in changes

        patch_job_field = None
        if patch_job:
            if changes.get('asset') is not None:
                patch_job_field = "ASSET"
            elif self.ConfigurationMetaNames.PROMPT_TEMPLATE in changes:
                patch_job_field = "PROMPT_TEMPLATE"
            elif self.ConfigurationMetaNames.SERVING_NAME in changes:
                patch_job_field = "SERVING_NAME"
            elif self.ConfigurationMetaNames.OWNER in changes:
                patch_job_field = "OWNER"

            if patch_job_field is None:
                raise WMLClientError("Unexpected patch job element.")

        if patch_job and (len(changes) > 1):
            msg = f"When {patch_job_field} is being updated/patched, other fields cannot be updated. If other fields are to be " \
                  f"updated, try without {patch_job_field} update. {patch_job_field} update triggers deployment with the new asset retaining " \
                  "the same deployment_id"
            print(msg)
            raise WMLClientError(msg)

        deployment_details = self.get_details(deployment_uid)
        serving_name_change = False
        new_serving_name = None
        if self.ConfigurationMetaNames.SERVING_NAME in changes:
            new_serving_name = changes.pop(self.ConfigurationMetaNames.SERVING_NAME)
            serving_name_change = True

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(deployment_details, changes, with_validation=True)

        if serving_name_change:
            replace = 'serving_name' in deployment_details['entity'].get('online').get('parameters', [])
            patch_payload.append({
                "op": "replace" if replace else "add",
                "path": "/online/parameters",
                "value": {"serving_name": new_serving_name}})

        ## As auto_online_deployment and auto_redeploy values are passed as 'bool' but service needs them in 'str' format to patch.
        for ele in patch_payload:
            if 'auto_online_deployment' in ele['path'] or 'auto_redeploy' in ele['path']:
                ele['value'] = str(ele['value']).lower()

        url = self._client.service_instance._href_definitions.get_deployment_href(deployment_uid)

        response = requests.patch(url, json=patch_payload, params=self._client._params(), headers=self._client._get_headers())

        if patch_job and response.status_code == 202:
            updated_details = self._handle_response(202, u'deployment asset patch', response)

            ret202 = True

            print(f"Since {patch_job_field} is patched, deployment with new asset id/rev is being started. " \
                  "Monitor the status using deployments.get_details(deployment_uid) api")
        elif response.status_code == 202:
            updated_details = self._handle_response(202, u'deployment scaling', response)
            ret202 = True
        else:
            updated_details = self._handle_response(200, u'deployment patch', response)

        if('auto_online_deployment' in changes):
           if response is not None:
               if response.status_code == 200:
                   deployment_details = self.get_details(deployment_uid)

                   import time
                   print_text_header_h1(u' deployment update for uid: \'{}\' started'.format(deployment_uid))

                   status = deployment_details[u'entity'][u'status'][u'state']

                   with StatusLogger(status) as status_logger:
                       while True:
                           time.sleep(5)
                           deployment_details = self.get_details(deployment_uid)
                           status = deployment_details[u'entity'][u'status'][u'state']
                           status_logger.log_state(status)

                           if status != u'DEPLOY_IN_PROGRESS' and status != u'UPDATE_IN_PROGRESS':
                               break

                   if status == u'DEPLOY_SUCCESS' or status == u'UPDATE_SUCCESS':
                       print(u'')
                       print_text_header_h2(
                           u'Successfully finished deployment update, deployment_uid=\'{}\''.format(deployment_uid))
                       return deployment_details
                   else:
                       print_text_header_h2(u'Deployment update failed')
                       self._deployment_status_errors_handling(deployment_details, 'update', deployment_uid)
               else:
                   error_msg = u'Deployment update failed'
                   reason = response.text
                   print(reason)
                   print_text_header_h2(error_msg)
                   raise WMLClientError(error_msg + u'. Error: ' + str(response.status_code) + '. ' + reason)

        if not ret202:
            return updated_details

    ## Below functions are for async scoring. They are just dummy functions.
    def _score_async(self,deployment_uid, scoring_payload,
                     transaction_id=None,
                     retention=None):

        Deployments._validate_type(deployment_uid, u'deployment_id', str, True)
        Deployments._validate_type(scoring_payload, u'scoring_payload', dict, True)
        headers = self._client._get_headers()

        if transaction_id is not None:
            headers.update({'x-global-transaction-id': transaction_id})
        # making change - connection keep alive
        scoring_url = self._client.service_instance._href_definitions.get_async_deployment_job_href()
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = None

        if not self._client.ICP and retention is not None:
            if not isinstance(retention, int) or retention < -1:
                raise TypeError("`retention` takes integer values greater or equal than -1.")
            params.update({'retention': retention})

        response_scoring = requests.post(
            scoring_url,
            params=params,
            json=scoring_payload,
            headers=headers)

        return self._handle_response(202, u'scoring asynchronously', response_scoring)

    def create_job(self, deployment_id, meta_props, retention=None, transaction_id=None, _asset_id=None):
        """Create an asynchronous deployment job.

        :param deployment_id:  Unique Id of Deployment
        :type deployment_id: str

        :param meta_props: metaprops. To see the available list of metanames
            use ``client.deployments.ScoringMetaNames.get()``
            or ``client.deployments.DecisionOptimizationmetaNames.get()``

        :type meta_props: dict

        :param retention: how many job days job meta should be retained,
            takes integer values >= -1, supported only on Cloud
        :type retention: int, optional

        :return: metadata of the created async deployment job
        :rtype: dict

        .. note::

            * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example**

        .. code-block:: python

            scoring_payload = {wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{'fields': ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                                                                                     'values': [['M',23,'Single','Student'],
                                                                                                ['M',55,'Single','Executive']]}]}
            async_job = client.deployments.create_job(deployment_id, scoring_payload)

         """

        WMLResource._chk_and_block_create_update_for_python36(self)
        Deployments._validate_type(deployment_id, u'deployment_uid', str, True)
        Deployments._validate_type(meta_props, u'meta_props', dict, True)

        _check_spec_uids(meta_props)

        if _asset_id:
            Deployments._validate_type(_asset_id, u'_asset_id', str, True)
            # We assume that _asset_id is the id of the asset that was deployed
            # in the deployment with id deployment_id, and we save one REST call
            asset = _asset_id
        else:
            deployment_details = self.get_details(deployment_id)
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                asset = deployment_details["entity"]["asset"]['id']
            else:
                asset = deployment_details["entity"]["asset"]["href"].split("/")[-1]

        do_model = False
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            asset_details = self._client.data_assets.get_details(asset)
            if 'wml_model' in asset_details["entity"] and 'type' in asset_details['entity']['wml_model']:
                if 'do' in asset_details['entity']['wml_model']['type']:
                    do_model = True
        else:
            asset_details = self._client.repository.get_details(asset)
            if "type" in asset_details["entity"]:
                if "do" in asset_details["entity"]["type"]:
                    do_model = True

        flag = 0 ## To see if it is async scoring or DecisionOptimization Job
        if do_model:
            payload = self.DecisionOptimizationMetaNames._generate_resource_metadata(meta_props,
                                                                                     with_validation=True,
                                                                                     client=self._client)
            flag = 1
        else:
            payload = self.ScoringMetaNames._generate_resource_metadata(meta_props, with_validation=True,
                                                                        client=self._client)

        scoring_data = None
        if "scoring" in payload and "input_data" in payload["scoring"]:
                scoring_data = payload["scoring"]["input_data"]

        if "decision_optimization" in payload and "input_data" in payload["decision_optimization"]:
                scoring_data = payload["decision_optimization"]["input_data"]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name='pandas')
                import pandas as pd
                if "values" in each_score_request:
                    scoring_values = each_score_request["values"]
                    # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                    if (isinstance(scoring_values, pd.DataFrame)):
                        fields_names = scoring_values.columns.values.tolist()
                        values = scoring_values.where(pd.notnull(scoring_values), None).values.tolist() #replace nan with None

                        each_score_request["values"] = values
                        if fields_names is not None:
                            each_score_request["fields"] = fields_names


                    ## If payload is a numpy dataframe

                    elif (isinstance(scoring_values, np.ndarray)):

                        values = np.where(pd.notnull(scoring_values), scoring_values, None).tolist() #replace nan with None
                        each_score_request["values"] = values


                # else:
                #     payload_score["values"] = each_score_request["values"]
                #     if "fields" in each_score_request:
                #         payload_score["fields"] = each_score_request["fields"]
                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

            if flag == 0:
                payload["scoring"]["input_data"] = score_payload
            if flag == 1:
                payload["decision_optimization"]["input_data"] = score_payload

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            import copy
            if 'input_data_references' in meta_props:
                Deployments._validate_type(meta_props.get('input_data_references'), u'input_data_references', list,
                                           True)
                modified_input_data_references=False
                input_data = copy.deepcopy(meta_props.get('input_data_references'))
                for i, input_data_fields in enumerate(input_data):
                    if 'connection' not in input_data_fields:
                        modified_input_data_references=True
                        input_data_fields.update({'connection': {}})
                if modified_input_data_references:
                    if 'scoring' in payload:
                        payload['scoring'].update({'input_data_references': input_data})
                    else:
                        payload['decision_optimization'].update({'input_data_references': input_data})

            if 'output_data_reference' in meta_props:
                Deployments._validate_type(meta_props.get('output_data_reference'), u'output_data_reference', dict,
                                           True)

                output_data = copy.deepcopy(meta_props.get('output_data_reference'))
                if 'connection' not in output_data:  # and output_data.get('connection', None) is not None:
                    output_data.update({'connection': {}})
                    payload['scoring'].update({'output_data_reference': output_data})

            if 'output_data_references' in meta_props:
                Deployments._validate_type(meta_props.get('output_data_references'), u'output_data_references', list, True)
                output_data = copy.deepcopy(meta_props.get('output_data_references'))
                modified_output_data_references = False
                for i, output_data_fields in enumerate(output_data):
                    if 'connection' not in output_data_fields:
                        modified_output_data_references = True
                        output_data_fields.update({'connection': {}})
                if modified_output_data_references and 'decision_optimization' in payload:
                    payload['decision_optimization'].update({'output_data_references': output_data})

            payload.update({"deployment": {"id": deployment_id}})
            if 'hardware_spec' in meta_props:
                payload.update({"hardware_spec": meta_props[self.ConfigurationMetaNames.HARDWARE_SPEC]})
            if 'hybrid_pipeline_hardware_specs' in meta_props:
                payload.update({"hybrid_pipeline_hardware_specs": meta_props[self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS]})

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                payload.update({'space_id': self._client.default_space_id})

                if 'name' not in payload:
                    import uuid
                    payload.update({'name': 'name_' + str(uuid.uuid4())})
        else:
            payload.update({"deployment": {"href": "/v4/deployments/" + deployment_id}})

        return self._score_async(deployment_id, payload,
                                 transaction_id=transaction_id,
                                 retention=retention)

    def get_job_details(self, job_uid=None, include=None, limit=None):
        """Get information about deployment job(s).
        If deployment job_uid is not passed, all deployment jobs details are fetched.

        :param job_uid: Unique Job ID
        :type job_uid: str, optional

        :param include: fields to be retrieved from 'decision_optimization'
            and 'scoring' section mentioned as value(s) (comma separated) as output response fields
        :type include: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :return: metadata of deployment job(s)
        :rtype: dict (if job_uid is not None) or {"resources": [dict]} (if job_uid is None)

        **Example**

        .. code-block:: python

            deployment_details = client.deployments.get_job_details()
            deployments_details = client.deployments.get_job_details(job_uid=job_uid)

        """
        if job_uid is not None:
            Deployments._validate_type(job_uid, u'job_uid', str, True)
        url = self._client.service_instance._href_definitions.get_async_deployment_job_href()

        params = self._client._params()
        if include:
            params['include'] = include

        return self._get_artifact_details(url, job_uid, limit,
                                          'async deployment job' if job_uid else 'async deployment jobs',
                                          query_params=params)

    def get_job_status(self, job_id):
        """Get the status of the deployment job.

        :param job_id: Unique Id of the deployment job
        :type job_id: str

        :return: status of the deployment job
        :rtype: dict

        **Example**

        .. code-block:: python

            job_status = client.deployments.get_job_status(job_uid)

        """

        job_details = self.get_job_details(job_id)

        if 'scoring' not in job_details['entity']:
            return job_details['entity']['decision_optimization']['status']
        return job_details['entity']['scoring']['status']

    def get_job_uid(self, job_details):
        """Get the Unique Id of the deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: Unique Id of the deployment job
        :rtype: str

        **Example**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_uid=job_uid)
            job_status = client.deployments.get_job_uid(job_details)

        """

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            return job_details['metadata']['id']
        else:
            return job_details['metadata']['guid']

    def get_job_href(self, job_details):
        """Get the href of the deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: href of the deployment job
        :rtype: str

        **Example**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_uid=job_uid)
            job_status = client.deployments.get_job_href(job_details)

        """
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            return "/ml/v4/deployment_jobs/{}".format(job_details[u'metadata'][u'id'])
        else:
            return job_details['metadata']['href']


    def delete_job(self, job_uid, hard_delete=False):
        """Cancels a deployment job that is currenlty running.  This method is also be used to delete metadata
        details of the completed or canceled jobs when hard_delete parameter is set to True.

        :param job_uid: Unique Id of deployment job which should be canceled
        :type job_uid: str

        :param hard_delete: specify `True` or `False`:

            `True` - To delete the completed or canceled job.

            `False` - To cancel the currently running deployment job.

        :type hard_delete: bool, optional


        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.deployments.delete_job(job_uid)

        """
        Deployments._validate_type(job_uid, u'job_uid', str, True)

        if job_uid is not None and not is_uid(job_uid):
            raise WMLClientError(u'\'job_uid\' is not an uid: \'{}\''.format(job_uid))

        url = self._client.service_instance._href_definitions.get_async_deployment_jobs_href(job_uid)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = {}

        if hard_delete is True:
            params.update({'hard_delete': u'true'})

        response_delete = requests.delete(
            url,
            headers=self._client._get_headers(),
            params=params)

        return self._handle_response(204, u'deployment async job deletion', response_delete, False)

    def _get_filter_func_by_spec_state(self, spec_state):
        def filter_func(resources):
            asset_ids = [i['metadata']['id']
                         for key, value
                         in {'model': self._client._models.get_details(get_all=True, spec_state=spec_state),
                             'function': self._client._functions.get_details(get_all=True, spec_state=spec_state)
                             }.items()
                         for i in value['resources']]

            return [r for r in resources if r['entity'].get('asset', {}).get('id') in asset_ids]

        return filter_func

    def _get_model_inference_text(self, deployment_id, inference_type: Literal["text", "text_stream"], params=None):
        """Based on provided deployment_id and params get ModelInference object.
        Verify that the deployment with the given deployment_id has generating methods.
        """
        # Import ModelInference here to avoid circular import error
        # Check from which package, wx or wml, is call
        ModelInference = _check_if_import_from_watsonx_ai(self.__module__,
                                                          'ibm_watson_machine_learning.foundation_models.inference.model_inference',
                                                          'ModelInference')
        match inference_type:
            case "text":
                generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_href(
                    deployment_id=deployment_id, item='text')
            case "text_stream":
                if self._client._use_fm_ga_api:
                    generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_stream_href(
                        deployment_id=deployment_id)
                else:  # Remove on CPD 5.0 release
                    generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_href(
                        deployment_id=deployment_id, item="text_stream")
            case _:
                raise InvalidValue(value_name="inference_type", reason=f"Available types: 'text', 'text_stream', got: {inference_type}.")


        inference_url_list = [url.get("url") for url in self.get_details(deployment_id, _silent=True)['entity'].get('status', {}).get("inference", {})]
        if not inference_url_list:
            inference_url_list = self.get_details(deployment_id, _silent=True)['entity'].get('status', {}).get('serving_urls', [])

        if generated_url not in inference_url_list:
            for inference_url in inference_url_list:  # Remove on CPD 5.0 release
                if "v1-beta/deployments" not in inference_url:  # Remove on CPD 5.0 release
                    raise WMLClientError(Messages.get_message(deployment_id, message_id="fm_deployment_has_not_inference_for_generation"))

        return ModelInference(deployment_id=deployment_id, params=params, api_client=self._client)

    def generate(self,
                 deployment_id,
                 prompt=None,
                 params=None,
                 guardrails=False,
                 guardrails_hap_params=None,
                 guardrails_pii_params=None,
                 concurrency_limit=10,
                 async_mode=False) -> dict:
        """Generate a raw response with `prompt` for given `deployment_id`.

        :param deployment_id: Id of deployment
        :type deployment_id: str

        :param prompt: prompt needed for text generation. If deployment_id points to Prompt Template asset then prompt argument must be None, defaults to None
        :type prompt: (str | None), optional

        :param params: meta props for text generation, use ``ibm_watson_machine_learning.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection 
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watson_machine_learning.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param concurrency_limit: number of requests that will be sent in parallel, max is 10
        :type concurrency_limit: int, optional

        :param async_mode: If True then yield results asynchronously (using generator). In this case both prompt and
                           generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool

        :return: scoring result containing generated content.
        :rtype: dict
        """        
        d_inference = self._get_model_inference_text(deployment_id, "text", params)
        return d_inference.generate(prompt=prompt,
                                    guardrails=guardrails,
                                    guardrails_hap_params=guardrails_hap_params,
                                    guardrails_pii_params=guardrails_pii_params,
                                    concurrency_limit=concurrency_limit,
                                    params=params,
                                    async_mode=async_mode)

    def generate_text(self,
                      deployment_id,
                      prompt=None,
                      params=None,
                      raw_response=False,
                      guardrails=False,
                      guardrails_hap_params=None,
                      guardrails_pii_params=None,
                      concurrency_limit=10) -> str:
        """Given the selected deployment (deployment_id), a text prompt as input, parameters and concurrency_limit,
        the selected inference will generate a completion text as generated_text response.

        :param deployment_id: Id of deployment
        :type deployment_id: str

        :param prompt: the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: (str | None), optional

        :param params: meta props for text generation, use ``ibm_watson_machine_learning.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict

        :param raw_response: return the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection 
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watson_machine_learning.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param concurrency_limit: number of requests that will be sent in parallel, max is 10
        :type concurrency_limit: int

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python
            
                import warnings
                from ibm_watson_machine_learning.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """        
        d_inference = self._get_model_inference_text(deployment_id, "text", params)
        return d_inference.generate_text(prompt=prompt,
                                         raw_response=raw_response,
                                         guardrails=guardrails,
                                         guardrails_hap_params=guardrails_hap_params,
                                         guardrails_pii_params=guardrails_pii_params,
                                         concurrency_limit=concurrency_limit,
                                         params=params)

    def generate_text_stream(self,
                             deployment_id,
                             prompt=None,
                             params=None,
                             raw_response=False,
                             guardrails=False,
                             guardrails_hap_params=None,
                             guardrails_pii_params=None):
        """Given the selected deployment (deployment_id), a text prompt as input and parameters,
        the selected inference will generate a streamed text as generate_text_stream.

        :param deployment_id: Id of deployment
        :type deployment_id: str

        :param prompt: the prompt string, defaults to None
        :type prompt: (str | None), optional

        :param params: meta props for text generation, use ``ibm_watson_machine_learning.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dictl
        
        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection 
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watson_machine_learning.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python
            
                import warnings
                from ibm_watson_machine_learning.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """        
        d_inference = self._get_model_inference_text(deployment_id, "text_stream", params)
        return d_inference.generate_text_stream(prompt=prompt,
                                                params=params,
                                                raw_response=raw_response,
                                                guardrails=guardrails,
                                                guardrails_hap_params=guardrails_hap_params,
                                                guardrails_pii_params=guardrails_pii_params)
