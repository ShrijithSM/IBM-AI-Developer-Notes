#!/usr/bin/env python3

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import importlib.util
import json
import logging
import os
import platform
import sys
from pathlib import Path

import ibm_watson_machine_learning._wrappers.requests as requests
import requests as req
from ibm_watson_machine_learning.utils.utils import get_module_version
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource


logger = logging.getLogger(__name__)

CRYPTO_LIBRARY = "pyhelayers"

def is_crypto_supported() -> bool:
    # pyhelayers is used for client side encryption
    # Test is pyhelayers is installed, and if so we assume crypto is supported

    pyhelayers_spec = importlib.util.find_spec(CRYPTO_LIBRARY)

    return pyhelayers_spec is not None

def import_diff(module_file_path):

    if "runtime-23-1" in module_file_path:
        return
    pathlist = Path(module_file_path).rglob('*.py')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        if not path_in_str.endswith("__init__.py"):
            module_name = ("ibmfl" + path_in_str.split("ibmfl")[2].replace(os.sep, "."))[:-3]
            module_spec = importlib.util.spec_from_file_location(
                module_name, path_in_str)
            loader = importlib.util.LazyLoader(module_spec.loader)
            module_spec.loader = loader
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)


def check_python_framework_version(client_reqs):
    client_system = platform.system()
    client_processor = platform.processor()

    if 'fl_extras' in client_reqs:
        install_msg = "  You can install this and other required packages by running pip install --upgrade 'ibm-watson-machine-learning[" + client_reqs['fl_extras'] + "]'"
    else:
        install_msg = "  See documentation for more information."

    py_version = platform.python_version()
    logger.info("Detected Client Python Version: {}".format(py_version))
    if not py_version.startswith(client_reqs['py_version']):
            raise Exception("The selected software spec requires python=={}.".format(client_reqs['py_version']))

    if client_system == "Darwin" and client_processor == "arm":
        try:
            tensorflowmacos_version = get_module_version(lib_name='tensorflow-macos')
            logger.info("Detected tensorflow-macos Version: {}".format(tensorflowmacos_version))
            if not tensorflowmacos_version.startswith(client_reqs['tensorflow_version']):
                raise Exception("Incompatible tensorflow-macos version found")
        except Exception as ex:
            logger.warning("{}, this may cause unexpected errors.".format(ex))
            logger.warning("The selected software spec requires tensorflow-macos=={}.".format(client_reqs['tensorflow_version']))
    else:
        try:
            tensorflow_version = get_module_version(lib_name='tensorflow')
            logger.info("Detected tensorflow Version: {}".format(tensorflow_version))
            if not tensorflow_version.startswith(client_reqs['tensorflow_version']):
                raise Exception("Incompatible tensorflow version found")
        except Exception as ex:
            logger.warning("{}, this may cause unexpected errors.".format(ex))
            logger.warning("The selected software spec requires tensorflow=={}.".format(client_reqs['tensorflow_version']) + install_msg)

    try:
        torch_version = get_module_version(lib_name='torch')
        logger.info("Detected torch Version: {}".format(torch_version))
        if not torch_version.startswith(client_reqs['torch_version']):
            raise Exception("Incompatible torch version found")
    except Exception as ex:
        logger.warning("{}, this may cause unexpected errors.".format(ex))
        logger.warning("The selected software spec requires torch=={}.".format(client_reqs['torch_version']) + install_msg)

    try:
        scikitlearn_version = get_module_version(lib_name='scikit-learn')
        logger.info("Detected scikit-learn Version: {}".format(scikitlearn_version))
        if not scikitlearn_version.startswith(client_reqs['scikitlearn_version']):
            raise Exception("Incompatible scikit-learn version found")
    except Exception as ex:
        logger.warning("{}, this may cause unexpected errors.".format(ex))
        logger.warning("The selected software spec requires scikit-learn=={}.".format(client_reqs['scikitlearn_version']) + install_msg)


def choose_software_version(software_spec):
    logger.info("Aggregator Software Spec: {}".format(software_spec))

    if software_spec == "runtime-23.1-py3.10" or software_spec == "336b29df-e0e1-5e7d-b6a5-f6ab722625b2":
        client_reqs = {'py_version': '3.10', 'tensorflow_version': '2.12', 'torch_version': '2.0', 'scikitlearn_version': '1.1', 'fl_extras': 'fl-rt23.1-py3.10'}
        check_python_framework_version(client_reqs)
        return "runtime-23-1"
    else:
        client_reqs = {'py_version': '3.10', 'tensorflow_version': '2.12', 'torch_version': '2.0', 'scikitlearn_version': '1.1', 'fl_extras': 'fl-rt23.1-py3.10'}
        check_python_framework_version(client_reqs)
        return "runtime-23-1"


fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)


class Party(WMLResource):
    """The Party class embodies a Federated Learning party, with methods to run, cancel, and query local training.
    Refer to the ``client.remote_training_system.create_party()`` API for more information about creating an
    instance of the Party class.
    """

    base_platform = 'runtime-23-1'
    default_software_spec = 'runtime-23.1-py3.10'
    SUPPORTED_PLATFORMS_MAP = {
        base_platform: "/runtime-23-1/ibmfl", #default
    }

    def __init__(self, client=None, **kwargs):

        libs_module = sys.modules['ibm_watson_machine_learning.libs']
        libs_location_list = libs_module.__path__

        # base location string, default to cloud location
        ibmfl_base_module_location = libs_location_list[0] + '/ibmfl'

        # process location
        ibmfl_module_location = ibmfl_base_module_location + self.SUPPORTED_PLATFORMS_MAP.get(self.base_platform)
        
        #check if using old connector script which is removed
        if not client:
            raise Exception("This version of the party connector script is outdated. "
                            "Please download the party connector script from your current Federated Learning experiment. "
                            "For more details, please refer to the documentation.")

        self.module_location = ibmfl_module_location
        self.args = kwargs
        self.Party = None
        self.connection = None
        self.log_level = None
        self.metrics_output = None

        if 'ibmfl' in sys.modules:
            del sys.modules['ibmfl']
        if 'ibmfl.party' in sys.modules:
            del sys.modules['ibmfl.party']
        if 'ibmfl.party.party' in sys.modules:
            del sys.modules['ibmfl.party.party']

        #install the general lib
        module_name = 'ibmfl'
        module_spec = importlib.util.spec_from_file_location(
            module_name, ibmfl_base_module_location + '/' + self.base_platform + '/ibmfl/__init__.py')
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        WMLResource.__init__(self, __name__, client)
        self._client = client
        self.auth_token = "Bearer " + self._client.wml_token
        self.project_id = self._client.project_id
        self.log_level = kwargs.get("log_level", "ERROR")

    def start(self):
        self.Party.start()

    def run(self, aggregator_id=None, experiment_id=None, asynchronous=True, verify=True, timeout: int = 60 * 10):
        """Connect to a Federated Learning aggregator and run local training.
        Exactly one of `aggregator_id` and `experiment_id` must be supplied.

        :param aggregator_id: aggregator identifier

            * If aggregator_id is supplied, the party will connect to the given aggregator.

        :type aggregator_id: str, optional
        :param experiment_id: experiment identifier

            * If experiment_id is supplied, the party will connect to the most recently created aggregator
                for the experiment.

        :type experiment_id: str, optional
        :param asynchronous:
            * `True`  - party starts to run the job in the background and progress can be checked later
            * `False` - method will wait until training is complete and then print the job status
        :type asynchronous: bool, optional
        :param verify: verify certificate
        :type verify: bool, optional
        :param timeout: timeout in seconds

            * If the aggregator is not ready within timeout seconds from now, exit.

        :type timeout: int, or None for no timeout

        **Examples**

        .. code-block:: python

            party.run( aggregator_id = "69500105-9fd2-4326-ad27-7231aeb37ac8", asynchronous = True, verify = True )
            party.run( experiment_id = "2466fa06-1110-4169-a166-01959adec995", asynchronous = False )

        """
        import time
        from datetime import datetime

        from ibmfl.exceptions import FLException

        timeout_time = None if timeout is None else timeout + time.time()
        if ( experiment_id is None and aggregator_id is None ) or ( experiment_id is not None and aggregator_id is not None ) :
            raise FLException("Exactly one of aggregator_id and experiment_id must be supplied")

        if ( experiment_id is not None ):
            while True:
                try:
                    details = self._client.training.get_details(get_all=True,training_definition_id=experiment_id, _internal=True)['resources']
                    details = [ d for d in details if d['entity']['status']['state'] in ['accepting_parties','pending','running'] ]
                    if not details:
                        if timeout_time and timeout_time < time.time():
                            raise FLException("Cannot find an aggregator for experiment %s", experiment_id)
                        else :
                            logger.info("Cannot find an aggregator for experiment %s. Retrying.", experiment_id)
                            time.sleep(30)
                            continue
                    else :
                        aggregator_id = max( [(t['metadata']['id'],datetime.strptime(t['metadata']['created_at'],'%Y-%m-%dT%H:%M:%S.%fZ'))
                                          for t in details],key=lambda d:d[1])[0]
                        logger.info("Using aggregator id %s", aggregator_id)
                        break
                except Exception as ex:
                    logger.exception(ex)
                    raise FLException(str(ex))

        # If crypto is enabled, test if this client supports it
        # WML training service uses the fusion_type field to determine if encryption is required.

        try:
            aggregator_details = self._client.training.get_details(
                training_uid=aggregator_id, _internal=True
            )
        except ApiRequestFailure as exc:
            logger.error("Failed get training details")
            raise FLException("Failed get training details") from exc

        # Response might not be for a federated learning training
        try:
            fusion_type = aggregator_details["entity"]["federated_learning"]["fusion_type"]
        except (KeyError, TypeError) as exc:
            logger.error("Failed to read fusion type from training details")
            raise FLException("Failed to read fusion type from training details") from exc

        if fusion_type == "crypto_iter_avg":
            logger.info("This training requires encryption")
            if not is_crypto_supported():
                logger.error(
                    "Encryption is required, but the '%s' module required for encryption was not found",
                    CRYPTO_LIBRARY,
                )
                raise FLException(f"'{CRYPTO_LIBRARY}' module not found")

        # Import the changed files in the desired lib version

        config_dict = self.args.get('config_dict')
        metrics_config = {
            "name": "WMLMetricsRecorder",
            "path": "ibmfl.party.metrics.metrics_recorder",
            "output_file": self.metrics_output,
            "output_type": "json",
            'compute_pre_train_eval': False,
            'compute_post_train_eval': False
        }
        if 'metrics_recorder' not in config_dict:
            config_dict["metrics_recorder"] = metrics_config
        wml_services_url = config_dict.get('aggregator').get('ip').split('/')[0]
        agg_info = wml_services_url + "/ml/v4/trainings/" + aggregator_id
        config_dict['aggregator']['ip'] = agg_info
        self.args['config_dict'] = config_dict

        # Verify ssl context
        if verify:
            try:
                req.get("https://" + wml_services_url + "/wml_services/training/heartbeat", verify=verify)
            except requests.exceptions.SSLError as ex:
                logger.error(str(ex))
                raise FLException("No valid certificate detected. Please replace the default certificate with your own "
                                  "TLS certificate, or set verify to False at your own risk. For more details, please see "
                                  "https://www.ibm.com/docs/en/cloud-paks/cp-data/4.0?topic=client-using-custom-tls-certificate-connect-platform")

        # Check for aggregator state and start job
        try:
            training_status = self._client.training.get_status(aggregator_id)
            state = training_status['state']
            ready = False
            if state == "pending":
                while state == "pending" and ( not timeout_time or timeout_time > time.time() ) :
                    logger.info("Waiting for aggregator accepting parties state..")
                    time.sleep(10)
                    training_status = self._client.training.get_status(aggregator_id)
                    state = training_status['state']
                if state != "accepting_parties":
                    raise FLException("The current state of training %s is %s, so the party is not able to start a job." % (aggregator_id,state))
                ready = True
            elif state == "running" or state == "accepting_parties":
                ready = True
            else:
                raise FLException("The current state of training %s is %s, so the party is not able to start a job." % (aggregator_id,state))

            if ready:

                details = self._client.training.get_details(aggregator_id, _internal=True)
                fl_entity = details["entity"]["federated_learning"]
                if "software_spec" in fl_entity:
                    software_spec = fl_entity["software_spec"]["name"] if "name" in fl_entity["software_spec"] else fl_entity["software_spec"]["id"]
                else:
                    software_spec = self.default_software_spec
                platform_env = choose_software_version(software_spec)
                logger.info("Loading {} environment..".format(platform_env))
                if "system" in details and "warnings" in details["system"]:
                    for warning in details["system"]["warnings"]:
                        logger.info("Warning: {}".format(warning["message"]))
                self.module_location = '/'.join(self.module_location.split('/')[0:-2]) + self.SUPPORTED_PLATFORMS_MAP.get(platform_env)
                
                import_diff(self.module_location)
                from ibmfl.party.party import Party
                self.Party = Party(**self.args, token=self.auth_token, self_signed_cert=not verify, log_level=self.log_level)
                self.connection = self.Party.connection
                self.start()
            else:
                raise FLException("The current state of training %s is %s, so the party is not able to start a job." % (aggregator_id,state))
            #wait for the job to finish if synchrounous
            if not asynchronous:
                while "completed" != state and "failed" != state and "canceled" != state and self.is_running():
                    training_status = self._client.training.get_status(aggregator_id)
                    state = training_status['state']
                    time.sleep(10)
                logger.info("The training finishes with %s status" % state)
        except FLException as ex:
            raise FLException(str(ex)) from ex
        except Exception as ex:
            logger.info("The party failed to start training")
            logger.exception(ex)

    def monitor_logs(self, log_level="INFO"):
        """Enable logging of the training job to standard output.
        This method should be called before calling the ``run()`` method.

        :param log_level: log level specified by user
        :type log_level: str, optional

        **Example**

        .. code-block:: python

            party.monitor_logs()

        """
        self.log_level = log_level

        # Configure logging locally as well
        from ibmfl.util.config import configure_logging_from_file
        configure_logging_from_file(log_level=log_level)
        logger.setLevel(log_level)

    def monitor_metrics(self, metrics_file="-"):
        """Enable output of training metrics.

        :param metrics_file: a filename specified by user to which the metrics should be written
        :type metrics_file: str, optional

        .. note::
            This method outputs the metrics to stdout if a filename is not specified

        **Example**

        .. code-block:: python

            party.monitor_metrics()

        """
        self.metrics_output = metrics_file

    def is_running(self):
        """Check if the training job is running.

        :return: if the job is running
        :rtype: bool

        **Example**

        .. code-block:: python

            party.is_running()

        """
        return not self.connection.stopped

    def get_round(self):
        """Get the current round number.

        :return: the current round number
        :rtype: int

        **Example**

        .. code-block:: python

            party.get_round()

        """
        return self.Party.proto_handler.metrics_recorder.get_round_no()

    def cancel(self):
        """Stop the local connection to the training on the party side.

        **Example**

        .. code-block:: python

            party.cancel()

        """
        self.Party.stop_connection()
