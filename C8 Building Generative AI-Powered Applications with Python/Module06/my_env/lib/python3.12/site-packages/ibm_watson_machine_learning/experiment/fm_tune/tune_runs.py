#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = [
    "TuneRuns"
]

from pandas import DataFrame

from ibm_watson_machine_learning.foundation_models.prompt_tuner import PromptTuner
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning import APIClient


class TuneRuns:
    """TuneRuns class is used to work with historical PromptTuner runs.

    :param client: APIClient to handle service operations
    :type client: APIClient

    :param filter: filter, user can choose which runs to fetch specifying tuning name
    :type filter: str, optional

    :param limit: int number of records to be returned
    :type limit: int
    """

    def __init__(self, client: 'APIClient', filter: str = None, limit: int = 50) -> None:

        self.client = client
        self.tuning_name = filter
        self.limit = limit

    def __call__(self, *, filter: str = None, limit: int = 50) -> 'TuneRuns':
        self.tuning_name = filter
        self.limit = limit
        return self

    def list(self) -> 'DataFrame':
        """Lists historical runs with status. If user has a lot of runs stored in the service,
        it may take long time to fetch all the information. If there is no limit set,
        get last 50 records.

        :return: Pandas DataFrame with runs IDs and state
        :rtype: pandas.DataFrame

        **Examples**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import TuneExperiment

            experiment = TuneExperiment(...)
            df = experiment.runs.list()
        """

        runs_details = self.client.training.get_details(get_all=True if self.tuning_name else False,
                                                        limit=None if self.tuning_name else self.limit,
                                                        training_type='prompt_tuning',
                                                        _internal=True)

        columns = ['timestamp', 'run_id', 'state', 'prompt tuning name']

        records = []
        for run in runs_details['resources']:
            if len(records) >= self.limit:
                break

            if {'entity', 'metadata'}.issubset(run.keys()):

                timestamp = run['metadata'].get('modified_at')
                run_id = run['metadata'].get('id', run['metadata'].get('guid'))
                state = run['entity'].get('status', {}).get('state')
                tuning_name = run['entity'].get('name', 'Unknown')

                record = [timestamp, run_id, state, tuning_name]

                if self.tuning_name is None or (self.tuning_name and self.tuning_name == tuning_name):
                    records.append(record)

        runs = DataFrame(data=records, columns=columns)
        return runs.sort_values(by=["timestamp"], ascending=False)

    def get_tuner(self, run_id: str) -> PromptTuner:
        """Create instance of PromptTuner based on tuning run with specific run_id.

        :param run_id: ID of the run
        :type run_id: str

        :return: prompt tuner object
        :rtype: PromptTuner class instance

        **Example**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            historical_tuner = experiment.runs.get_tuner(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')
        """
        # note: normal scenario

        if not isinstance(run_id, str):
            raise WMLClientError(f"Provided run_id type was {type(run_id)} (should be a string)")

        entity = self.client.training.get_details(run_id).get('entity')
        if not entity:
            raise WMLClientError("Provided run_id was invalid")

        tuning_params = entity['prompt_tuning']

        prompt_tuner = PromptTuner(name=entity.get('name'),
                                   task_id=tuning_params.get('task_id'),
                                   description=entity.get('description'),
                                   base_model=tuning_params.get('base_model', {}).get('name'),
                                   accumulate_steps=tuning_params.get('accumulate_steps'),
                                   batch_size=tuning_params.get('batch_size'),
                                   init_method=tuning_params.get('init_method'),
                                   init_text=tuning_params.get('init_text'),
                                   learning_rate=tuning_params.get('learning_rate'),
                                   max_input_tokens=tuning_params.get('max_input_tokens'),
                                   max_output_tokens=tuning_params.get('max_output_tokens'),
                                   num_epochs=tuning_params.get('num_epochs'),
                                   tuning_type=tuning_params.get('tuning_type'),
                                   verbalizer=tuning_params.get('verbalizer'),
                                   auto_update_model=entity.get('auto_update_model'))

        prompt_tuner.id = run_id
        prompt_tuner._client = self.client
        return prompt_tuner

    def get_run_details(self, run_id: str = None, include_metrics: bool = False) -> dict:
        """Get run details. If run_id is not supplied, last run will be taken.

        :param run_id: ID of the run
        :type run_id: str, optional

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: run configuration parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import TuneExperiment
            experiment = TuneExperiment(credentials, ...)

            experiment.runs.get_run_details(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')
            experiment.runs.get_run_details()
        """
        if run_id is None:
            details = self.client.training.get_details(limit=1, training_type='prompt_tuning', _internal=True).get('resources')[0]
        else:
            details = self.client.training.get_details(training_uid=run_id, _internal=True)

        if include_metrics:
            return details

        if details['entity']['status'].get('metrics', False):
            del details['entity']['status']['metrics']

        return details
