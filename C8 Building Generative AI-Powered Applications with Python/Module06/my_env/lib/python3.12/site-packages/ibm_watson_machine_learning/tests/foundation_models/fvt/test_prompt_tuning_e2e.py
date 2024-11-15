#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import allure

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, PromptTuningInitMethods, \
    PromptTuningTypes


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Tuning")
def test_prompt_tuning_c(prompt_tuning_step, universal_step, data_storage, space_id):
    prompt_tuning_info_c = dict(
        name="SDK test Classification Container",
        task_id="summarization",
        base_model=ModelTypes.LLAMA_2_13B_CHAT,
        init_method=PromptTuningInitMethods.TEXT,
        init_text='Summarize: ',
        num_epochs=5
    )

    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "PT deployment SDK tests",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
    universal_step.data_storage.space_id = universal_step.get_space_id()
    universal_step.set_space_id(universal_step.data_storage.space_id)

    prompt_tuning_step.write_data_to_container()
    universal_step.initialize_tune_experiment()
    prompt_tuning_step.data_reference_setup_c()
    prompt_tuning_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_c)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.promote_model_to_deployment_space()
    universal_step.get_promoted_model_details()
    universal_step.list_repository()
    universal_step.create_deployment(data_storage, meta_props, item_id=data_storage.promoted_model_id)
    universal_step.response_from_deployment_inference()
    universal_step.delete_deployment()
    universal_step.delete_experiment()
    universal_step.delete_models()
    prompt_tuning_step.delete_container()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Tuning")
def test_prompt_tuning_ca(prompt_tuning_step, universal_step, space_id):
    prompt_tuning_info_ca_default = dict(
        name="SDK test Classification with COS Connected Asset",
        task_id="generation",
        base_model=ModelTypes.GRANITE_13B_INSTRUCT_V2,
        num_epochs=3
    )

    universal_step.data_storage.space_id = universal_step.get_space_id()
    universal_step.set_space_id(universal_step.data_storage.space_id)

    prompt_tuning_step.prepare_COS_instance_and_connection()
    universal_step.initialize_tune_experiment()
    prompt_tuning_step.data_reference_setup_ca_default()
    prompt_tuning_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_ca_default)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.promote_model_to_deployment_space()
    universal_step.get_promoted_model_details()
    universal_step.list_repository()
    prompt_tuning_step.read_results_reference_filename()
    universal_step.response_from_deployment_inference()
    universal_step.delete_experiment()
    universal_step.delete_models()
    prompt_tuning_step.delete_connection_and_connected_data_asset()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Tuning")
def test_prompt_tuning_da(prompt_tuning_step, universal_step, data_storage, project_id, space_id):
    prompt_tuning_info_da_default = dict(
        name="SDK test Classification",
        task_id="classification",
        base_model='google/flan-t5-xl',
        num_epochs=2,
        max_input_tokens=128,
        max_output_tokens=4,
        accumulate_steps=2,
        learning_rate=0.1,
        tuning_type=PromptTuningTypes.PT,
        verbalizer='Input: {{input}} Output:',
        auto_update_model=False
    )

    prompt_tuning_step.data_storage.SPACE_ONLY = False
    universal_step.data_storage.SPACE_ONLY = False

    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "PT deployment SDK tests",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {}
    }

    universal_step.set_project_id(project_id)
    prompt_tuning_step.prepare_data_asset()
    universal_step.initialize_tune_experiment()
    prompt_tuning_step.data_reference_setup_da()
    prompt_tuning_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_da_default)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.set_space_id(space_id)
    universal_step.promote_model_to_deployment_space()
    universal_step.get_promoted_model_details()
    universal_step.list_repository()
    universal_step.create_deployment(data_storage, meta_props, item_id=data_storage.stored_model_id)
    universal_step.response_from_deployment_inference()
    universal_step.delete_deployment()
    universal_step.delete_experiment()
    universal_step.delete_models()
    prompt_tuning_step.delete_data_asset()
