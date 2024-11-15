#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import logging
import allure


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_create_prompt(prompt_template_step):
    prompt_template_step.create_prompt()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_create_prompt_for_space(prompt_template_step, space_id):
    prompt_template_step.data_storage.prompt_mgr.params = {'space_id': space_id}
    prompt_template_step.create_prompt()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_create_freeform_prompt(prompt_template_step):
    prompt_template_step.create_freeform_prompt()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_update_prompt(prompt_template_step):
    first_id_element = prompt_template_step.find_existing_prompt(0)
    prompt_template_step.edit_existing_prompt(first_id_element)


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_unlock_prompts(prompt_template_step):
    prompt_list = prompt_template_step.get_prompts_list()
    position = 0

    for _ in range(len(prompt_list)):
        prompt = prompt_template_step.find_existing_prompt(position)
        lock_state = prompt_template_step.get_lock_state(prompt)

        if lock_state["locked"]:
            prompt_template_step.unlock_prompt(prompt)
            logging.info(f'\nUnlocked - Prompt with ID:{prompt}!')
        else:
            logging.info(f'\nPrompt with ID:{prompt} already unlocked!')
        position += 1


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_lock_prompts(prompt_template_step):
    prompt_list = prompt_template_step.get_prompts_list()
    position = 0

    for _ in range(len(prompt_list)):
        prompt = prompt_template_step.find_existing_prompt(position)
        lock_state = prompt_template_step.get_lock_state(prompt)

        if not lock_state["locked"]:
            prompt_template_step.lock_prompt(prompt)
            print(f'\nLocked - Prompt with ID:{prompt}!')
        else:
            print(f'\nPrompt with ID:{prompt} already locked!')
        position += 1


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_show_lock_state(prompt_template_step):
    prompt_list = prompt_template_step.get_prompts_list()
    position = 0

    for _ in range(len(prompt_list)):
        prompt = prompt_template_step.find_existing_prompt(position)
        state_list = prompt_template_step.get_lock_state(prompt)
        logging.info(state_list)
        assert state_list.get('locked') in [True, False]
        position += 1


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template")
def test_delete_prompt(prompt_template_step):
    position = 0
    prompt_template_step.create_prompt()
    prompt_list = prompt_template_step.get_prompts_list()
    for _ in range(len(prompt_list)):
        if len(prompt_list) > 0:
            prompt = prompt_template_step.find_existing_prompt(position)
            prompt_template_step.delete_prompt_template(prompt)
            logging.info("\nPrompt: " + prompt + ": has been deleted")


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_prompt_template_deployment(universal_step, data_storage, prompt_id, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id}

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_prompt_template_deployment_without_base_model(universal_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {}
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_prompt_template_deployment_without_project_space(universal_step, prompt_id,
                                                          data_storage):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    copy_space = data_storage.api_client.default_space_id
    copy_project = data_storage.api_client.default_project_id
    data_storage.api_client.default_space_id = None
    data_storage.api_client.default_project_id = None

    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)

    data_storage.api_client.default_space_id = copy_space
    data_storage.api_client.default_project_id = copy_project


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_generate_without_prompt_variables(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.generate_without_prompt_variables()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_generate(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.generate()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_generate_text(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.generate_text()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_generate_stream_text(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.generate_stream_text()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_model_generate_without_variables(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.model_generate_without_variables()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_model_generate(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.model_generate()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_model_generate_text(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.model_generate_text()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_model_credentials_generate(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.model_credentials_generate()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()


@allure.parent_suite("Foundation Models")
@allure.suite("Prompt Template Deployment")
def test_update_deployment(universal_step, prompt_template_step, prompt_id, data_storage, project_id):
    meta_props = {
        data_storage.api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
        data_storage.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        data_storage.api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: data_storage.base_model_id
    }

    universal_step.set_project_id(project_id)
    universal_step.create_deployment(data_storage, meta_props, item_id=prompt_id)
    prompt_template_step.update_deployment_name()
    universal_step.get_deployment_details(prompt_id)
    universal_step.delete_deployment()
