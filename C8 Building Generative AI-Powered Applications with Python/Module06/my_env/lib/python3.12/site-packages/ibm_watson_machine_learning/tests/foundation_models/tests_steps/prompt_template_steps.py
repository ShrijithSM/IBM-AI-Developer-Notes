#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest

from ibm_watson_machine_learning.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplate
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.wml_client_error import WMLClientError


class PromptTemplateSteps:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    check_value_error_message = "ERROR: Value it is NOT the same!"
    prompt_list_is_empty_message = "INFO: Prompt list is empty!"
    update_error_message = "ERROR: Name it is not updated!"
    nothing_to_unlock_message = "INFO: There is not any prompt to unlock!"
    lock_changed_error_message = "ERROR: Lock did not changed!"
    nothing_to_lock_message = "INFO: There is not any prompt to unlock!"
    prompt_list_is_not_empty_error_message = "ERROR: list it is not empty"
    lock_state_error_message = "ERROR: Cannot get lock state of prompts!"
    prompt_name = "Testing Prompt - TG"

    def create_prompt(self):
        instruction = "Write a summary"
        input_prefix = "Text"
        output_prefix = "Summary"
        input_text = "Bob has a dog"
        examples = [["Text1", "Summary1"]]

        prompt_template = PromptTemplate(name=self.prompt_name,
                                         model_id=ModelTypes.FLAN_UL2,
                                         input_text=input_text,
                                         instruction=instruction,
                                         input_prefix=input_prefix,
                                         output_prefix=output_prefix,
                                         examples=examples)

        self.data_storage.prompt_mgr.store_prompt(prompt_template)

        assert prompt_template.name == self.prompt_name, self.check_value_error_message
        assert prompt_template.instruction == instruction, self.check_value_error_message
        assert prompt_template.input_prefix == input_prefix, self.check_value_error_message
        assert prompt_template.input_text == input_text, self.check_value_error_message
        assert prompt_template.output_prefix == output_prefix, self.check_value_error_message
        assert prompt_template.examples == examples, self.check_value_error_message

    def get_prompts_list(self):
        prompt_list = self.data_storage.prompt_mgr.list()['ID']

        if len(prompt_list) == 0:
            raise Exception(self.prompt_list_is_empty_message)

        return prompt_list

    def find_existing_prompt(self, position):
        prompt_list = self.data_storage.prompt_mgr.list()['ID']
        first_existing_prompt = self.data_storage.prompt_mgr.list()['ID'][position]

        assert prompt_list.eq(first_existing_prompt).any(), self.prompt_list_is_empty_message

        return first_existing_prompt

    def edit_existing_prompt(self, first_id_element):
        new_name = "New Test Template Name - TG"
        new_instruction = "Updated Write a summary"
        new_input_prefix = "Updated Text"
        new_output_prefix = "Updated Summary"
        new_input_text = "Updated Input Text"
        new_examples = [["Updated Text - 1", "Updated Summary - 1"],
                        ["Updated Text - 2", "Updated Summary - 2"]]

        loaded_old_prompt = self.data_storage.prompt_mgr.load_prompt(first_id_element)
        loaded_old_prompt.name = new_name
        loaded_old_prompt.instruction = new_instruction
        loaded_old_prompt.input_prefix = new_input_prefix
        loaded_old_prompt.output_prefix = new_output_prefix
        loaded_old_prompt.input_text = new_input_text
        loaded_old_prompt.examples = new_examples

        self.data_storage.prompt_mgr.update_prompt(first_id_element, loaded_old_prompt)

        loaded_updated_prompt = self.data_storage.prompt_mgr.load_prompt(first_id_element)

        assert loaded_updated_prompt.name == new_name, self.update_error_message
        assert loaded_updated_prompt.instruction == new_instruction, self.update_error_message
        assert loaded_updated_prompt.input_prefix == new_input_prefix, self.update_error_message
        assert loaded_updated_prompt.output_prefix == new_output_prefix, self.update_error_message
        assert loaded_updated_prompt.input_text == new_input_text, self.update_error_message
        assert loaded_updated_prompt.examples == new_examples, self.update_error_message

    def unlock_prompt(self, prompt_id):
        self.data_storage.prompt_mgr.unlock(prompt_id)
        lock_state = self.data_storage.prompt_mgr.get_lock(prompt_id)

        assert not lock_state["locked"], self.lock_changed_error_message

    def lock_prompt(self, prompt_id):
        self.data_storage.prompt_mgr.lock(prompt_id)
        lock_state = self.data_storage.prompt_mgr.get_lock(prompt_id)

        assert lock_state["locked"], self.lock_changed_error_message

    def get_lock_state(self, prompt_id):
        lock_state = self.data_storage.prompt_mgr.get_lock(prompt_id)

        assert True or False in lock_state, self.lock_state_error_message
        return lock_state

    def load_prompt(self, prompt_id):
        loaded_prompt = self.data_storage.prompt_mgr.load_prompt(prompt_id)

        return loaded_prompt

    def create_freeform_prompt(self):
        input_text = "Bob has a {object}"

        prompt_template = PromptTemplate(name=self.prompt_name,
                                         input_text=input_text,
                                         input_variables=["object"],
                                         model_id=ModelTypes.FLAN_UL2)

        self.data_storage.prompt_mgr.store_prompt(prompt_template)

        assert prompt_template.name == self.prompt_name, self.check_value_error_message
        assert prompt_template.input_text == input_text, self.check_value_error_message

    def delete_prompt_template(self, prompt_id):
        self.data_storage.prompt_mgr.delete_prompt(prompt_id, force=True)

        assert prompt_id not in self.data_storage.prompt_mgr.list()['ID']

    def generate_without_prompt_variables(self):
        with pytest.raises(WMLClientError):
            self.data_storage.api_client.deployments.generate(self.data_storage.deployment_id)

    def generate(self):
        generate_response = self.data_storage.api_client.deployments.generate(self.data_storage.deployment_id,
                                                                              params={"prompt_variables": {
                                                                                  "object": "loan"}})
        assert isinstance(generate_response, dict)
        assert generate_response.get('model_id', "") == self.data_storage.base_model_id
        assert isinstance(generate_response.get('results', [{}])[0].get('generated_text'), str), \
            'Generated text it is not a `String Type`!'

    def generate_text(self):
        generated_text = self.data_storage.api_client.deployments.generate_text(self.data_storage.deployment_id,
                                                                                params={"prompt_variables": {
                                                                                    "object": "loan"}})

        assert isinstance(generated_text, str), 'Generated text it is not a `String Type`!'

    def generate_stream_text(self):
        generated_text_stream = \
            list(self.data_storage.api_client.deployments.generate_text_stream(self.data_storage.deployment_id,
                                                                               params={"prompt_variables": {
                                                                                   "object": "loan"}}))[0]

        assert isinstance(generated_text_stream, str), 'Text stream it is not a `String Type`!'

    def model_generate_without_variables(self):
        model = ModelInference(deployment_id=self.data_storage.deployment_id,
                               api_client=self.data_storage.api_client)

        with pytest.raises(WMLClientError):
            model.generate(self.data_storage.deployment_id)

    def model_generate(self):
        model = ModelInference(deployment_id=self.data_storage.deployment_id,
                               api_client=self.data_storage.api_client)

        generate_response = model.generate(params={"prompt_variables": {"object": "loan"}})

        assert isinstance(generate_response, dict), 'Generated response it is not a `Dict Type`!'
        assert generate_response.get('model_id', "") == self.data_storage.base_model_id, \
            'Generated response model it is not equal to `base_model_id`!'
        assert isinstance(generate_response.get('results', [{}])[0].get('generated_text'), str), \
            'Generated response it is not a `String Type`!'

    def model_generate_text(self):
        model = ModelInference(deployment_id=self.data_storage.deployment_id,
                               api_client=self.data_storage.api_client)
        assert isinstance(model.generate_text(params={"prompt_variables": {"object": "loan"}}), str), \
            "'Generated text model it is not a `String Type`!'"

    def model_generate_stream_text(self):
        model = ModelInference(deployment_id=self.data_storage.deployment_id,
                               api_client=self.data_storage.api_client)
        assert isinstance(list(model.generate_text_stream(params={"prompt_variables": {"object": "loan"}}))[0], str), \
            "'Generated model stream text it is not a `String Type`!'"

    def model_credentials_generate(self):
        model = ModelInference(deployment_id=self.data_storage.deployment_id,
                               credentials=self.data_storage.credentials, project_id=self.data_storage.project_id)
        assert isinstance(list(model.generate_text_stream(params={"prompt_variables": {"object": "loan"}}))[0], str), \
            "Generated model credentials it is not `String Type`!"

    def update_deployment_name(self):
        new_name = "Changed Name"
        metadata = self.data_storage.api_client.deployments.update(self.data_storage.deployment_id,
                                                                   changes={
                                                                       self.data_storage.api_client.deployments.ConfigurationMetaNames.NAME: new_name})
        assert metadata.get('entity', {}).get('name', "") == new_name, "Update has not been applied"
