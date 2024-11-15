#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import unittest

import pandas

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.tests.conftest import load_updated_credentials, create_project, load_api_client, \
    load_cos_credentials, load_original_credentials, delete_project
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager, PromptTemplate
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, PromptTemplateFormats
from ibm_watson_machine_learning.wml_client_error import MissingValue, ValidationError, InvalidMultipleArguments, WMLClientError
from langchain.prompts import PromptTemplate as LcPromptTemplate


class TestPromptTemplate(unittest.TestCase):
    """
    These tests covers:
    - create new prompts from PromptTemplate and Langchain.PrormptTemplate object
    - create prompt with missing name (check if MissingValue is raised)
    - load prompt asset and return PromptTemplate, Langchain.PrormptTemplate and string
    - update prompot
    - delete prompt
    - get prompt lock state and change its state
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_id = create_project(load_api_client(load_original_credentials()), load_cos_credentials(), load_original_credentials())
        cls.wml_credentials = load_updated_credentials(cls.project_id)
        cls.client = APIClient(wml_credentials=cls.wml_credentials,
                               project_id=cls.project_id)
        cls.prompt_mgr = PromptTemplateManager(cls.wml_credentials, project_id=cls.project_id)
        cls.create = {"id": None}
        cls.attributes = {"prompt_id": None,
                          "input_text": None,
                          "template": None}
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.attributes["prompt_id"] is not None:
            cls.prompt_mgr.delete_prompt(cls.attributes["prompt_id"], force=True)
        delete_project(cls.project_id, cls.client)
        return super().tearDownClass()

    def tearDown(self) -> None:
        """
        Remove prompt if created during test
        """
        if self.create['id'] is not None:
            self.prompt_mgr.delete_prompt(prompt_id=self.create['id'], force=True)
        self.create['id'] = None

        return super().tearDown()

    def test_00a_create_invalid_prompt_template(self):
        self.assertRaises(ValidationError,
                          PromptTemplate,
                          name="My promot",
                          input_text="{object}",
                          input_variables=["item"])
    
    def test_00b_create_valid_prompt_template(self):
        input_text = "{object}"
        prompt = PromptTemplate(name="My promot",
                                input_text=input_text,
                                input_variables=["item"],
                                validate_template=False)
        self.assertEqual(prompt.input_text, input_text)
    
    def test_01_initialize_prompt_mgr_without_space_project(self):
        wml_credentials = TestPromptTemplate.wml_credentials
        if wml_credentials.get('project_id'):
            wml_credentials.pop('project_id')
        if wml_credentials.get('space_id'):
            wml_credentials.pop('space_id')
        
        self.assertRaises(WMLClientError,
                          PromptTemplateManager,
                          wml_credentials)

    def test_02a_store_prompt(self):
        instruction = "Write a summary"
        input_prefix = "Text"
        output_prefix = "Summary"
        input_text = "Bob has a dog"
        examples = [["Text1", "Summary1"]]
        TestPromptTemplate.attributes["input_text"] = input_text
        input_template = (f"{instruction}\n\n"
                          f"{input_prefix} "
                          f"{examples[0][0]}\n"
                          f"{output_prefix} "
                          f"{examples[0][1]}\n\n"
                          f"{input_prefix} "
                          f"{input_text}\n"
                          f"{output_prefix}")
        TestPromptTemplate.attributes["template"] = input_template

        prompt_template = PromptTemplate(name="My prompt",
                                         model_id=ModelTypes.FLAN_UL2,
                                         input_text=input_text,
                                         instruction=instruction,
                                         input_prefix=input_prefix,
                                         output_prefix=output_prefix,
                                         examples=examples)
        stored_prompt = self.prompt_mgr.store_prompt(prompt_template)
        TestPromptTemplate.attributes["prompt_id"] = stored_prompt.prompt_id

        self.assertTrue(TestPromptTemplate.attributes["prompt_id"])
        self.assertFalse(stored_prompt.is_template)

    def test_02b_store_prompt_template(self):
        input_text = "Bob has a {object}"

        prompt_template = PromptTemplate(name="My prompt",
                                         model_id=ModelTypes.FLAN_UL2,
                                         input_text=input_text,
                                         input_variables=["object"])
        stored_prompt_template = self.prompt_mgr.store_prompt(prompt_template)
        self.assertTrue(stored_prompt_template.is_template)
        self.assertEqual(stored_prompt_template.model_id, ModelTypes.FLAN_UL2.value)
        self.create["id"] = stored_prompt_template.prompt_id

    def test_02c_store_langchain(self):
        template = ("Generate a random question"
                    " about {topic}: Question: ")
        langchain_prompt = LcPromptTemplate(input_variables=["topic"],
                                            template=template)
        stored_prompt_template = self.prompt_mgr.store_prompt(langchain_prompt)
        self.assertEqual(stored_prompt_template.input_text, template)
        self.assertListEqual(list(stored_prompt_template.input_variables.keys()), ["topic"])

        self.create["id"] = stored_prompt_template.prompt_id

    def test_02d_missing_name(self):
        self.assertRaises(MissingValue, self.prompt_mgr.store_prompt,
                          PromptTemplate(input_text="Hello World!"))

    def test_03a_load_prompt(self):
        loaded_prompt = self.prompt_mgr.load_prompt(TestPromptTemplate.attributes["prompt_id"])

        self.assertTrue(isinstance(loaded_prompt, PromptTemplate))
        self.assertEqual(loaded_prompt.input_text,
                         TestPromptTemplate.attributes["input_text"])

    def test_03b_load_prompt(self):
        loaded_prompt = self.prompt_mgr.load_prompt(TestPromptTemplate.attributes["prompt_id"],
                                                    PromptTemplateFormats.LANGCHAIN)

        self.assertTrue(isinstance(loaded_prompt, LcPromptTemplate))

        self.assertEqual(loaded_prompt.template,
                         TestPromptTemplate.attributes["template"])

    def test_03c_load_prompt(self):
        loaded_prompt = self.prompt_mgr.load_prompt(TestPromptTemplate.attributes["prompt_id"],
                                                    PromptTemplateFormats.STRING)

        self.assertTrue(isinstance(loaded_prompt, str))
        self.assertEqual(loaded_prompt,
                         TestPromptTemplate.attributes["template"])

    def test_04_update_prompt(self):
        old_input = TestPromptTemplate.attributes["input_text"]
        new_input_text = old_input + "Hello World"
        new_prompt = PromptTemplate(input_text=new_input_text)
        self.prompt_mgr.update_prompt(TestPromptTemplate.attributes["prompt_id"],
                                      new_prompt)
        
        self.assertEqual(self.prompt_mgr.load_prompt(TestPromptTemplate.attributes["prompt_id"],
                                                     PromptTemplateFormats.STRING),
                         TestPromptTemplate.attributes["template"].replace(old_input,
                                                                           new_input_text))

    def test_05a_list(self):
        df_prompts = self.prompt_mgr.list()

        self.assertTrue(isinstance(df_prompts, pandas.core.frame.DataFrame))
        self.assertTrue(df_prompts["ID"].isin([TestPromptTemplate.attributes["prompt_id"]]).any())

    def test_05b_list_sorted_by_last_modified(self):
        df_prompts = self.prompt_mgr.list()
        df_sorted_by_last_modified = df_prompts.sort_values("LAST MODIFIED", ascending=False)

        self.assertEqual(df_sorted_by_last_modified["LAST MODIFIED"].to_list(),
                         sorted(df_sorted_by_last_modified["LAST MODIFIED"].to_list(), reverse=True),
                         msg="ERROR: Sorted in wrong order!")

    def test_06_delete_prompt(self):
        template = ("Generate a random question"
                    " about {topic}: Question: ")
        langchain_prompt = LcPromptTemplate(input_variables=["topic"],
                                            template=template)
        stored_prompt_template = self.prompt_mgr.store_prompt(langchain_prompt)
        number_of_prompts = len(self.prompt_mgr.list())
        self.prompt_mgr.delete_prompt(prompt_id=stored_prompt_template.prompt_id,
                                      force=True)

        self.assertEqual(len(self.prompt_mgr.list()), number_of_prompts - 1)

    def test_07_lock_prompt(self):
        lock_state = self.prompt_mgr.get_lock(TestPromptTemplate.attributes["prompt_id"])

        self.assertTrue(lock_state["locked"])

        self.prompt_mgr.unlock(TestPromptTemplate.attributes["prompt_id"])
        lock_state = self.prompt_mgr.get_lock(TestPromptTemplate.attributes["prompt_id"])

        self.assertFalse(lock_state["locked"])

    def test_08_project_and_space_id(self):
        self.assertRaises(InvalidMultipleArguments,
                          PromptTemplateManager,
                          self.wml_credentials,
                          project_id="111",
                          space_id="2222")
