#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024 .
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import pytest

from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplate
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

"""
When adding some fixture here follow that pattern:
- first "little" fixtures, that  are returning something simple as `prompt_id`;
- second "more complex setups" like `fixture_setup_prompt_mgr`;
- last one "tear down methods"
"""


@pytest.fixture(scope='function', name="prompt_id")
def fixture_prompt_id(data_storage: object):
    """
    Fixture that is getting prompt ID
        return:
            str: prompt_id
    """
    prompt_id = data_storage.prompt_mgr.store_prompt(PromptTemplate(name="My test prompt",
                                                                    model_id=ModelTypes.FLAN_T5_XL,
                                                                    input_text="What is a {object} and how does it work?",
                                                                    input_variables=["object"])).prompt_id
    return prompt_id


@pytest.fixture(scope='class', name="model_id")
def fixture_model_id():
    """
    Fixture that is getting model ID
        return:
           str: model_id
    """
    model_id = ModelTypes.STARCODER.value

    return model_id


@pytest.fixture(scope='function', name="prompt_mgr")
def fixture_prompt_mgr(credentials, project_id):
    """
    Fixture that initialize prompt manager that is working on `project_id`.
    To change to `space_id` in test you need to "prompt_mgr.params = {'space_id': space_id}"
        return:
           str: model_id
    """
    prompt_mgr = PromptTemplateManager(credentials, project_id=project_id)
    return prompt_mgr

