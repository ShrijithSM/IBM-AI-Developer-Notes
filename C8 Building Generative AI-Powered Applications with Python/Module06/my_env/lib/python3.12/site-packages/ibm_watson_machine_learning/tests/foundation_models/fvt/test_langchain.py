#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import os
import pytest
from ibm_watson_machine_learning.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain import PromptTemplate
from langchain.chains import (LLMChain, SimpleSequentialChain, SequentialChain, TransformChain, ConversationChain,
                              LLMMathChain)
from langchain.chains.router import MultiPromptChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods


# wml_credentials = get_wml_credentials()
# model_types_list = [model.value for model in ModelTypes]
# available_models = [model_spec['model_id'] for model_spec in get_model_specs(wml_credentials.get('url')).get('resources', []) 
#                     if model_spec['model_id'] in model_types_list ]

# For automatic tests we select only one model
available_models = ['google/flan-ul2']


class TestLangchain:
    """
    This tests covers:
    - response using LLMChain
    - response using SequentialChain
    - response using SimpleSequentialChain
    - response using TransformChain
    - response using ConversationChain
    - response using `to_langchain` wrapper
    """

    @pytest.mark.parametrize("model_type", available_models)
    def test_01_llm_chain(self, model_type, project_id, api_client):
        prompt_template = "What is a good name for a company that makes {product} ?\n"
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        product = 'car'
        print("\n" + prompt_template.replace('{product}', product))
        review = llm_chain(product)
        print(f"Respond by use 'llm_chain(product)': {review['text']}")
        assert product == review['product']
        assert review['text']
        review_run = llm_chain.run(product)
        print(f"Respond by use 'llm_chain.run(product)': {review_run}")
        assert review['text'] == review_run
        review_predict = llm_chain.predict(product=product)
        print(f"Respond by use 'llm_chain.predict(product)': {review_predict}")
        assert review['text'] == review_predict

    @pytest.mark.parametrize("model_type", available_models)
    def test_02_llm_chain_model_with_params(self, model_type, project_id, api_client):
        params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.STOP_SEQUENCES: ['\n\n']
        }
        prompt_template = "What color is the {flower}?\n"
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            params=params,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        flower = 'sunflower'
        print("\n" + prompt_template.replace('{flower}', flower))
        review = llm_chain(flower)
        print(f"Respond by use 'llm_chain(flower)': {review['text']}")
        assert flower == review['flower']
        assert review['text']
        review_run = llm_chain.run(flower)
        print(f"Respond by use 'llm_chain.run(flower)': {review['text']}")
        assert review['text'] == review_run
        review_predict = llm_chain.predict(flower=flower)
        print(f"Respond by use 'llm_chain.predict(flower)': {review['text']}")
        assert review['text'] == review_predict

    @pytest.mark.parametrize("model_type", available_models)
    def test_03_sequential_chain(self, model_type, project_id, api_client):
        template_1 = """You are a playwright. 
        Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

        Title: {title}
        Era: {era}
        Playwright: This is a synopsis for the above play:\n"""
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        prompt_template_1 = PromptTemplate(input_variables=["title", "era"], template=template_1)
        synopsis_chain = LLMChain(llm=llm, prompt=prompt_template_1, output_key="synopsis")

        template_2 = """You are a play critic from the New York Times. 
        Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:\n"""
        prompt_template_2 = PromptTemplate(input_variables=["synopsis"], template=template_2)
        review_chain = LLMChain(llm=llm, prompt=prompt_template_2, output_key="review")

        overall_chain = SequentialChain(
            chains=[synopsis_chain, review_chain],
            input_variables=["era", "title"],
            output_variables=["synopsis", "review"],
            verbose=True)
        title = "Tragedy at sunset on the beach"
        era = "Victorian England"
        review = overall_chain({"title": title, "era": era})
        print(review)
        assert len(review) == 4, 'We should have 4 elements'
        assert review['title'] == title
        assert review['era'] == era
        assert review['synopsis']
        assert review['review']

    @pytest.mark.parametrize("model_type", available_models)
    def test_04_simple_sequential_chain(self, model_type, project_id, api_client):
        params = {
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenParams.MAX_NEW_TOKENS: 100,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }
        pt_1 = PromptTemplate(
            input_variables=["topic"],
            template="Generate a random question about {topic}: Question: "
        )
        pt_2 = PromptTemplate(
            input_variables=["question"],
            template="Answer the following question: {question}",
        )
        model_1 = ModelInference(model_id=model_type, api_client=api_client, project_id=project_id, params=params)
        llm_1 = WatsonxLLM(model=model_1)
        model_2 = ModelInference(model_id=model_type, api_client=api_client, project_id=project_id, params=params)
        llm_2 = WatsonxLLM(model=model_2)
        prompt_to_flan = LLMChain(llm=llm_1, prompt=pt_1)
        flan_to_t5 = LLMChain(llm=llm_2, prompt=pt_2)

        qa = SimpleSequentialChain(chains=[prompt_to_flan, flan_to_t5], verbose=True)
        assert len(qa.chains) == 2
        assert pt_1.template == qa.chains[0].prompt.template
        assert pt_2.template == qa.chains[1].prompt.template
        review = qa.run("cat")
        assert review

    @pytest.mark.parametrize("model_type", available_models)
    def test_05_transformation_chain(self, model_type, project_id, api_client):
        with open(os.path.join(os.path.dirname(__file__), "../artifacts/state_of_the_union.txt")) as f:
            state_of_the_union = f.read()

        def transform_func(inputs: dict) -> dict:
            text = inputs["text"]
            shortened_text = "\n\n".join(text.split("\n\n")[:10])
            return {"output_text": shortened_text}

        transform_chain = TransformChain(
            input_variables=["text"], output_variables=["output_text"], transform=transform_func)
        transform_review = transform_chain(state_of_the_union)
        print(f"\n Transform_review: {transform_review['output_text']}")
        assert transform_review['output_text'] in state_of_the_union

        template = """Summarize this text:

        {output_text}

        Summary:\n"""
        prompt = PromptTemplate(input_variables=["output_text"], template=template)
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
        review = sequential_chain.run(state_of_the_union)
        print(f'\nReview: {review}')
        assert review

    @pytest.mark.skip(reason="This scenario not supported yet")
    def test_06_router_chain(self, project_id, api_client):
        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.

        Here is a question:
        {input}"""

        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.

        Here is a question:
        {input}"""
        prompt_infos = [
            {
                "name": "physics",
                "description": "Good for answering questions about physics",
                "prompt_template": physics_template,
            },
            {
                "name": "math",
                "description": "Good for answering math questions",
                "prompt_template": math_template,
            }
        ]
        params = {
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.REPETITION_PENALTY: 2
        }
        model = ModelInference(
            model_id="eleutherai/gpt-neox-20b",
            api_client=api_client,
            project_id=project_id,
            params=params)
        llm = WatsonxLLM(model=model)
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=llm, output_key="text")

        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser()
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True
        )
        print(chain.run("What is black body radiation?"))

    @pytest.mark.parametrize("model_type", available_models)
    def test_07_conversation_buffer_memory_chain(self, model_type, project_id, api_client):
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory()
        )
        qa_1 = "Answer briefly. What are the first 3 colors of a rainbow?\n"
        print(f"\n{qa_1}")
        response_1 = conversation(qa_1)
        print(response_1['response'])
        assert qa_1 == response_1['input']
        assert response_1['response']
        assert not response_1['history']

        qa_2 = "And the first 4?"
        print(qa_2)
        response_2 = conversation(qa_2)
        print(response_2['response'])
        assert qa_2 == response_2['input']
        assert response_2['response']
        history = response_2['history'].split('\n', 2)
        assert qa_1 in history[0]+'\n'
        assert response_1['response'] in history[2]

    @pytest.mark.parametrize("model_type", [model for model in available_models if model not in ['google/flan-t5-xxl', 'google/flan-ul2', 'ibm/granite-13b-chat-v1']])
    def test_08_math_chain(self, model_type, project_id, api_client):
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm = WatsonxLLM(model=model)
        llm_math = LLMMathChain.from_llm(llm, verbose=True)
        qa = "What is 2 raised to the 3 power?"
        response = llm_math(qa)
        assert qa == response['question']
        assert response['answer']

    @pytest.mark.parametrize("model_type", available_models)
    def test_09_to_langchain_wrapper(self, model_type, project_id, api_client):
        prompt_template = "What color is the {flower}?\n"
        model = ModelInference(
            model_id=model_type,
            api_client=api_client,
            project_id=project_id)
        llm_chain = LLMChain(
            llm=model.to_langchain(),
            prompt=PromptTemplate.from_template(prompt_template)
        )
        flower = 'sunflower'
        print("\n" + prompt_template.replace('{flower}', flower))
        review = llm_chain(flower)
        print(f"Respond by use 'llm_chain(flower)': {review['text']}")
        assert flower == review['flower']
        assert review['text']
        review_run = llm_chain.run(flower)
        print(f"Respond by use 'llm_chain.run(flower)': {review['text']}")
        assert review['text'] == review_run
        review_predict = llm_chain.predict(flower=flower)
        print(f"Respond by use 'llm_chain.predict(flower)': {review['text']}")
        assert review['text'] == review_predict
