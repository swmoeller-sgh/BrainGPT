from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

import os
import openai

# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)


# [LOAD enironment variables]
openai.api_key = os.getenv("OPENAI_API_KEY")        # openAI-settings


response_schemas = [
    ResponseSchema(name="question", description="State the initial question"),
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="citation", description="Citation form the source with max. 20 words"),
    ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)

_input = prompt.format_prompt(question="Describe the capital of france.")
output = model(_input.to_string())

output_parser.parse(output)

print(output)
