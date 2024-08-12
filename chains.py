import datetime
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-4-turbo-preview")

## Output parsers
# Transofrm repsose from LLM to a dictionary
parser = JsonOutputToolsParser(return_id=True)

# Take the respnose from LLM and search for the function calling 
# invocation, parse it and transform it into a AnswerQuestion object
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# create prompt and surrounding system messsages and user prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """,
        ),
        # store message history and share it with Revisor agent
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

# Prepare prompts prior to shipping to LLM
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# Create first resonder chain
# Send response to LLM and bind/define the AnswerQuestion object as a tool for function calling
# tool_choice will force the LLM to always use the AnswerQuestion tool, thus grounding the object
# we want to receive. 
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """
Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In the form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not 250 words.
"""

# Revise actor_prompt_template with the revised instructions and force llm to use ReviseAnswer function/tool 
revisor = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

### ------ main ------- ###
if __name__ == '__main__':
    print("Hello Reflexion")
    human_message = HumanMessage(
        content="Write about AI-powered SOC / autonomous soc problem domain,"
        "list startups that do that and raised capital."
    )
    
    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    
    # invoke chain
    res = chain.invoke(input={"messages": [human_message]})
    print(res)