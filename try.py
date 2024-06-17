import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
load_dotenv()
import os
import openai

from autogen import ConversableAgent
api_key = os.getenv("OPENAI_API_KEY")

cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": api_key}]},
    human_input_mode="NEVER",  # Never ask for human input.
)

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": api_key}]},
    human_input_mode="NEVER",  # Never ask for human input.
)

cathy_reply = cathy.generate_reply(messages=[{"content": "Tell me about AI.", "role": "user"}])
joe_reply = joe.generate_reply(messages=[{"content": "Tell me about AI.", "role": "user"}])

result = joe.initiate_chat(cathy, message="Cathy, tell me about AI.", max_turns=2)






