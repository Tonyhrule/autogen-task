import os
import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

import os
import json
import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
        }
    },
)

#1. Create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

#2. Create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

#Given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
math_problem = "What is the inverse of the matrix $[[1,2,-1],[-2,0,1],[1,-1,0]]$ ?"
#We call initiate_chat to start the conversation.
#When setting message=mathproxyagent.message_generator, you need to pass in the problem through the problem parameter.
mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)
