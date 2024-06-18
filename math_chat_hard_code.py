import time
import autogen
from autogen.cache import Cache
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from typing_extensions import Annotated

matrix = [[1,2,-1],[-2,0,1],[1,-1,0]]

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
        }
    },
)
"""
PARAMETERS FOR FUNCTIONS
{'functions': [{'function': {'description': 'Search Google for recent results.', 'name': 'google_search', 'parameters' : 
                         {'properties': {'__arg1': {...}}, 'required': ['__arg1'], 'type': 'object'}}, 'type': 'function'}]} 
"""
tools = [{
    "name": "inverse_matrix", 
    "description": "Finds the inverse of a given matrix",
    "parameters": {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "description": "A matrix that is used to find the inverse.",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                }
            }
        },
        "required": ["matrix"]
    }
}]

# Create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="For coding tasks, only use the tools you have been provided with.",
    llm_config={
        "tools": tools,  # Change from 'functions' to 'tools'
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)


#2. Create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    system_message="A proxy for the user that is used to execute code.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"), #finishes with Terminate message
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

def find_inverse_matrix(matrix) -> list:
    identity_matrix = [[1,0,0],[0,1,0],[0,0,1]]
    n = len(matrix)
    row_index = list(range(n))
    #going through each row in the matrix
    for row in range(n): #finding equalizer to make diagonals as 1
        equalizer = 1/matrix[row][row]

        for col in range(n): #getting 1s in the diagonal
            matrix[row][col] = matrix[row][col] * equalizer
            identity_matrix[row][col] = identity_matrix[row][col] * equalizer
        
        #making the outsides zeros
        for out_x in row_index[:row] + row_index[row + 1:]: #finding the equalizer to make zeros
            equalizer2 = matrix[out_x][row]
            
            for out_y in range(n): #applying the equalizer to make the outsides zeroes
                matrix[out_x][out_y] = matrix[out_x][out_y] - equalizer2 * matrix[row][out_y]
                identity_matrix[out_x][out_y] = identity_matrix[out_x][out_y] - equalizer2 * identity_matrix[row][out_y]
    return identity_matrix

@mathproxyagent.register_for_execution()
@assistant.register_for_llm(name="inverse_matrix", description="A function that inverses a matrix.")
def matrix_execute(matrix: Annotated[list, "A 3 by 3 matrix with integers."]) -> list:
    inversed_matrix = find_inverse_matrix(matrix)
    return inversed_matrix

autogen.agentchat.register_function(
    matrix_execute,
    caller=assistant,
    executor=mathproxyagent,
    description="inverse a matrix",
)

#Given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
math_problem = "What is the inverse of the matrix $[[1,2,-1],[-2,0,1],[1,-1,0]]$ ?"
#We call initiate_chat to start the conversation.
#When setting message=mathproxyagent.message_generator, you need to pass in the problem through the problem parameter.
#mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)
with Cache.disk() as cache:
    mathproxyagent.initiate_chat(assistant, message=math_problem, cache=cache)

