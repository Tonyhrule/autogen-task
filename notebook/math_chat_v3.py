import autogen
from autogen.cache import Cache
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from typing_extensions import Annotated
from typing import List
import numpy as np
from math_utils import get_answer, check_inverse_correctness  # Import get_answer and check_inverse_correctness from math_utils

# Define the matrix to invert
matrix = [[1, 2, -1], [-2, 0, 1], [1, -1, 0]]

# Define the configuration list for autogen
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={"tags": ["tool"]}
)

# Define the function to find the inverse of a matrix
def find_inverse_matrix(matrix) -> list:
    np_matrix = np.array(matrix)
    identity_matrix = np.linalg.inv(np_matrix)
    return identity_matrix.tolist()

# Create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="For coding tasks, execute the provided function to find the inverse of a matrix.",
    llm_config={
        "functions": [{
            "name": "inverse_matrix",
            "description": "Finds the inverse of a given matrix",
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "description": "A matrix that needs to be inversed.",
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
        }],
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

# Create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    system_message="A proxy for the user that is used to execute code and correct the assistant if the assistant makes a calculation mistake.",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

# Register the function for execution in MathUserProxyAgent
@mathproxyagent.register_for_execution()
def matrix_execute(matrix: Annotated[List[List[float]], "A 3 by 3 matrix with integers."]):
    inversed_matrix = find_inverse_matrix(matrix)
    # Check correctness
    expected_inverse = np.linalg.inv(np.array(matrix)).tolist()
    is_correct = np.allclose(expected_inverse, inversed_matrix)
    return inversed_matrix, is_correct

# Given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
math_problem = "What is the inverse of the matrix $[[1,2,-1],[-2,0,1],[1,-1,0]]$ ?"

# Initiate the conversation using MathUserProxyAgent
with Cache.disk() as cache:
    chat_result = mathproxyagent.initiate_chat(assistant, message=math_problem, cache=cache)

# Process the assistant's response
assistant_message = chat_result.message
inverse_matrix = assistant_message[0]['result']
is_correct = assistant_message[0]['correct']

# Print the inverse matrix and correctness status
print("Assistant's Inverse Matrix:")
print(inverse_matrix)
print("Is Correct:", is_correct)

# Example of using get_answer from math_utils
question = "What is the answer to life, the universe, and everything?"
answer = get_answer(question)
print("Answer:", answer)
