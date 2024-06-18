import autogen
from autogen.cache import Cache
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from typing_extensions import Annotated
from typing import List
import numpy as np

#define the matrix to invert
matrix = [[1, 2, -1], [-2, 0, 1], [1, -1, 0]]

#define the configuration list for autogen
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={"tags": ["tool"]} 
)

#define the function to find the inverse of a matrix
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

#define the tool configuration for the assistant
functions = [{
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
}]

#create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="For coding tasks, execute the provided function to find the inverse of a matrix.",
    llm_config={
        "functions": functions,
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

#create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    system_message="A proxy for the user that is used to execute code and correct the assistant if the assistant makes a calculation mistake. If the assistant finds the incorrect inverse, make the assistant recalculate the inverse.",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)


#register the function for execution in MathUserProxyAgent
@mathproxyagent.register_for_execution()
def matrix_execute(matrix: Annotated[List[List[int]], "A 3 by 3 matrix with integers."]) -> List[List[float]]:
    inversed_matrix = find_inverse_matrix(matrix)
    return inversed_matrix

#given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
math_problem = "What is the inverse of the matrix $[[1,2,-1],[-2,0,1],[1,-1,0]]$ ?"

#initiate the conversation using MathUserProxyAgent
mathproxyagent.initiate_chat(assistant, message=math_problem)

correct_output = find_inverse_matrix([[1,2,-1],[-2,0,1],[1,-1,0]])
print(f"The correct inverse: {correct_output}")

