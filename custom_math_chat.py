from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import autogen
from autogen import AssistantAgent, UserProxyAgent