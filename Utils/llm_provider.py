from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BEDROCK = "bedrock"