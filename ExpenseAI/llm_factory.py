import os
import instructor
import boto3

from typing import Any, Dict, List, Type
from  ExpenseAI.config.settings import get_settings
from openai import OpenAI
from pydantic import BaseModel, Field

class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
            "bedrock": lambda s: instructor.from_bedrock(
                boto3.client(region_name=os.environ["AWS_REGION"],
                        service_name='bedrock-runtime',
                        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                        aws_session_token=os.environ["AWS_SESSION_TOKEN"])
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "response_model": response_model,
            "messages": messages,
        }

        if self.settings.default_model.startswith("gpt-5"):
            completion_params["max_completion_tokens"] = kwargs.get("max_tokens", self.settings.max_tokens)
        else:
            completion_params["max_tokens"] = kwargs.get("max_tokens", self.settings.max_tokens)

        return self.client.chat.completions.create(**completion_params)

if __name__ == "__main__":

    class CompletionModel(BaseModel):
        response: str = Field(description="Your response to the user.")
        reasoning: str = Field(description="Explain your reasoning for the response.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "If it takes 2 hours to dry 1 shirt out in the sun, how long will it take to dry 5 shirts?",
        },
    ]

    llm = LLMFactory("bedrock")
    completion = llm.create_completion(
        response_model=CompletionModel,
        messages=messages,
    )
    assert isinstance(completion, CompletionModel)

    print(f"Response: {completion.response}\n")
    print(f"Reasoning: {completion.reasoning}")