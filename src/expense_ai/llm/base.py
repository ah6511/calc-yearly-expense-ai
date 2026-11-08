"""Protocol for structured LLM completion clients."""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Protocol for structured output LLM completion."""

    def create_completion(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> T:
        """Create a structured completion parsed into `response_model`."""
        ...
