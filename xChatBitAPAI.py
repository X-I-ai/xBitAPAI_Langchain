import os
import http.client
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Literal, Union
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ChatMessage,
    FunctionMessage,
    SystemMessage,
)
from pydantic import Field

load_dotenv()

logger = logging.getLogger(__name__)


def _lcmessages_to_conversation(
    messages: List[BaseMessage], errors="warn"
) -> List[Dict[str, str]]:
    """
    Convert a list of messages into a conversation format.

    Args:
        messages: A list of messages.
        errors: Error handling strategy. Options are "raise", "warn", and "ignore".

    Returns:
        A list of dictionaries representing the conversation.
    """
    conversation: List[Dict[str, str]] = []

    for message in messages:
        message_dict = None

        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "prompt": message.content}

        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "prompt": message.content}

        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "prompt": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs[
                    "function_call"
                ]

        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "prompt": message.content}

        elif isinstance(message, FunctionMessage):
            match errors:
                case "raise":
                    raise ValueError(f"FunctionMessages not supported in BitAPAI.")
                case "warn":
                    logger.warning(f"FunctionMessages not supported in BitAPAI.")
                case _:
                    continue
        else:
            match errors:
                case "raise":
                    raise ValueError(f"Got unknown type {message}")
                case "warn":
                    logger.warning(f"Got unknown type {message}")
                case _:
                    continue

        if message_dict:
            conversation.append(message_dict)

    return conversation


class xChatBitAPAI(SimpleChatModel):
    """
    A class to interact with the BitAPAI chat model.

    Attributes:
        uids: Unique identifiers for the model.
        max_retries: Maximum number of retries to make when generating.
        generations_count: Number of chat completions to generate for each prompt.
        errors: Error handling strategy. Options are "raise", "warn", and "ignore".
        return_all: Whether to return all results or not.
        pool_id: Identifier for the pool.
    """

    uids: str | int | List = Field(default=[], alias="model")
    max_retries: int = 0
    generations_count: int = 1
    errors: Literal["raise", "warn", "ignore"] = "warn"
    return_all: bool = False
    pool_id: int | None = None

    bitAPAI_key = os.environ.get("BITAPAI_API_KEY", "Ξ")
    if bitAPAI_key == "Ξ":
        raise KeyError("Environment variable BITAPAI_KEY not found.")

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    def _call(
        self,
        messages: List[BaseMessage] | BaseMessage,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the BitAPAI API.

        Args:
            messages: A list of messages or a single message.
            uids: Unique identifiers for the model. [...]
            run_manager: A manager for the run.
            **kwargs: Additional keyword arguments such as:
                - return_all: Whether to return all results or not.
                [...]

        Returns:
            The response from the API.
        """


        BitAPAI_API_payload = {
            "conversation": _lcmessages_to_conversation(
                messages if isinstance(messages, List) else [messages],
                errors=self.errors,
            ),
            "uids": self._safe_uids_list(kwargs.get("uids", self.uids)),
            "count": kwargs.get("generations_count", self.generations_count),
            "pool_id": kwargs.get("pool_id", self.pool_id),
            "return_all": kwargs.get("return_all", self.return_all),
        }
        payload = json.dumps(BitAPAI_API_payload)

        headers = {"Content-Type": "application/json", "X-API-KEY": self.bitAPAI_key}
        conn = http.client.HTTPSConnection("api.bitapai.io")

        conn.request("POST", "/text", payload, headers)

        res = conn.getresponse()

        data = res.read()

        data_dict = json.loads(data.decode("utf-8"))
        
        final_output = {}

        for i, each_response_data in enumerate(data_dict["response_data"]):
            final_output[f"{i}"] = each_response_data.get("response")

        return json.dumps(final_output)

    def _safe_uids_list(
        self, content: Union[List[Union[str, int]], str, int]
    ) -> List[int]:
        """
        Convert the content into a list of integers.

        Args:
            content: The content to convert.

        Returns:
            A list of integers.
        """
        if not isinstance(content, List):
            content = [content]

        int_list = []
        for c in content:
            if isinstance(c, int):
                int_list.append(c)
            elif isinstance(c, str):
                try:
                    int_list.append(int(c))
                except ValueError:
                    if self.errors == "raise":
                        raise ValueError(f"Cannot convert {c} to int.")
                    elif self.errors == "warn":
                        logger.warning(f"Cannot convert {c} to int. Ignoring.")

        return int_list

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "uids": self.uids,
            "count": self.generations_count,
            "pool_id": self.pool_id,
            "max_retries": self.max_retries,
            "return_all": self.return_all,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.uids}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "BitAPAI-chat"


