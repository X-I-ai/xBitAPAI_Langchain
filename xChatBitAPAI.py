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
    Transforms a list of messages into a conversation format. It's like turning raw thoughts into a meaningful dialogue.

    Args:
        messages: A list of messages, like the thoughts in your head.
        errors: How we handle errors. We can "raise" a fuss, "warn" you about it, or "ignore" it like it never happened.

    Returns:
        A list of dictionaries representing the conversation. It's like a transcript of a chat.
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
    This is the Admiral KΞRC's LangChain component for Bittensor's new API. It's like a translator between you and the BitAPAI chat model.

    Attributes:
        uids: [...]
        max_retries: How many times we'll try again when generating. It's like our patience level.
        generations_count: How many chat completions we'll generate for each prompt. It's like our productivity level.
        errors: How we handle errors. We can "raise" a fuss, "warn" you about it, or "ignore" it like it never happened.
        return_all: Whether to return all results or not. It's like choosing between a buffet and a set menu.
        pool_id: [...]
    """

    uids: str | int | List = Field(default=[], alias="model")
    max_retries: int = 0
    generations_count: int = 1
    errors: Literal["raise", "warn", "ignore"] = "warn"
    return_all: bool = True
    pool_id: int = 0

    bitAPAI_key = os.environ.get("BITAPAI_API_KEY", "Ξ")
    if bitAPAI_key == "Ξ":
        raise KeyError("Environment variable BITAPAI_KEY not found.")

    class Config:
        allow_population_by_field_name = True

    def _call(
        self,
        messages: List[BaseMessage] | BaseMessage,
        uids: str | int | List = [],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        return_all: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the BitAPAI API. It's like dialing a friend.

        Args:
            messages: A list of messages or a single message. It's like what you want to say.
            uids: [...]
            run_manager: A manager for the run. It's like a supervisor.
            return_all: Whether to return all results or not. It's like choosing between a buffet and a set menu.
            **kwargs: Additional keyword arguments. It's like extra toppings. You can

        Returns:
            The response from the API. It's like what your friend says back.
        """
        #
        self.return_all = return_all if return_all else self.return_all

        self.uids = self._safe_uids_list(self.uids) + self._safe_uids_list(uids)

        if "pool_id" in kwargs.keys():
            self.pool_id = kwargs["pool_id"]
        
        #
        BitAPAI_API_payload = {
            "conversation": _lcmessages_to_conversation(
                messages if isinstance(messages, List) else [messages],
                errors=self.errors,
            ),
            "uids": self.uids,
            "count": self.generations_count,
            # "pool_id": self.pool_id,
            "return_all": return_all,
        }

        #
        payload = json.dumps(BitAPAI_API_payload)

        headers = {"Content-Type": "application/json", "X-API-KEY": self.bitAPAI_key}

        #
        conn = http.client.HTTPSConnection("api.bitapai.io")

        conn.request("POST", "/text", payload, headers)

        res = conn.getresponse()

        data = res.read()



        #
        try:
            data_dict = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode JSON response from BitAPAI: {e}\n\nReceived: {data.decode()}")
            data_dict = {}

        #
        final_output = {}
        for i, each_response_data in enumerate(data_dict["response_data"]):
            final_output[f"{i}"] = each_response_data.get("response")

        # -- >
        return json.dumps(final_output) or str(data_dict)
        # -- >


    def _safe_uids_list(
        self, content: Union[List[Union[str, int]], str, int]
    ) -> List[int]:
        """
        Convert the content into a list of integers. It's like turning words into numbers.

        Args:
            content: The content to convert. It's like the words.

        Returns:
            A list of integers. It's like the numbers.
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
        """Get the default parameters for calling OpenAI API. It's like the standard settings."""
        return {
            "uids": self.uids,
            "count": self.generations_count,
            "pool_id": self.pool_id,
            "max_retries": self.max_retries,
            "return_all": self.return_all,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters. It's like the model's ID card."""
        return {**{"model_name": self.uids}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model. It's like the model's species."""
        return "BitAPAI-chat"


