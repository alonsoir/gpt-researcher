import json
import os
import logging
import time
from typing import Sequence, List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall
import nest_asyncio
from dotenv import load_dotenv

# Apply necessary async adjustments
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="ollama_agent.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


# Define computational tools for the agent
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and return the result"""
    return a + b


# Create FunctionTool instances from defined functions
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)


# Define the custom OpenAI agent class
class YourOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(
            temperature=0,
            model="gpt-3.5-turbo-0613",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        """Reset the conversation history"""
        self._chat_history = []

    def chat(self, message: str) -> str:
        """Process a message through the agent, managing tool invocations and responses"""
        global ai_message
        self._chat_history.append(ChatMessage(role="user", content=message))
        tools = [tool.metadata.to_openai_tool() for _, tool in self._tools.items()]

        # Retry logic with exponential backoff
        max_retries = 5  # Set the maximum number of retries
        retry_delay = 1  # Initial retry delay in seconds
        for attempt in range(max_retries + 1):
            try:
                ai_message = self._llm.chat(self._chat_history, tools=tools).message
                self._chat_history.append(ai_message)
                break
            except ConnectionError as e:
                logger.error(f"APIConnectionError on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise  # Raise the error after the maximum retries
                time.sleep(retry_delay)
                retry_delay *= 2  # Increase retry delay exponentially

        # Handle parallel function calling
        tool_calls = ai_message.additional_kwargs.get("tool_calls", [])
        for tool_call in tool_calls:
            function_message = self._call_function(tool_call)
            self._chat_history.append(function_message)
            ai_message = self._llm.chat(self._chat_history).message
            self._chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call: ChatCompletionMessageToolCall) -> ChatMessage:
        """Invoke a function based on a tool call"""
        tool = self._tools[tool_call.function.name]
        output = tool(**json.loads(tool_call.function.arguments))
        return ChatMessage(
            name=tool_call.function.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
            },
        )


if __name__ == "__main__":
    load_dotenv()
    # Create and test the agent
    agent = YourOpenAIAgent(tools=[multiply_tool, add_tool])
    print(agent.chat("Hi"))  # Expected response:
    print(agent.chat("Multiply 3 and 5"))  # Expected response: "15"
