
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import logging
logger = logging.getLogger(__name__)

class HFTGIChat(LLM):
    """A custom chat model that queries the chat API of HuggingFace Text Generation Inference

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.
    """

    client: OpenAI
    timeout: int = 120
    model_name: str
    stop: Optional[List[str]] = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        temperature = kwargs.get("temperature", 0.2)
        system_prompt = "You are a helpful assistant in solving various tasks. You should only output one answer to the task, nothing more, no explanations, and nothing around. Just read the instruction carefully, understand the positive and negative examples provided, and generate one single answer in the same way as the example's output."
        if "mixtral" in self.model_name.lower() or "gemma" in self.model_name.lower(): # mixtral 8x7b / 8x22b and Gemma 7b have no system prompt
            messages = [
                {"role": "user", "content": system_prompt + "\n" + prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        chat_completion = self.client.chat.completions.create(
            model="tgi",
            messages=messages,
            stop=stop or self.stop,
            stream=False,
            temperature=temperature
        )

        return chat_completion.choices[0].message.content or ""

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        chat_completion = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": "You are a helpful assistant." },
                {"role": "user", "content": prompt}
            ],
            stop=stop or self.stop,
            stream=True
        )

        # iterate and yield stream
        for message in chat_completion:
            yield message

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


if __name__ == '__main__':
    endpoint_url = "http://127.0.0.1:8080" # No trailing slash
    client = OpenAI(
        base_url=f"{endpoint_url}/v1",
        api_key="-"
    )
    llm = HFTGIChat(
        client=client,
    )
    # For LLaMA 3
    print(llm.invoke("What is deep learning? Write only one sentence!", stop=[
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|reserved_special_token|>"
    ]))
    print(llm.batch(["woof woof woof", "meow meow meow"], stop=[
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|reserved_special_token|>"
    ]))
    # print(llm.invoke("This is a foobar thing"))
    # print(llm.batch(["woof woof woof", "meow meow meow"]))
