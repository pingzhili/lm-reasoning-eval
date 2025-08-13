# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_litellm_available
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

if is_litellm_available():
    import litellm
    from litellm import encode
    from litellm.caching.caching import Cache
    from litellm.utils import Choices, Message, ModelResponse as LitellmModelResponse

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type="disk")
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()

class LiteLLMModelConfig(ModelConfig):
    """
    Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).

    Example:
        ```python
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model_name,
            model_sha="",
            model_dtype=None,
            model_size=-1,
        )
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.reasoning_effort = self.generation_parameters.reasoning_effort

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = 10  # 100 leads to hitting Anthropic rate limits

        self._tokenizer = encode
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.set_verbose = False
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens):
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if "o1" in self.model:
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, 32000)
        return max_new_tokens

    def call_litellm_completion_with_thinking_budget(self, thinking_budget, thinking_terminate_token="<|end|>",
                                                     **kwargs):
        """Call LiteLLM completion with thinking budget support."""
        # Extract key parameters
        initial_messages = kwargs["messages"]

        # First completion: Generate thinking with budget constraint
        context = self.hf_tokenizer.apply_chat_template(
            initial_messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
        )
        first_output = litellm.text_completion(
            model=kwargs["model"],
            prompt=context,
            max_tokens=thinking_budget,
            stop=thinking_terminate_token,
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            api_key="nan",
            base_url=self.base_url,
            extra_body={"skip_special_tokens": False},
        )

        finish_reason = first_output.choices[0].finish_reason
        progress_text = first_output.choices[0].text
        num_repeat_steps = 0
        max_repeat_steps = 3

        while finish_reason != "length":
            # if not thinking enough, go ahead
            # remove thinking end
            progress_steps = progress_text.split("\n\n")
            if len(progress_steps) > 1 and progress_steps[-1] == progress_steps[-2]:
                num_repeat_steps += 1
            else:
                num_repeat_steps = 0

            progress_text = "\n\n".join(progress_steps) + "\n\nWait, "

            if num_repeat_steps >= max_repeat_steps:
                logger.warning(f"STOP thinking due to {num_repeat_steps} repeat steps of {progress_steps[-1]}.")
                break

            iter_output = litellm.text_completion(
                model=kwargs["model"],
                prompt=context + progress_text,
                max_tokens=thinking_budget,
                stop=thinking_terminate_token,
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                api_key="nan",
                base_url=self.base_url,
                extra_body={"skip_special_tokens": False},
            )
            progress_text = progress_text + iter_output.choices[0].text
            finish_reason = iter_output.choices[0].finish_reason

        # reach thinking budget
        progress_text = "\n\n".join(progress_text.split("\n\n")[:-1])
        thinking_text = progress_text.split("<|channel|>analysis<|message|>")[-1]

        initial_messages.append({
            "role": "assistant", "thinking": thinking_text, "content": "[CONTENT_PLACEHOLDER]"
        })
        final_prompt = self.hf_tokenizer.apply_chat_template(
            initial_messages,
            reasoning_effort=self.reasoning_effort,
            tokenize=False,
            continue_final_message=True
        )
        final_prompt.replace("[CONTENT_PLACEHOLDER]", "")
        final_output = litellm.text_completion(
            model=kwargs["model"],
            prompt=final_prompt,
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            api_key="nan",
            base_url=self.base_url,
            extra_body={"skip_special_tokens": False},
        )

        response = LitellmModelResponse(
            choices=[
                Choices(
                    finish_reason='stop',
                    index=0,
                    message=Message(
                        reasoning_content=thinking_text,
                        content=final_output.choices[0].text,
                        role='assistant'
                    )
                )
            ]
        )
        logger.info(response)
        return response

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901
        """Make API call with retries."""
        response = LitellmModelResponse()
        for attempt in range(self.API_MAX_RETRY):
            try:
                stop_sequence = self._prepare_stop_sequence(stop_sequence)
                max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

                if return_logits and not self.provider == "openai":
                    logger.warning("Returning logits is not supported for this provider, ignoring.")

                # Prepare kwargs for completion call
                kwargs = {
                    "model": self.model,
                    "messages": prompt,
                    "logprobs": return_logits if self.provider == "openai" else None,
                    "base_url": self.base_url,
                    "n": num_samples,
                    "caching": False,  # We don't want caching with same response
                    "api_key": self.api_key,
                }

                if num_samples > 1 and self.generation_parameters.temperature == 0:
                    raise ValueError(
                        "num_samples > 1 but temperature is set to 0, this will not sample different outputs."
                    )

                if "o1" in self.model:
                    logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
                else:
                    kwargs.update(self.generation_parameters.to_litellm_dict())
                    logger.info(f"Generation kwargs: {kwargs}")

                if kwargs.get("max_completion_tokens", None) is None:
                    kwargs["max_completion_tokens"] = max_new_tokens

                if "reasoning_effort" in kwargs:
                    kwargs["extra_body"] = {"reasoning_effort": kwargs["reasoning_effort"]}
                    del kwargs["reasoning_effort"]

                if self.generation_parameters.thinking_budget is None:
                    response = litellm.completion(**kwargs)
                else:
                    response = self.call_litellm_completion_with_thinking_budget(
                        thinking_budget=self.generation_parameters.thinking_budget,
                        **kwargs,
                    )

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if response.choices[0].message.content is None:
                    kwargs["caching"] = False
                    logger.info("Response is empty, retrying without caching")
                    response = litellm.completion(**kwargs)
                return response
            except litellm.BadRequestError as e:
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2 ** attempt))  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return LitellmModelResponse()

    def __call_api_parallel(
            self,
            prompts,
            return_logits: bool | list[bool],
            max_new_tokens: int | list[int] | None,
            num_samples: int | list[int],
            stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
                len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"
        )

        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            for entry in tqdm(
                    executor.map(
                        self.__call_api,
                        prompts,
                        return_logitss,
                        max_new_tokenss,
                        num_sampless,
                        stop_sequencess,
                    ),
                    total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def greedy_until(
            self,
            docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
                dataset.splits_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response, context in zip(responses, contexts):
                result: list[str] = [choice.message.content for choice in response.choices]
                reasoning_result: list[str] = [choice.message.reasoning_content for choice in response.choices]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    reasoning_text=reasoning_result if reasoning_result[0] else [""],
                    input=context,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
