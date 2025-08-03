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

import gc
import itertools
import logging
import os
from typing import Optional

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import ModelConfig, _simplify_name
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_vllm_available


logger = logging.getLogger(__name__)


if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.transformers_utils.tokenizer import get_tokenizer

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    from unittest.mock import Mock

    LLM = SamplingParams = get_tokenizer = ray = distribute = destroy_distributed_environment = (
        destroy_model_parallel
    ) = RequestOutput = Mock()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class VLLMModelConfig(ModelConfig):
    """
    Configuration class for VLLM inference engine.

    This configuration is used to load and configure models using the VLLM inference engine,
    which provides high-performance inference for large language models with features like
    PagedAttention, continuous batching, and efficient memory management.

    vllm doc: https://docs.vllm.ai/en/v0.7.1/serving/engine_args.html

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to the model to load.
        revision (str):
            Git revision of the model. Defaults to "main".
        dtype (str):
            Data type for model weights. Defaults to "bfloat16". Options: "float16", "bfloat16", "float32".
        tensor_parallel_size (PositiveInt):
            Number of GPUs to use for tensor parallelism. Defaults to 1.
        data_parallel_size (PositiveInt):
            Number of GPUs to use for data parallelism. Defaults to 1.
        pipeline_parallel_size (PositiveInt):
            Number of GPUs to use for pipeline parallelism. Defaults to 1.
        gpu_memory_utilization (NonNegativeFloat):
            Fraction of GPU memory to use. Lower this if running out of memory. Defaults to 0.9.
        max_model_length (PositiveInt | None):
            Maximum sequence length for the model. If None, automatically inferred.
            Reduce this if encountering OOM issues (4096 is usually sufficient).
        quantization (str | None):
            Quantization method.
        load_format (str | None):
            The format of the model weights to load. choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer.
        swap_space (PositiveInt):
            CPU swap space size in GiB per GPU. Defaults to 4.
        seed (NonNegativeInt):
            Random seed for reproducibility. Defaults to 1234.
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        multichoice_continuations_start_space (bool):
            Whether to add a space before multiple choice continuations. Defaults to True.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately for loglikelihood evals. Defaults to False.
        max_num_seqs (PositiveInt):
            Maximum number of sequences per iteration. Controls batch size at prefill stage. Defaults to 128.
        max_num_batched_tokens (PositiveInt):
            Maximum number of tokens per batch. Defaults to 2048.
        subfolder (str | None):
            Subfolder within the model repository. Defaults to None.
        use_chat_template (bool):
            Whether to use chat templates for conversation-style prompts. Defaults to False.
        hf_overrides (dict):
            Overrides for HuggingFace model configuration. Defaults to empty dict.

    Example:
        ```python
        config = VLLMModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_length=4096,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    revision: str = "main"  # revision of the model
    dtype: str = "bfloat16"
    tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    pipeline_parallel_size: PositiveInt = 1  # how many GPUs to use for pipeline parallelism
    gpu_memory_utilization: NonNegativeFloat = 0.9  # lower this if you are running out of memory
    max_model_length: PositiveInt | None = (
        None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    )
    quantization: str | None = None
    load_format: str | None = None
    swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
    seed: NonNegativeInt = 1234
    trust_remote_code: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    max_num_seqs: PositiveInt = 128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
    max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
    subfolder: str | None = None
    use_chat_template: bool = False

    enable_thinking: bool = True  # if you can think, think
    self_judge_thinking: bool = False  # whether to let model self-judge if thinking is needed per question
    chat_template: str | None = None

    hf_overrides: dict = None


class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = config.data_parallel_size
        self.tensor_parallel_size = config.tensor_parallel_size
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config)

        # Store self-judgments for evaluation details
        self._self_judgments = {}

        self._max_length = config.max_model_length if config.max_model_length is not None else None

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.model_name)
        self.model_sha = ""
        self.precision = config.dtype

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.pairwise_tokenization = config.pairwise_tokenization

        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt,
            enable_thinking=config.enable_thinking,
            chat_template=config.chat_template,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            del self.model
        gc.collect()
        ray.shutdown()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: VLLMModelConfig) -> Optional[LLM]:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "hf_overrides": dict(config.hf_overrides) if config.hf_overrides is not None else None,
        }

        if config.quantization is not None:
            self.model_args["quantization"] = config.quantization
        if config.load_format is not None:
            self.model_args["load_format"] = config.load_format

        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"
            return None

        if self.model_args["hf_overrides"] is not None and len(self.model_args["hf_overrides"]) > 0:
            logger.info(f"Using HF overrides: {self.model_args['hf_overrides']}")

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def _create_auto_tokenizer(self, config: VLLMModelConfig):
        tokenizer = get_tokenizer(
            config.model_name,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _greedy_until_with_thinking_budget(
        self,
        docs: list[Doc],
        thinking_budget: int,
    ) -> list[ModelResponse]:
        """
        Generates responses with thinking budget constraint.

        Args:
            docs: list of documents to generate responses for
            thinking_budget: maximum tokens allowed for thinking

        Returns:
            list of model responses
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,
        ):
            for doc in tqdm(split, desc="Generating responses"):
                # Process each doc individually for thinking budget tracking
                max_new_tokens = self._config.generation_parameters.max_new_tokens or doc.generation_size
                returns_logits = self._config.generation_parameters.returns_logits
                num_samples = doc.num_samples
                assert num_samples == 1, f"Found num_samples = {num_samples} for {doc}"

                # Prepare the initial prompt
                context = self.prompt_manager.prepare_prompt(doc)
                tokenized = self.tokenizer([context], add_special_tokens=self.add_special_tokens)
                inputs = tokenized["input_ids"]

                # Track thinking tokens and steps
                thinking_tokens = 0
                thinking_steps = []
                historical_step_tokens = []
                current_input = inputs[0]
                last_wait = False

                # Generate thinking steps with "\n\n" as stop token
                while True:
                    # Generate one thinking step
                    vllm_output = self._generate(
                        inputs=[current_input],
                        stop_tokens=["\n\n"],
                        returns_logits=returns_logits,
                        num_samples=1,  # Force single sample for thinking phase
                        use_tqdm=False
                    )[0]

                    step_text = vllm_output.outputs[0].text
                    num_step_tokens = len(vllm_output.outputs[0].token_ids)
                    historical_step_tokens.append(num_step_tokens)
                    thinking_tokens += num_step_tokens

                    if last_wait:
                        step_text = "Wait, " + step_text

                    # Check if we've hit the budget
                    if thinking_tokens >= thinking_budget:
                        # Complete current step and add closing tag
                        thinking_steps.append(step_text)
                        thinking_text = "\n\n".join(thinking_steps)
                        thinking_text += "\n</think>\n\n"
                        break

                    # Check if thinking naturally ended (e.g., model generated </think>)
                    # Enforce to continue
                    if "</think>" in step_text:
                        thinking_steps.append(step_text.split("</think>")[0])
                        thinking_text = "\n\n".join(thinking_steps) + "\n\n"
                        next_prompt = context + thinking_text + "Wait, "
                        last_wait = True
                    else:
                        # Continue thinking
                        thinking_steps.append(step_text)
                        thinking_text = "\n\n".join(thinking_steps) + "\n\n"
                        next_prompt = context + thinking_text
                        last_wait = False

                    current_input = self.tokenizer([next_prompt], add_special_tokens=False)["input_ids"][0]

                logger.info(f"Stop thinking with historical step tokens (used/budget={thinking_tokens}/{thinking_budget}): {historical_step_tokens}")

                # Prepare final prompt with completed thinking
                final_prompt = context + thinking_text
                final_input = self.tokenizer([final_prompt], add_special_tokens=False)["input_ids"][0]

                # Generate final answer without stop tokens
                final_output = self._generate(
                    inputs=[final_input],
                    max_new_tokens=max_new_tokens,
                    returns_logits=returns_logits,
                    num_samples=1,
                    use_tqdm=False,
                )[0]

                # Construct full response
                full_text = thinking_text + final_output.outputs[0].text

                # Collect all output tokens (from thinking + final answer)
                for step in thinking_steps:
                    # We don't have the exact tokens from thinking steps, so we'll use the final output tokens
                    # In a more complete implementation, we'd store tokens from each thinking step
                    pass

                # Create response object
                output_tokens = None
                output_logprobs = None
                if returns_logits:
                    raise NotImplementedError("returns_logits not implemented for budget thinking")

                cur_response = ModelResponse(
                    input=context,
                    text=[output.text for output in final_output.outputs] if num_samples > 1 else [full_text],
                    output_tokens=output_tokens,
                    input_tokens=inputs[0],
                    logprobs=output_logprobs,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

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
            list[GenerateReturn]: list of generated responses.
        """
        # Check if we should use thinking budget
        thinking_budget = self._config.generation_parameters.thinking_budget
        if thinking_budget > 0 and self.prompt_manager.enable_thinking:
            logger.info(f"Using thinking budget of {thinking_budget}.")
            return self._greedy_until_with_thinking_budget(docs, thinking_budget)

        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = split[0].stop_sequences or []

            max_new_tokens = self._config.generation_parameters.max_new_tokens or split[0].generation_size
            returns_logits = self._config.generation_parameters.returns_logits
            num_samples = split[0].num_samples

            context = [self.prompt_manager.prepare_prompt(doc) for doc in split]
            tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)

            # The main question for this step is the following:
            # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
            # of losing some meaning, or have some generations that are exceedingly short?
            # The choice we go for here is to avoid truncating the prompt if we can, since it
            # should have been managed by the prompt creator/few shot manager if requested by the user.
            inputs = tokenized["input_ids"]
            context_size = len(inputs[0])

            # left truncate the inputs to the maximum length
            if max_new_tokens is not None:
                if context_size + max_new_tokens > self.max_length:
                    logger.warning(
                        f"{context_size + max_new_tokens=} which is greater than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
                    )
                    context_size = self.max_length - max_new_tokens
                    if context_size < 0:
                        logger.critical(
                            f"{context_size=} is less than 0, either reduce the max_new_tokens or increase model max length."
                        )
                        raise ValueError("Context size is less than 0.")
                    inputs = [input[-context_size:] for input in inputs]
            else:
                if context_size > self.max_length:
                    logger.warning(
                        f"{context_size=} which is greater than {self.max_length=}. Truncating context to {self.max_length} tokens."
                    )
                    context_size = self.max_length
                    inputs = [input[-context_size:] for input in inputs]

            vllm_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=returns_logits,
                num_samples=num_samples,
            )

            for i, vllm_output in enumerate(vllm_outputs):
                output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                output_logprobs = [
                    [
                        {
                            str(idx): {"logprob": lp.logprob, "rank": lp.rank, "decoded_token": lp.decoded_token}
                            for idx, lp in token_logprobs.items()
                            if lp is not None and hasattr(lp, "logprob") and lp.logprob is not None
                        }
                        for token_logprobs in outputs.logprobs
                    ]
                    for outputs in vllm_output.outputs
                ]
                result = [output.text for output in vllm_output.outputs]
                input_token_ids = vllm_output.prompt_token_ids

                cur_response = ModelResponse(
                    input=context[i],
                    text=result,
                    output_tokens=list(output_token_ids),
                    input_tokens=input_token_ids,
                    logprobs=output_logprobs if returns_logits else None,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    # Self-judging thinking prompt template
    THINKING_JUDGE_TEMPLATE = """Analyze the following question and determine if it requires very long step-by-step thinking for you to solve correctly:

Question: {question}

Does this question require very long, complex thinking? Answer with only 'YES' or 'NO'."""

    def _create_judge_doc(self, doc: Doc) -> Doc:
        """Create a judgment doc from the original doc."""
        judge_prompt = self.THINKING_JUDGE_TEMPLATE.format(question=doc.query)

        # Create a new doc for judgment with the same structure but different query
        judge_doc = Doc(
            query=judge_prompt,
            choices=["YES", "NO"],  # For judgment, we only need these two options
            gold_index=0,  # Dummy gold index
            task_name=doc.task_name,
            instruction=doc.instruction,
            unconditioned_query="",
            num_asked_few_shots=0,  # No few-shot for judgment
            num_effective_few_shots=0,
            original_query=doc.original_query,
            id=doc.id,
            num_samples=1,  # Always single sample for judgment
            generation_size=10,  # Small size for YES/NO
            stop_sequences=doc.stop_sequences,
        )
        return judge_doc

    def _parse_thinking_judgment(self, response_text: str) -> bool:
        """Parse the judgment response to determine if thinking is needed."""
        # Extract the first generated text (in case of multiple samples)
        if isinstance(response_text, list):
            response_text = response_text[0] if response_text else ""

        # Clean and normalize the response
        response_text = response_text.strip().upper()

        # Check if the response contains YES
        return "YES" in response_text

    def greedy_until_self_judge(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Two-step generation process:
        1. Ask model if thinking is needed for each question
        2. Generate answer with or without thinking based on judgment
        """
        if not self._config.self_judge_thinking:
            # If self-judging is disabled, use the regular greedy_until
            return self.greedy_until(docs)

        # Step 1: Create judgment requests
        judge_docs = [self._create_judge_doc(doc) for doc in docs]

        # Generate judgments
        judge_responses = self.greedy_until(judge_docs)

        # Step 2: Prepare for actual generation with dynamic thinking
        # We need to temporarily modify the prompt manager's enable_thinking for each request
        results = []

        # Process each doc individually based on its judgment
        for doc, judge_response in zip(docs, judge_responses):
            needs_thinking = self._parse_thinking_judgment(judge_response.text)

            # Temporarily set enable_thinking based on judgment
            original_enable_thinking = self.prompt_manager.enable_thinking
            self.prompt_manager.enable_thinking = needs_thinking

            # Generate the actual response
            # If thinking budget is set and thinking is needed, use the budget-aware method
            thinking_budget = self._config.generation_parameters.thinking_budget
            if thinking_budget > 0 and needs_thinking:
                response = self._greedy_until_with_thinking_budget([doc], thinking_budget)[0]
            else:
                response = self.greedy_until([doc])[0]

            # Log the self-judging decision
            logger.info(f"Doc {doc.id}: Self-judged thinking = {needs_thinking}")

            # Store the judgment for later use in metrics
            self._self_judgments[doc.id] = needs_thinking

            results.append(response)

            # Restore original enable_thinking
            self.prompt_manager.enable_thinking = original_enable_thinking

        return results

    def _generate(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
        use_tqdm: bool = True,
    ) -> list:
        """Contains the actual logic of the generation."""
        # remove "returns_logits" from arg dict to SamplingParams
        sampling_params_args = {
            k: v for k, v in self._config.generation_parameters.to_vllm_dict().items() if k != "returns_logits"
        }
        sampling_params = SamplingParams(**sampling_params_args)
        # sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())

        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 10 if returns_logits else 0
            if num_samples > 1 and sampling_params.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            # Hynek: With the newest vllm, it actually breaks when tensor_parallel_size == 1 and num_gpus not set,
            # as VLLM complains about no GPUs available.
            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(model_args: dict, sampling_params: SamplingParams, requests):
                llm = LLM(**model_args)
                return llm.generate(prompt_token_ids=requests, sampling_params=sampling_params)

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, inputs)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            outputs = [
                x
                for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
                if x is not None
            ]
        else:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )

        return outputs

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        return self._loglikelihood_tokens(docs)

    def _loglikelihood_tokens(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=1)
        res = []

        for split in tqdm(dataset.splits_iterator()):
            contexts = [self.prompt_manager.prepare_prompt(doc) for doc in split]

            inputs = []
            tokenized_continuations_batch = []
            tokenized_contexts_batch = []

            for context, doc in zip(contexts, split):
                tokenized_contexts, tokenized_continuations = self.tok_encode_pair(
                    context, doc.choices, pairwise=self.pairwise_tokenization
                )
                for tokenized_context, tokenized_continuation in zip(tokenized_contexts, tokenized_continuations):
                    inputs.append(tokenized_context + tokenized_continuation)
                    tokenized_continuations_batch.append(tokenized_continuation)
                    tokenized_contexts_batch.append(tokenized_context)

            # Left truncate the inputs to the maximum length
            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            flat_index = 0
            for i, doc in enumerate(split):
                outputs_doc = outputs[flat_index : flat_index + len(doc.choices)]
                tokenized_continuations_doc = tokenized_continuations_batch[flat_index : flat_index + len(doc.choices)]
                tokenized_contexts_doc = tokenized_contexts_batch[flat_index : flat_index + len(doc.choices)]
                logprobs_doc = []
                argmax_doc = []
                output_tokens_doc = []
                input_tokens_doc = []

                for output, context, continuation in zip(
                    outputs_doc, tokenized_contexts_doc, tokenized_continuations_doc
                ):
                    continuation_logprobs = []
                    for token, logprobs in zip(continuation[::-1], output.prompt_logprobs[::-1]):
                        continuation_logprobs.append(logprobs[token])

                    bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                    continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                    continuation_logprobs = sum(continuation_logprobs)
                    logprobs_doc.append(continuation_logprobs)
                    argmax_doc.append(bool_score)
                    output_tokens_doc.append(continuation)
                    input_tokens_doc.append(context)

                answer = ModelResponse(
                    input=contexts[i],
                    input_tokens=input_tokens_doc,
                    output_tokens=output_tokens_doc,
                    logprobs=logprobs_doc,
                    argmax_logits_eq_gold=argmax_doc,
                )
                res.append(answer)
                flat_index += len(doc.choices)

        return dataset.get_original_order(res)

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError()
