from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model = LLM("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
requests = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False
)
sampling_params = SamplingParams(logprobs=10)
outputs = model.generate(prompt_token_ids=requests, sampling_params=sampling_params)
breakpoint()
