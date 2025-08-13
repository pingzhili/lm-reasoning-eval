import litellm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')
kwargs = {
    "model": 'openai/gpt-oss-120b',
    "messages": [{'role': 'user',
                  'content': "Question: A tank has a capacity of 18000 gallons. Wanda and Ms. B decided to pump water from a pond to fill the tank in two days. On the first day, working in shifts, Wanda filled 1/4 of the tank's capacity with water, and Ms. B pumped 3/4 as much water as Wanda pumped into the tank that day. On the second day, Wanda pumped 2/3 of the amount of water she pumped on the previous day, while Ms. B only pumped 1/3 of the number of gallons she pumped on the first day. How many gallons of water are remaining for the tank to be full?\nAnswer:"}],
    "base_url": "http://localhost:8000/v1",
    "n": 1,
    "caching": False,  # We don't want caching with same response
    "api_key": "nan",
}

# init
thinking_budget = 128
thinking_terminate_token = "<|end|>"

initial_messages = kwargs.get('messages')
final_max_tokens = kwargs.get('max_completion_tokens', 1024)

thinking_kwargs = kwargs.copy()
thinking_kwargs['max_completion_tokens'] = thinking_budget
thinking_kwargs['stop'] = thinking_terminate_token

init_response = litellm.completion(**thinking_kwargs)
ongoing_text = init_response.choices[0].message.content or ""
stop_reason = init_response.choices[0].finish_reason

while stop_reason != "length":
    thinking_steps = init_response.choices[0].message.content.split("\n\n")
    ongoing_text = "\n\n".join(thinking_steps[:-1])
    ongoing_text = ongoing_text + "\n\nWait, "









