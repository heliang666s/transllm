from openai import OpenAI
from src.transllm.adapters.openai import OpenAIAdapter
from src.transllm.adapters.anthropic import AnthropicAdapter
from src.transllm.converters.request_converter import RequestConverter
from src.transllm.core.schema import Provider

client = OpenAI()
openai_adapter = OpenAIAdapter()
anthropic_adapter = AnthropicAdapter()

stream = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
    stream=True
)

for chunk in stream:
    event = openai_adapter.to_unified_stream_event(chunk.model_dump())
    anthropic_event = anthropic_adapter.from_unified_stream_event(event)
    print(f"[Anthropic] {anthropic_event}", flush=True)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
    stream=False
)

converter = RequestConverter()
anthropic_response = converter.convert(
    response.model_dump(),
    from_provider=Provider.openai,
    to_provider=Provider.anthropic
)
print(f"[Anthropic] {anthropic_response}")