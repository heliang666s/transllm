from openai import OpenAI
from src.transllm.converters.request_converter import RequestConverter
from src.transllm.converters.stream_converter import StreamConverter
from src.transllm.core.schema import Provider

client = OpenAI()
converter = StreamConverter()

stream = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn.",
        }
    ],
    stream=True,
)

for chunk in stream:
    anthropic_event = converter.convert_stream_event(
        chunk.model_dump(), Provider.openai, Provider.anthropic
    )
    print(f"[Anthropic] {anthropic_event}", flush=True)

response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn.",
        }
    ],
    stream=False,
)

converter = RequestConverter()
anthropic_response = converter.convert(
    response.model_dump(), from_provider=Provider.openai, to_provider=Provider.anthropic
)
print(f"[Anthropic] {anthropic_response}")
