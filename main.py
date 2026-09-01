"""
Test script for google/gemma-4-E2B-it tool/function calling.

What this does:
1. Loads the Gemma 4 E2B instruct model + processor.
2. Defines a sample Python tool (get_current_weather) with a proper docstring
   so `transformers` can auto-generate its JSON schema.
3. Sends a user question that should trigger a tool call.
4. Parses the model's tool call, "executes" it (mocked), feeds the result back,
   and gets the model's final natural-language answer.

Setup:
    pip install -U transformers torch accelerate huggingface_hub

Notes:
- google/gemma-4-E2B-it is ~5.1B params (2.3B "effective"). It'll run on a
  single consumer GPU in bf16, or on CPU (slowly) if you don't have one.
- Put your Hugging Face token in a `.env` file as `HF_TOKEN=...`.
- On first run the script checks the local HF cache; if the model is missing
  it downloads it automatically (multiple GB).
"""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.utils import get_json_schema

MODEL_ID = "google/gemma-4-E2B-it"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_hf_token() -> str:
    """Load HF_TOKEN from .env (if present) and the process environment."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN in .env or your environment."
        )
    return token


def ensure_model_cached(model_id: str, token: str) -> str:
    """Return the local snapshot path, downloading the model if it is not cached."""
    print(f"Checking local cache for {model_id} ...")
    try:
        cache_path = snapshot_download(repo_id=model_id, token=token, local_files_only=True)
        print(f"Model found locally at: {cache_path}")
        return cache_path
    except LocalEntryNotFoundError:
        print(f"Model not found locally. Downloading {model_id} ...")
        cache_path = snapshot_download(repo_id=model_id, token=token)
        print(f"Download complete: {cache_path}")
        return cache_path


# ---------------------------------------------------------------------------
# 1. Define a sample tool
# ---------------------------------------------------------------------------
# transformers can auto-build the JSON schema from a plain Python function as
# long as it has type hints and a Google-style docstring.
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location.

    Args:
        location: The city and state/country, e.g. "Chennai, India".
        unit: The temperature unit to use. Must be "celsius" or "fahrenheit".

    Returns:
        A short string describing the current weather.
    """
    # This is a MOCKED implementation for testing tool-calling behavior.
    # Swap this out for a real weather API call if you want live data.
    fake_weather_db = {
        "chennai": "32°C, humid, partly cloudy",
        "london": "15°C, light rain",
        "tokyo": "21°C, clear skies",
    }
    key = location.split(",")[0].strip().lower()
    condition = fake_weather_db.get(key, "28°C, sunny (default mock data)")
    return f"Weather in {location}: {condition}"


# Map tool name -> actual callable, so we can execute whatever the model asks for.
AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather,
}


def main():
    token = load_hf_token()
    ensure_model_cached(MODEL_ID, token)

    print(f"Loading {MODEL_ID} into memory ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
        token=token,
    )

    # -----------------------------------------------------------------------
    # 2. Build the conversation + pass the tool schema
    # -----------------------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Chennai right now?",
        }
    ]

    tool_schemas = [get_json_schema(get_current_weather)]

    inputs = processor.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # -----------------------------------------------------------------------
    # 3. First model turn: it should emit a tool call instead of plain text
    # -----------------------------------------------------------------------
    print("\n--- Generating model's first response (expecting a tool call) ---")
    outputs = model.generate(**inputs, max_new_tokens=512)
    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(raw_response, prefix=inputs["input_ids"])

    print("\nRaw model output:")
    print(raw_response)
    print("\nParsed response object:")
    print(parsed)

    # `parsed` will contain a tool_calls entry if the model decided to call a tool.
    tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None

    if not tool_calls:
        print("\nModel answered directly without calling a tool. Done.")
        return

    # -----------------------------------------------------------------------
    # 4. Execute the requested tool(s) locally
    # -----------------------------------------------------------------------
    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]
        if isinstance(fn_args, str):
            fn_args = json.loads(fn_args)

        print(f"\nModel requested tool call: {fn_name}({fn_args})")

        fn = AVAILABLE_TOOLS.get(fn_name)
        if fn is None:
            result = f"Error: tool '{fn_name}' is not available."
        else:
            result = fn(**fn_args)

        print(f"Tool result: {result}")

        # Feed the tool's result back into the conversation.
        messages.append(
            {
                "role": "tool",
                "name": fn_name,
                "content": str(result),
            }
        )

    # -----------------------------------------------------------------------
    # 5. Second model turn: it should now produce a natural-language answer
    # -----------------------------------------------------------------------
    inputs2 = processor.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len2 = inputs2["input_ids"].shape[-1]

    print("\n--- Generating model's final answer (after tool result) ---")
    outputs2 = model.generate(**inputs2, max_new_tokens=256)
    final_raw = processor.decode(outputs2[0][input_len2:], skip_special_tokens=False)
    final_parsed = processor.parse_response(final_raw, prefix=inputs2["input_ids"])

    print("\nFinal raw output:")
    print(final_raw)
    print("\nFinal answer:")
    print(final_parsed)


if __name__ == "__main__":
    main()