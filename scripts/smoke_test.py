"""Smoke test: does the model answer, and does it call tools sensibly?"""

import argparse
import sys
import time

from jarvis.config import load_settings
from jarvis.core.llm import LLMError
from jarvis.core.ollama_provider import OllamaProvider
from jarvis.logging_config import setup_logging

SAVE_NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a personal note or fact that the user wants remembered.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The note content"},
            },
            "required": ["text"],
        },
    },
}

# (label, prompt, should the model call a tool?)
CASES = [
    ("Plain chat", "Say hello in one short sentence.", False),
    ("Should call tool", "Please remember that my dentist appointment is on Friday at 10:00.", True),
    ("Should NOT call tool", "What is the capital of France?", False),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the model from .env")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings.log_level)
    model = args.model or settings.model
    provider = OllamaProvider(model=model, host=settings.ollama_host, num_ctx=settings.num_ctx)

    print(f"Model: {model} (num_ctx={settings.num_ctx})\n")
    failures = 0
    for label, prompt, expect_tool in CASES:
        started = time.perf_counter()
        try:
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[SAVE_NOTE_TOOL],
            )
        except LLMError as exc:
            print(f"[ERROR] {label}: {exc}")
            return 1
        elapsed = time.perf_counter() - started

        called = bool(response.tool_calls)
        ok = called == expect_tool
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label} ({elapsed:.1f}s)")
        print(f"        prompt: {prompt}")
        if called:
            for call in response.tool_calls:
                print(f"        tool call: {call.name}({call.arguments})")
        else:
            print(f"        reply: {response.text.strip()}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())