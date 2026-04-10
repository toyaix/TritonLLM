import argparse
import time

import torch
from openai_harmony import Message, Role

from tritonllm.gpt_oss.bench import HarmonyChatTool
from tritonllm.utils import get_model_with_checkpoint


def run_decode_bench(
    checkpoint: str,
    prompt: str,
    warmup: int,
    steps: int,
    reasoning_effort: str,
) -> None:
    checkpoint = get_model_with_checkpoint(checkpoint)
    tool = HarmonyChatTool(checkpoint, reasoning_effort=reasoning_effort)

    messages = tool.base_messages.copy()
    messages.append(Message.from_role_and_content(Role.USER, prompt))
    prompt_tokens = tool._render_with_truncation(messages)
    stop_tokens = tool.encoding.stop_tokens_for_assistant_actions()

    for _ in range(warmup):
        for _token in tool.generator.generate(
            prompt_tokens,
            stop_tokens,
            temperature=0.0,
            max_tokens=1,
        ):
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    num_tokens = 0
    for _token in tool.generator.generate(
        prompt_tokens,
        stop_tokens,
        temperature=0.0,
        max_tokens=steps,
    ):
        num_tokens += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if num_tokens == 0:
        raise RuntimeError("Decode benchmark generated zero tokens; try a larger --steps value.")

    print(f"prompt={prompt!r}")
    print(f"warmup_steps={warmup}")
    print(f"decode_tokens={num_tokens}")
    print(f"elapsed={elapsed:.6f}s")
    print(f"ms_per_token={elapsed * 1000.0 / num_tokens:.3f}")
    print(f"tokens_per_second={num_tokens / elapsed:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark single-sequence decode speed.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="20b",
        type=str,
        help="Path to the SafeTensors checkpoint (default: %(default)s with modelscope)",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍 Transformer。",
        type=str,
        help="Prompt used for decode benchmarking.",
    )
    parser.add_argument(
        "--warmup",
        default=8,
        type=int,
        help="Number of one-token warmup runs before measurement.",
    )
    parser.add_argument(
        "--steps",
        default=64,
        type=int,
        help="Number of decode tokens to measure.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort used to initialize HarmonyChatTool.",
    )
    args = parser.parse_args()

    run_decode_bench(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        warmup=args.warmup,
        steps=args.steps,
        reasoning_effort=args.reasoning_effort,
    )
