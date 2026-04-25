import time
from tritonllm.gpt_oss.bench import HarmonyChatTool
from tritonllm.utils import get_model_with_checkpoint
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HarmonyChatTool")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="20b",
        type=str,
        help="Path to the SafeTensors checkpoint (default: %(default)s with modelscope)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run benchmark with a single prompt instead of prompt files",
    )
    args = parser.parse_args()

    checkpoint = get_model_with_checkpoint(args.checkpoint)
    tool = HarmonyChatTool(checkpoint, reasoning_effort="high")

    if args.prompt:
        tool.print_system_info()
        messages = tool.base_messages.copy()
        token_begin = time.perf_counter()
        stats = tool._benchmark_inference(args.prompt, messages)
        elapsed = time.perf_counter() - token_begin
        token_num = stats["generated_tokens"]
        decode_time = stats["decode_time_s"]
        prefill_time = stats["prefill_time_s"]
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {token_num} tokens")
        print(f"Decode TPS: {token_num / decode_time:.3f}" if decode_time > 0 else "Decode TPS: N/A")
        print(f"E2E TPS: {token_num / elapsed:.3f}" if elapsed > 0 else "E2E TPS: N/A")
        print(f"Prefill: {prefill_time * 1000:.1f} ms")
        print(f"Decode: {decode_time * 1000:.1f} ms")
    else:
        result = tool.benchmark_mode(
            warmup_prompts_per_file=1,
            warmup_max_tokens=16,
        )
