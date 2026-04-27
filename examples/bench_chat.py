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
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        default=None,
        dest="cpu_offload",
        help="Enable CPU offload for MoE expert weights",
    )
    args = parser.parse_args()

    checkpoint = get_model_with_checkpoint(args.checkpoint)
    tool = HarmonyChatTool(checkpoint, reasoning_effort="high", cpu_offload=args.cpu_offload)

    if args.prompt:
        tool.print_system_info()
        response = tool.single_inference(args.prompt, interactive=True)
        stats = tool.generator.last_generation_stats or {}
        prefill_time = stats.get("prefill_time_s", 0)
        decode_time = stats.get("decode_time_s", 0)
        if prefill_time or decode_time:
            print(f"Prefill: {prefill_time * 1000:.1f} ms | Decode: {decode_time * 1000:.1f} ms")
    else:
        result = tool.benchmark_mode(
            warmup_prompts_per_file=1,
            warmup_max_tokens=16,
        )
