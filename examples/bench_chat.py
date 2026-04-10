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
        "--context-length",
        default=131072,
        type=int,
        help="Runtime KV-cache context length (default: %(default)s)"
    )
    parser.add_argument(
        "--benchmark-max-tokens",
        default=0,
        type=int,
        help="Per-prompt decode cap for benchmark. 0 means natural stop within context budget."
    )
    args = parser.parse_args()
    checkpoint = get_model_with_checkpoint(args.checkpoint)
    tool = HarmonyChatTool(
        checkpoint,
        context_length=args.context_length,
        reasoning_effort="high",
    )
    result = tool.benchmark_mode(
        warmup_prompts_per_file=1,
        warmup_max_tokens=16,
        benchmark_max_tokens=args.benchmark_max_tokens,
    )
