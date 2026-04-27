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
        "--cpu-offload",
        action="store_true",
        default=None,
        dest="cpu_offload",
        help="Enable CPU offload for MoE expert weights",
    )
    args = parser.parse_args()
    checkpoint = get_model_with_checkpoint(args.checkpoint)
    tool = HarmonyChatTool(checkpoint, reasoning_effort="high", cpu_offload=args.cpu_offload)
    result = tool.only_output()
