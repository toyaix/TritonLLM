# Model parallel inference
# Note: This script is for demonstration purposes only. It is not designed for production use.
#       See gpt_oss.chat for a more complete example with the Harmony parser.
# torchrun --nproc-per-node=4 -m gpt_oss.generate -p "why did the chicken cross the road?" model/

import torch
import argparse

from gpt_oss.tokenizer import get_tokenizer
from tritonllm.utils import get_model_with_checkpoint


def generate(args):
    from gpt_oss.triton.model import TokenGenerator as TritonGenerator
    device = torch.device(f"cuda:0")
    checkpoint = get_model_with_checkpoint(args.checkpoint)
    cpu_offload = getattr(args, 'cpu_offload', None)
    generator = TritonGenerator(checkpoint, context=args.context_length, device=device, cpu_offload=cpu_offload)

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(args.prompt)
    for token, logprob in generator.generate(tokens,
                                             stop_tokens=[tokenizer.eot_token],
                                             temperature=args.temperature,
                                             max_tokens=args.limit,
                                             return_logprobs=True):
        tokens.append(token)
        decoded_token = tokenizer.decode([token])
        print(decoded_token, end="")
    print()

def get_parser_args():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        nargs="?",
        default="20b",
        type=str,
        help="Path to the SafeTensors checkpoint (default: %(default)s with modelscope)"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="How are you?",
        help="LLM prompt",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "-l",
        "--limit",
        metavar="LIMIT",
        type=int,
        default=0,
        help="Limit on the number of tokens (0 to disable)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        default=None,
        dest="cpu_offload",
        help="Enable CPU offload for MoE expert weights",
    )
    return parser.parse_args()
