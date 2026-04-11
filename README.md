<h3 align="center">
TritonLLM: LLM Inference via Triton 🚀
</h3>

<h4 align="center">
Flexible and modular LLM inference for mini-batch
</h4>

<p align="center">
<a href="https://tritonllm.top"><b>🔗 tritonllm.top</b></a>
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

TritonLLM implements modular [Triton](https://github.com/triton-lang/triton)-backed LLM inference with an emphasis on kernel optimization using CUBINs. The initial target is the [gpt-oss](https://github.com/openai/gpt-oss) model, executed via [triton_runner](https://github.com/toyaix/triton_runner) and will be tuned for **RTX 5090** (sm120). Now support an NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda-gpus) sm120(RTX 5090, RTX PRO 6000, etc.), sm90(H100, H200, H20, etc.), sm80(A800, A100), sm89(RTX 4090, RTX 6000, L40, etc.) and sm86(RTX 3090, A10, etc.). If the GPU memory is greater than or equal to **24 GB**, you can run the **gpt-oss-20b**; if it is greater than or equal to **80 GB**, you can run the **gpt-oss-120b**.

The project is compatible with PyTorch 2.10 and runs correctly in that environment. However, for the best performance, we recommend using PyTorch 2.8 together with Triton 3.4.0.

## Quick Installation

You can install the latest stable release of `tritonllm` from pip:

```shell
pip install tritonllm
```

To enable the optional `triton_runner` JIT backend:

```shell
pip install "tritonllm[runner]"
```

## 🚀 Command Line Interface (CLI)

To quickly launch with the **gpt-oss-20b** model and automatically download it from ModelScope:

```shell
tritonllm
```

You can explore all available options with:

```shell
tritonllm --help
```

### Usage

```shell
usage: tritonllm [-h] [-r REASONING_EFFORT] [-a] [-b] [--show-browser-results] [-p]
                 [--developer-message DEVELOPER_MESSAGE] [-c CONTEXT] [--raw]
                 [FILE]

```

#### Positional arguments

| Argument | Description |
|----------|-------------|
| `FILE`   | Path to the SafeTensors checkpoint. If not provided, downloads the **20B model** from ModelScope. You can also run `tritonllm 120b` to directly use the **120B model** from ModelScope.   |

#### Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show this help message and exit. |
| `-r REASONING_EFFORT, --reasoning-effort REASONING_EFFORT` | Set reasoning effort level (`low` / `medium` / `high`). Default: `high`. |
| `-a, --apply-patch` | Make the internal `apply_patch` function available to the model. Default: `False`. |
| `-b, --browser` | Enable browser tool so the model can fetch web content. Default: `False`. |
| `--show-browser-results` | Show fetched browser results in the output. Default: `False`. |
| `-p, --python` | Enable Python execution tool (run Python snippets). Default: `False`. |
| `--developer-message DEVELOPER_MESSAGE` | Provide a developer/system message that influences the model’s behavior. |
| `-c CONTEXT, --context CONTEXT` | Maximum context length (tokens). Default: `8192`. |
| `--raw` | Raw mode. Disable Harmony encoding and render plain output. Default: `False`. |


## Install from source

```shell
git clone https://github.com/toyaix/tritonllm
cd tritonllm

pip install -e .
```

Install the optional runner backend from source:

```shell
pip install -e ".[runner]"
```

## JIT backend selection

By default the project keeps using `@triton.jit`. To switch the package-managed kernels to `@triton_runner.jit`, set:

```shell
export TRITONLLM_JIT_BACKEND=triton_runner
```

Supported values are `triton` and `triton_runner`. If `triton_runner` is selected without the optional dependency installed, the import will fail fast with a clear error.

## example code

```Python
from tritonllm.gpt_oss.chat import chat, get_parser_args


if __name__ == "__main__":
    chat(get_parser_args())
```

## Run

```shell
# test
python examples/generate.py

# chat
python examples/chat.py
```

## Benchmark

I am currently optimizing **Tokens Per Second**(TPS), the number of tokens generated per second during autoregressive decoding.

```shell
python examples/bench_chat.py

# show output
python examples/only_output.py
```

## Run use streamlit with Responses API(has bug)

You can also use Streamlit to interact with the [Responses API](https://github.com/openai/gpt-oss?tab=readme-ov-file#responses-api), providing a convenient web interface for managing the project.

```shell
pip install streamlit

python -m gpt_oss.responses_api.serve

streamlit run streamlit/streamlit_chat.py
```

## triton_kernels

triton_kernels is a set of kernels that enable fast moe on different architectures. These kernels are compatible with different precision (e.g bf16, mxfp4)

Original code here https://github.com/triton-lang/triton/tree/main/python/triton_kernels

The current version is the following commit de4376e90a3c2b5ca30ada25a50cccadeadf7f1a and use BlackwellMXValueLayout with commit 19ca20fda4cfd3ae0d3eabde5e547db581fbb7ee。
