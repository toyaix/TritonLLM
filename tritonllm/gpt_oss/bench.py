"""
Unified Harmony Chat Tool with Interactive, Benchmark, and Tools Support
"""

import argparse
import datetime
import time
import random
import os
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import torch.distributed as dist
import termcolor

# Tools imports
try:
    from gpt_oss.tools import apply_patch
    from gpt_oss.tools.simple_browser import SimpleBrowserTool
    from gpt_oss.tools.simple_browser.backend import ExaBackend
    from gpt_oss.tools.python_docker.docker_tool import PythonTool
    from gpt_oss.tokenizer import get_tokenizer
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


class HarmonyChatTool:
    """Unified Harmony Chat Tool supporting multiple modes"""

    def __init__(self, checkpoint_path: str, context_length: int = 8192,
                 reasoning_effort: str = "high", developer_message: str = "",
                 enable_browser: bool = False, enable_python: bool = False,
                 enable_apply_patch: bool = False, show_browser_results: bool = False,
                 raw_mode: bool = False):
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.reasoning_effort = reasoning_effort
        self.developer_message = developer_message
        self.enable_browser = enable_browser
        self.enable_python = enable_python
        self.enable_apply_patch = enable_apply_patch
        self.show_browser_results = show_browser_results
        self.raw_mode = raw_mode

        # Initialize components
        self.device = torch.device("cuda:0")
        self.generator = None
        self.encoding = None
        self.tokenizer = None
        self.system_message = None
        self.base_messages = []

        # Tools
        self.browser_tool = None
        self.python_tool = None

        self._initialize_components()

    def get_file_lines(self, file_name: str, shuffle: bool = False) -> List[str]:
        """Load lines from file with optional shuffling"""
        try:
            import tritonllm.gpt_oss as gpt_oss
            file_path = os.path.join(Path(gpt_oss.__file__).parent.parent, "bin", file_name)
        except ImportError:
            # Fallback to current directory
            file_path = file_name

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            if shuffle:
                random.shuffle(lines)

            return [line for line in lines if line.strip()]  # Filter empty lines
        except FileNotFoundError:
            print(termcolor.colored(f"Warning: File {file_name} not found", "red"))
            return []


    def _initialize_components(self):
        """Initialize generator, encoding, and system messages"""
        print(termcolor.colored("Initializing Harmony Chat Tool...", "yellow"))

        # Initialize tokenizer
        if TOOLS_AVAILABLE:
            print(termcolor.colored("Loading tokenizer...", "yellow"), flush=True)
            self.tokenizer = get_tokenizer()
            print(termcolor.colored("✓ Tokenizer loaded successfully", "green"), flush=True)

        # Initialize generator with loading message
        try:
            from tritonllm.gpt_oss.triton.model import TokenGenerator as TritonGenerator
            self.generator = TritonGenerator(self.checkpoint_path, self.context_length, self.device)
        except ImportError:
            try:
                from gpt_oss.triton.model import TokenGenerator as TritonGenerator
                self.generator = TritonGenerator(self.checkpoint_path, self.context_length, self.device)
            except ImportError:
                try:
                    # Try the new import path from the tools code
                    from .triton.model import TokenGenerator as TritonGenerator
                    self.generator = TritonGenerator(self.checkpoint_path, self.context_length, self.device)
                except ImportError:
                    raise ImportError("Could not import TokenGenerator. Please check your installation.")
        print(termcolor.colored("✓ Model checkpoint loaded successfully", "green"), flush=True)

        # Initialize encoding
        print(termcolor.colored("Loading Harmony encoding...", "yellow"), flush=True)
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        print(termcolor.colored("✓ Harmony encoding loaded successfully", "green"), flush=True)

        # Initialize tools
        if TOOLS_AVAILABLE:
            self._initialize_tools()

        # Create system message
        print(termcolor.colored("Setting up system configuration...", "yellow"), flush=True)
        system_message_content = (
            SystemContent.new()
            .with_reasoning_effort(REASONING_EFFORT[self.reasoning_effort])
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
        )

        # Add tools to system message
        if self.enable_browser and self.browser_tool:
            system_message_content = system_message_content.with_tools(self.browser_tool.tool_config)

        if self.enable_python and self.python_tool:
            system_message_content = system_message_content.with_tools(self.python_tool.tool_config)

        self.system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)

        # Create base messages with developer message
        self.base_messages = [self.system_message]

        if self.enable_apply_patch and TOOLS_AVAILABLE:
            apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
            developer_message = ""
            if self.developer_message:
                developer_message = self.developer_message + "\n"
            developer_message += apply_patch_instructions.read_text()
            developer_message_content = (
                DeveloperContent.new()
                .with_instructions(developer_message)
                .with_function_tools([
                    ToolDescription.new(
                        "apply_patch",
                        "Patch a file",
                        parameters={
                            "type": "string",
                            "description": "Formatted patch code",
                            "default": "*** Begin Patch\n*** End Patch\n",
                        }
                    ),
                ])
            )
            self.base_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
        elif self.developer_message:
            developer_message_content = DeveloperContent.new().with_instructions(self.developer_message)
            self.base_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))

        print(termcolor.colored("✓ System configuration completed", "green"), flush=True)
        print(termcolor.colored("🚀 Ready to start!", "green", attrs=['bold']), flush=True)
        print()

    def _initialize_tools(self):
        """Initialize tools if available and enabled"""
        if self.enable_browser:
            print(termcolor.colored("Initializing browser tool...", "yellow"), flush=True)
            backend = ExaBackend(source="web")
            self.browser_tool = SimpleBrowserTool(backend=backend)
            print(termcolor.colored("✓ Browser tool initialized", "green"), flush=True)

        if self.enable_python:
            print(termcolor.colored("Initializing Python tool...", "yellow"), flush=True)
            self.python_tool = PythonTool()
            print(termcolor.colored("✓ Python tool initialized", "green"), flush=True)

    def get_user_input(self):
        """Get user input with distributed support"""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            user_input = input()
        else:
            user_input = ""
        user_input_list = [user_input]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(user_input_list, 0)
        return user_input_list[0]

    def print_system_info(self):
        """Print system and developer message information"""
        if self.raw_mode:
            # In raw mode, print the actual conversation tokens
            conversation = Conversation.from_messages(self.base_messages)
            tokens = self.encoding.render_conversation(conversation)
            system_message_text = self.encoding.decode(tokens)
            print(system_message_text, flush=True, end="")
            return

        # system_message.content is a list, get the first SystemContent object
        system_content = self.system_message.content[0] if self.system_message.content else None

        print(termcolor.colored("System Message:", "cyan"), flush=True)
        if system_content and hasattr(system_content, 'model_identity'):
            print(termcolor.colored("Model Identity:", "cyan"), system_content.model_identity, flush=True)
            print(termcolor.colored("Reasoning Effort:", "cyan"), system_content.reasoning_effort, flush=True)
            print(termcolor.colored("Conversation Start Date:", "cyan"), system_content.conversation_start_date, flush=True)
            print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_content.knowledge_cutoff, flush=True)
        else:
            print(termcolor.colored("System content loaded successfully", "cyan"), flush=True)

        # Tool status
        print(termcolor.colored("Browser Tool:", "cyan"), "Enabled" if self.enable_browser else "Disabled", flush=True)
        print(termcolor.colored("Python Tool:", "cyan"), "Enabled" if self.enable_python else "Disabled", flush=True)
        print(termcolor.colored("Apply Patch Function:", "cyan"), "Enabled" if self.enable_apply_patch else "Disabled", flush=True)

        # Developer message
        if len(self.base_messages) > 1:
            developer_content = self.base_messages[1].content[0] if self.base_messages[1].content else None
            if developer_content and hasattr(developer_content, 'instructions'):
                print(termcolor.colored("Developer Message:", "yellow"), flush=True)
                print(developer_content.instructions, flush=True)
        print()

    def _interactive_inference(self, user_message: str, messages: List[Message]) -> str:
        """Perform interactive inference with streaming output"""
        MESSAGE_PADDING = 12
        print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
        print(user_message)

        user_msg = Message.from_role_and_content(Role.USER, user_message)
        messages.append(user_msg)
        conversation = Conversation.from_messages(messages)
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
        current_output_text = ""
        output_text_delta_buffer = ""
        field_created = False
        token_begin = time.perf_counter()
        token_num = 0

        for predicted_token in self.generator.generate(
            tokens, self.encoding.stop_tokens_for_assistant_actions()
        ):
            token_num += 1
            parser.process(predicted_token)

            if parser.state == StreamState.EXPECT_START:
                print("")  # new line
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            output_text_delta_buffer += parser.last_content_delta
            print(output_text_delta_buffer, end="", flush=True)
            current_output_text += output_text_delta_buffer
            output_text_delta_buffer = ""

        # Adjust token count
        token_num = max(0, token_num - 10)
        token_end = time.perf_counter()
        elapsed = token_end - token_begin

        if elapsed > 0:
            print(termcolor.colored(f'\nTPS(Tokens Per Second) {token_num / elapsed:.3f}\n', "yellow"), flush=True)

        return current_output_text

    def _benchmark_inference(self, user_message: str, messages: List[Message], max_tokens: int = 0) -> dict:
        """Perform benchmark inference returning generation stats."""
        user_msg = Message.from_role_and_content(Role.USER, user_message)
        messages.append(user_msg)

        conversation = Conversation.from_messages(messages)
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        token_num = 0
        for predicted_token in self.generator.generate(
            tokens, self.encoding.stop_tokens_for_assistant_actions(), max_tokens=max_tokens
        ):
            token_num += 1

        generation_stats = self.generator.last_generation_stats or {}
        return {
            "generated_tokens": token_num,
            "prefill_time_s": generation_stats.get("prefill_time_s", 0.0),
            "decode_time_s": generation_stats.get("decode_time_s", 0.0),
        }

    def _process_tool_call(self, message: Message) -> List[Message]:
        """Process tool calls and return result messages"""
        if not message.recipient:
            return []

        if message.recipient.startswith("browser."):
            if not self.enable_browser or not self.browser_tool:
                raise ValueError("Browser tool is not enabled")

            async def run_browser_tool():
                results = []
                async for msg in self.browser_tool.process(message):
                    results.append(msg)
                return results

            return asyncio.run(run_browser_tool())

        elif message.recipient.startswith("python"):
            if not self.enable_python or not self.python_tool:
                raise ValueError("Python tool is not enabled")

            async def run_python_tool():
                results = []
                async for msg in self.python_tool.process(message):
                    results.append(msg)
                return results

            return asyncio.run(run_python_tool())

        elif message.recipient == "functions.apply_patch":
            if not self.enable_apply_patch or not TOOLS_AVAILABLE:
                raise ValueError("Apply patch tool is not enabled")

            text = message.content[0].text
            tool_output = None

            if text.startswith("{"):
                # this is json, try to extract the patch from it
                import json
                try:
                    some_dict = json.loads(text)
                    _, text = some_dict.popitem()
                except Exception as e:
                    tool_output = f"Error parsing JSON: {e}"

            if tool_output is None:
                try:
                    tool_output = apply_patch.apply_patch(text)
                except Exception as e:
                    tool_output = f"Error applying patch: {e}"

            result_message = (
                Message(
                    author=Author.new(Role.TOOL, message.recipient),
                    content=[TextContent(text=tool_output)]
                )
                .with_recipient("assistant")
            )
            if message.channel:
                result_message = result_message.with_channel(message.channel)

            return [result_message]

        else:
            raise ValueError(f"Unknown tool or function call: {message.recipient}")

    def _get_tool_name(self, recipient: str) -> str:
        """Get display name for tool"""
        if recipient.startswith("browser."):
            return "Search"
        elif recipient.startswith("python"):
            return "Python"
        elif recipient == "functions.apply_patch":
            return "Apply Patch"
        return "Unknown Tool"

    def interactive_mode(self):
        """Run interactive chat mode with tools support"""
        self.print_system_info()

        if self.raw_mode:
            empty_user_message_tokens = self.encoding.render(Message.from_role_and_content(Role.USER, ""))
            user_message_start = self.encoding.decode(empty_user_message_tokens[:-1])
            user_message_end = self.encoding.decode(empty_user_message_tokens[-1:])
        else:
            print(termcolor.colored("Interactive Chat Mode - Type 'quit' to exit", "green"))
            print(termcolor.colored("-" * 50, "green"))

        messages = self.base_messages.copy()
        MESSAGE_PADDING = 12

        try:
            while True:
                try:
                    last_message = messages[-1] if messages else None

                    # Handle tool calls first
                    if last_message and last_message.recipient is not None:
                        tool_results = self._process_tool_call(last_message)
                        messages += tool_results

                        # Display tool results
                        if not self.raw_mode:
                            tool_name = self._get_tool_name(last_message.recipient)
                            print(termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"), flush=True)
                            if tool_name == "Search" and not self.show_browser_results:
                                print("[Search results fed to the model]")
                            else:
                                print(tool_results[0].content[0].text)
                        else:
                            rendered_result = self.encoding.render_conversation(Conversation.from_messages(tool_results))
                            print(self.encoding.decode(rendered_result), flush=True, end="")

                        # Continue to generate assistant response
                        conversation = Conversation.from_messages(messages)
                        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

                        if self.raw_mode:
                            print(self.encoding.decode(tokens[-2:]), flush=True, end="")

                        # Generate and display assistant response
                        self._generate_response(tokens, messages)
                        continue

                    # Get user input
                    if self.raw_mode:
                        print(user_message_start, end="", flush=True)
                        user_input = self.get_user_input()
                        print(user_message_end, flush=True, end="")
                    else:
                        print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                        user_input = self.get_user_input().strip()

                        if user_input.lower() in ['quit', 'exit', 'q']:
                            break
                        if not user_input:
                            continue

                    user_msg = Message.from_role_and_content(Role.USER, user_input)
                    messages.append(user_msg)

                    # Generate assistant response
                    conversation = Conversation.from_messages(messages)
                    tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

                    if self.raw_mode:
                        print(self.encoding.decode(tokens[-2:]), flush=True, end="")

                    self._generate_response(tokens, messages)

                except KeyboardInterrupt:
                    print("\nChat interrupted by user")
                    break
                except Exception as e:
                    print(termcolor.colored(f"Error during inference: {e}", "red"))

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")

    def _generate_response(self, tokens, messages):
        """Generate and display assistant response"""
        parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        token_begin = time.perf_counter()
        token_num = 0

        for predicted_token in self.generator.generate(tokens, self.encoding.stop_tokens_for_assistant_actions()):
            token_num += 1
            parser.process(predicted_token)

            if self.raw_mode:
                print(self.encoding.decode([predicted_token]), end="", flush=True)
                continue

            if parser.state == StreamState.EXPECT_START:
                print("")  # new line
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta

            # Handle browser citations if enabled
            if self.enable_browser and self.browser_tool:
                updated_output_text, _annotations, has_partial_citations = self.browser_tool.normalize_citations(
                    current_output_text + output_text_delta_buffer
                )
                output_text_delta_buffer = updated_output_text[len(current_output_text):]
                if has_partial_citations:
                    should_send_output_text_delta = False

            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

        # Calculate and display TPS
        token_num = max(0, token_num - 10)
        token_end = time.perf_counter()
        elapsed = token_end - token_begin

        if elapsed > 0 and not self.raw_mode:
            print(termcolor.colored(f'\nTPS(Tokens Per Second) {token_num / elapsed:.3f}', "yellow"), flush=True)

        # Add parsed messages to conversation
        messages += parser.messages

    def only_output(self):
        prompt_files = ["prompt_zh.txt", "prompt.txt"]
        for file_name in prompt_files:
            for line in self.get_file_lines(file_name):
                self.single_inference(line, interactive=True)

    def benchmark_mode(
        self,
        prompt_files: List[str] = None,
        warmup_prompts_per_file: int = 1,
        warmup_max_tokens: int = 16,
    ):
        """Run benchmark mode"""
        if prompt_files is None:
            prompt_files = ["prompt_zh.txt", "prompt.txt"]

        self.print_system_info()
        print(termcolor.colored("Benchmark Mode", "green"))
        print(termcolor.colored("-" * 50, "green"))

        # Warm up
        print(termcolor.colored("Warming up...", "yellow"))
        for file_name in prompt_files:
            lines = self.get_file_lines(file_name, shuffle=False)
            if not lines:
                continue
            for user_message in lines[:warmup_prompts_per_file]:
                messages = self.base_messages.copy()
                self._benchmark_inference(
                    user_message,
                    messages,
                    max_tokens=warmup_max_tokens,
                )

        # Run benchmarks
        overall_stats = {
            "total_time": 0.0,
            "total_tokens": 0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "num_prompts": 0,
        }

        for prompt_file in prompt_files:
            lines = self.get_file_lines(prompt_file, shuffle=False)
            if not lines:
                continue

            print(termcolor.colored(f"\nBenchmarking {prompt_file}...", "cyan"))
            time_sum, token_sum = 0.0, 0
            prefill_time_sum, decode_time_sum = 0.0, 0.0

            for i, user_message in enumerate(lines):
                print(f"Processing prompt {i+1}/{len(lines)}: {user_message[:50]}...")

                token_begin = time.perf_counter()
                messages = self.base_messages.copy()
                stats = self._benchmark_inference(user_message, messages)
                elapsed = time.perf_counter() - token_begin
                token_num = stats["generated_tokens"]
                prefill_time = stats["prefill_time_s"]
                decode_time = stats["decode_time_s"]

                time_sum += elapsed
                token_sum += token_num
                prefill_time_sum += prefill_time
                decode_time_sum += decode_time

                decode_tps = token_num / decode_time if decode_time > 0 else 0.0
                e2e_tps = token_num / elapsed if elapsed > 0 else 0.0
                first_row_left = f"Decode TPS: {decode_tps:>8.3f}"
                second_row_left = f"Decode: {decode_time * 1000:>8.3f} ms"
                print(
                    termcolor.colored(
                        f"  {first_row_left:<29} | "
                        f"E2E TPS: {e2e_tps:>8.3f} | "
                        f"Generated tokens: {token_num:>6}\n"
                        f"  {second_row_left:<29} | "
                        f"Prefill: {prefill_time * 1000:>8.3f} ms",
                        "yellow",
                    )
                )

            if time_sum > 0:
                avg_prefill_ms = (prefill_time_sum / len(lines)) * 1000.0
                avg_decode_tps = token_sum / decode_time_sum if decode_time_sum > 0 else 0.0
                avg_e2e_tps = token_sum / time_sum
                print(termcolor.colored(f'{prompt_file} AVG Prefill: {avg_prefill_ms:.3f} ms', "green"))
                print(termcolor.colored(f'{prompt_file} AVG Decode TPS: {avg_decode_tps:.3f}', "green"))
                print(termcolor.colored(f'{prompt_file} AVG E2E TPS: {avg_e2e_tps:.3f}', "green"))
                overall_stats["total_time"] += time_sum
                overall_stats["total_tokens"] += token_sum
                overall_stats["prefill_time"] += prefill_time_sum
                overall_stats["decode_time"] += decode_time_sum
                overall_stats["num_prompts"] += len(lines)

        # Overall statistics
        if overall_stats["total_time"] > 0:
            overall_avg_prefill_ms = (
                overall_stats["prefill_time"] / overall_stats["num_prompts"] * 1000.0
                if overall_stats["num_prompts"] > 0
                else 0.0
            )
            overall_decode_tps = (
                overall_stats["total_tokens"] / overall_stats["decode_time"]
                if overall_stats["decode_time"] > 0
                else 0.0
            )
            overall_e2e_tps = overall_stats["total_tokens"] / overall_stats["total_time"]
            print(termcolor.colored(f'\nOverall AVG Prefill: {overall_avg_prefill_ms:.3f} ms', "magenta"))
            print(termcolor.colored(f'Overall AVG Decode TPS: {overall_decode_tps:.3f}', "magenta"))
            print(termcolor.colored(f'Overall AVG E2E TPS: {overall_e2e_tps:.3f}', "magenta"))

    def single_inference(self, user_message: str, interactive: bool = True) -> str:
        """Perform single inference - API interface"""
        messages = self.base_messages.copy()

        if interactive:
            return self._interactive_inference(user_message, messages)
        else:
            # Silent inference for API use
            user_msg = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_msg)
            conversation = Conversation.from_messages(messages)
            tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

            parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
            output_text = ""

            for predicted_token in self.generator.generate(
                tokens, self.encoding.stop_tokens_for_assistant_actions()
            ):
                parser.process(predicted_token)
                if parser.last_content_delta:
                    output_text += parser.last_content_delta

            return output_text

    def create_conversation_session(self):
        """Create a conversation session that maintains history"""
        return ConversationSession(self)


class ConversationSession:
    """Maintains conversation history for multiple exchanges"""

    def __init__(self, chat_tool: HarmonyChatTool):
        self.chat_tool = chat_tool
        self.messages = chat_tool.base_messages.copy()

    def send_message(self, user_message: str, interactive: bool = True, show_separator: bool = True) -> str:
        """Send message with conversation history"""
        if interactive and show_separator:
            print(termcolor.colored("=" * 60, "blue"))

        if interactive:
            response = self.chat_tool._interactive_inference(user_message, self.messages)
        else:
            # Silent inference with history
            user_msg = Message.from_role_and_content(Role.USER, user_message)
            self.messages.append(user_msg)
            conversation = Conversation.from_messages(self.messages)
            tokens = self.chat_tool.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

            parser = StreamableParser(self.chat_tool.encoding, role=Role.ASSISTANT)
            response = ""

            for predicted_token in self.chat_tool.generator.generate(
                tokens, self.chat_tool.encoding.stop_tokens_for_assistant_actions()
            ):
                parser.process(predicted_token)
                if parser.last_content_delta:
                    response += parser.last_content_delta

            # Add assistant response to history
            assistant_msg = Message.from_role_and_content(Role.ASSISTANT, response)
            self.messages.append(assistant_msg)

        return response

    def reset_conversation(self):
        """Reset conversation to initial state"""
        self.messages = self.chat_tool.base_messages.copy()

    def get_conversation_length(self) -> int:
        """Get number of exchanges in current conversation"""
        return len([m for m in self.messages if m.role in [Role.USER, Role.ASSISTANT]]) // 2
