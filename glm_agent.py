#!/usr/bin/env python3
"""
Baseline LangGraph Agent with GLM (ZAI) as brain.
Inspired by Hermes Agent architecture.

Uses OpenClaw's auth token from auth-profiles.json:
  - API: https://api.z.ai/api/coding/paas/v4
  - Model: glm-4.7 (reasoning model - returns content + reasoning_content)

Features:
- Function calling (tools: calculator, read_file)
- Conversation memory (message history)
- Handles GLM reasoning model (content + reasoning_content fields)
- Streaming ready

Run:
    python3 glm_agent.py
"""

import json
import sys

# -----------------------------------------------------------------------
# Load OpenClaw Auth Token
# -----------------------------------------------------------------------
def get_zai_config():
    auth_file = "/root/.openclaw/agents/main/agent/auth-profiles.json"
    config_file = "/root/.openclaw/openclaw.json"
    
    with open(auth_file) as f:
        auth_data = json.load(f)
    with open(config_file) as f:
        config_data = json.load(f)
    
    token = auth_data["profiles"]["zai:default"]["key"]
    base_url = config_data["models"]["providers"]["zai"]["baseUrl"]
    # NOTE: glm-4.7-flash returns empty content but has reasoning_content
    #       glm-4.7 works correctly with both content + reasoning_content
    model = "glm-4.7"
    
    return token, base_url, model

API_KEY, BASE_URL, MODEL_NAME = get_zai_config()

from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)

# -----------------------------------------------------------------------
# Tool Definitions (for function calling)
# -----------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression safely. Use for arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '123 * 456' or '2^10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    }
]

# -----------------------------------------------------------------------
# Tool Implementations
# -----------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Evaluate a math expression safely (no arbitrary code execution)."""
    try:
        allowed = set("0123456789+-*/.()^ ")
        if set(expression) - allowed:
            return "❌ Invalid characters in expression."
        # Safe eval: replace ^ with ** for python
        safe_expr = expression.replace("^", "**")
        result = eval(safe_expr, {"__builtins__": {}, "abs": abs, "round": round})
        return f"✅ {expression} = {result}"
    except Exception as e:
        return f"❌ Calculation error: {e}"

def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > 2000:
            content = content[:2000] + f"\n... [truncated, total {len(content)} bytes]"
        return f"📄 File: {path}\n```\n{content}\n```"
    except Exception as e:
        return f"❌ Cannot read file: {e}"

TOOL_MAP = {
    "calculator": calculator,
    "read_file": read_file,
}

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def extract_content(choice) -> str:
    """Extract text from LLM response - handles GLM-4.7 reasoning model."""
    msg = choice.message
    if msg.content and msg.content.strip():
        return msg.content
    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
        # For glm-4.7-flash: content is empty, reasoning has the answer
        return msg.reasoning_content
    return "(no content)"

def call_glm(messages: list, max_tokens: int = 2048, tools: list = None) -> dict:
    """Call GLM with function calling support."""
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0]

# -----------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🤖 GLM Agent — LangGraph Terminal Q&A (Baseline)")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API: {BASE_URL}")
    print("   Type 'exit' or Ctrl+C to quit")
    print("=" * 60)
    print()

    SYSTEM = (
        "Kamu adalah asisten AI berbasis GLM-4.7. "
        "Gunakan bahasa Indonesia untuk merespons. "
        "Kamu bisa menggunakan tool calculator dan read_file kalau perlu. "
        "Respon langsung, jelas, dan helpful."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    MAX_TURNS = 10

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "keluar"):
            print("👋 Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("\n🤖 Agent: ", end="", flush=True)

        # LLM → Tool → LLM loop (like Hermes)
        for turn in range(MAX_TURNS):
            choice = call_glm(messages, tools=TOOLS)

            if choice.finish_reason == "stop":
                content = extract_content(choice)
                print(content)
                messages.append({"role": "assistant", "content": content})
                break

            elif choice.finish_reason == "tool_calls":
                # Execute tools, add results to messages
                for tc in choice.message.tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    func = TOOL_MAP.get(func_name)
                    
                    if func:
                        result = func(**args)
                    else:
                        result = f"❌ Unknown tool: {func_name}"
                    
                    print(f"[{func_name}] ", end="", flush=True)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": str(result),
                    })
                print()
                print("🤖 Agent: ", end="", flush=True)
                # Continue loop to get final response

            elif choice.finish_reason == "length":
                content = extract_content(choice)
                print(f"[trimmed] {content[:200]}...")
                messages.append({"role": "assistant", "content": content})
                break

            else:
                print(f"[finish: {choice.finish_reason}]")
                break

        else:
            print("⚠️ Max turns reached (possible loop).")

        print()

if __name__ == "__main__":
    main()