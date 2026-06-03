#!/usr/bin/env python3
"""
GLM Agent - Proper LangGraph Implementation
Inspired by Hermes Agent architecture.

LangGraph State Graph with:
- Typed State (AgentState)
- Message history with add_messages reducer
- Conditional edges (tool calling routing)
- Separate nodes: llm, tools, routing
- Graceful tool error handling

Run:
    python3 glm_agent.py
"""

import json
import sys
from typing import Annotated, Literal
from typing_extensions import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# -----------------------------------------------------------------------
# Auth Config
# -----------------------------------------------------------------------
def get_zai_config():
    auth_file = "/root/.openclaw/agents/main/agent/auth-profiles.json"
    config_file = "/root/.openclaw/openclaw.json"
    with open(auth_file) as f:
        auth_data = json.load(f)
    with open(config_file) as f:
        config_data = json.load(f)
    return (
        auth_data["profiles"]["zai:default"]["key"],
        config_data["models"]["providers"]["zai"]["baseUrl"],
        "glm-4.7",
    )

API_KEY, BASE_URL, MODEL_NAME = get_zai_config()

# OpenAI client (ZAI compatible)
llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)

# -----------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------
class AgentState(TypedDict):
    """Typed state with message history managed by add_messages reducer."""
    messages: Annotated[list, add_messages]
    tool_calls_count: int  # Track tool calls to prevent infinite loops


# -----------------------------------------------------------------------
# Tool Definitions (OpenAI function calling format)
# -----------------------------------------------------------------------
TOOLS_SCHEMA = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list"
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
    """Safe math evaluation."""
    try:
        allowed = set("0123456789+-*/.()^ ")
        if set(expression) - allowed:
            return "❌ Invalid characters in expression."
        safe_expr = expression.replace("^", "**")
        result = eval(safe_expr, {"__builtins__": {}, "abs": abs, "round": round, "sqrt": __import__("math").sqrt})
        return f"✅ {expression} = {result}"
    except Exception as e:
        return f"❌ Calculation error: {e}"

def read_file(path: str) -> str:
    """Read file contents with truncation."""
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > 2000:
            content = content[:2000] + f"\n... [truncated, total {len(content)} bytes]"
        return f"📄 File: {path}\n```\n{content}\n```"
    except Exception as e:
        return f"❌ Cannot read file: {e}"

def list_directory(path: str) -> str:
    """List directory contents."""
    try:
        import os
        entries = os.listdir(path)
        formatted = [f"  📁 {e}/" if os.path.isdir(os.path.join(path, e)) else f"  📄 {e}"
                     for e in sorted(entries)]
        return f"📂 Directory: {path}\n" + "\n".join(formatted[:20])
    except Exception as e:
        return f"❌ Cannot list directory: {e}"

TOOL_MAP = {
    "calculator": calculator,
    "read_file": read_file,
    "list_directory": list_directory,
}

# -----------------------------------------------------------------------
# LLM Call Helper
# -----------------------------------------------------------------------
def call_glm(messages: list) -> dict:
    """Call GLM-4.7 with function calling."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0]

def extract_content(msg) -> str:
    """Extract text from LLM message - handles reasoning model."""
    if msg.content and msg.content.strip():
        return msg.content
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    return "(no response)"


def msg_to_dict(msg) -> dict:
    """Convert message object to serializable dict for LangGraph."""
    if isinstance(msg, dict):
        return msg
    result = {"role": msg.role, "content": msg.content or ""}
    # Handle tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ]
    return result


# -----------------------------------------------------------------------
# LangGraph Nodes
# -----------------------------------------------------------------------

def llm_node(state: AgentState) -> AgentState:
    """
    LLM Node: Calls GLM-4.7 with current message history.
    Returns updated state with assistant response.
    """
    messages = state["messages"]
    
    # Convert all messages to dict format for API
    api_messages = []
    for msg in messages:
        if hasattr(msg, "model"):  # OpenAI message object
            api_messages.append(msg_to_dict(msg))
        elif isinstance(msg, dict):
            api_messages.append(msg)
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    choice = call_glm(api_messages)
    msg = choice.message
    
    if choice.finish_reason == "tool_calls":
        # LLM wants to call tools - add message and route to tools
        return {
            "messages": [msg_to_dict(msg)],
            "tool_calls_count": state.get("tool_calls_count", 0),
        }
    elif choice.finish_reason == "stop":
        content = extract_content(msg)
        return {
            "messages": [{"role": "assistant", "content": content}],
            "tool_calls_count": 0,  # Reset on successful response
        }
    elif choice.finish_reason == "length":
        content = extract_content(msg)
        return {
            "messages": [{"role": "assistant", "content": content + "\n[response truncated]"}],
            "tool_calls_count": 0,
        }
    else:
        return {
            "messages": [{"role": "assistant", "content": f"[finish: {choice.finish_reason}]"}],
            "tool_calls_count": 0,
        }


def tools_node(state: AgentState) -> AgentState:
    """
    Tools Node: Executes tool calls from LLM response.
    Returns updated state with tool results.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    # Get tool calls from message
    tool_calls = None
    if isinstance(last_msg, dict):
        tool_calls = last_msg.get("tool_calls")
    elif hasattr(last_msg, "tool_calls"):
        tool_calls = last_msg.tool_calls
    
    if not tool_calls:
        return state
    
    new_messages = []
    for tc in tool_calls:
        # Handle dict format from msg_to_dict
        if isinstance(tc, dict):
            func_name = tc.get("function", {}).get("name", "")
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            tool_id = tc.get("id", "")
        else:
            func_name = tc.function.name
            args = json.loads(tc.function.arguments)
            tool_id = tc.id
        
        func = TOOL_MAP.get(func_name)
        if func:
            try:
                result = func(**args)
            except Exception as e:
                result = f"❌ Tool error: {e}"
        else:
            result = f"❌ Unknown tool: {func_name}"
        
        # Add tool result message
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": func_name,
            "content": str(result),
        })
    
    return {
        "messages": new_messages,
        "tool_calls_count": state.get("tool_calls_count", 0) + len(tool_calls),
    }


def routing_node(state: AgentState) -> Literal["tools", "end"]:
    """
    Router Node: Decides next step based on state.
    
    Routes:
    - If last message has tool_calls → go to tools_node
    - If tool_calls_count exceeded → end (prevent infinite loops)
    - Otherwise → end (final response ready)
    """
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_msg = messages[-1]
    
    # Check for tool calls
    has_tools = False
    if isinstance(last_msg, dict):
        has_tools = bool(last_msg.get("tool_calls"))
    elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        has_tools = True
    
    if has_tools:
        # Check loop prevention
        if state.get("tool_calls_count", 0) >= 10:
            return "end"
        return "tools"
    
    return "end"


# -----------------------------------------------------------------------
# Build LangGraph
# -----------------------------------------------------------------------
def build_graph():
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("llm", llm_node, name="LLM")
    builder.add_node("tools", tools_node, name="Tools")
    
    # Edges: START → llm → tools → llm (loop) → llm → END
    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")  # After tools, go back to LLM for final response
    
    # Conditional edges from llm
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Route after llm call: if tool_calls → tools, else → END."""
        messages = state["messages"]
        if not messages:
            return "__end__"
        
        last_msg = messages[-1]
        has_tools = False
        if isinstance(last_msg, dict):
            has_tools = bool(last_msg.get("tool_calls"))
        elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            has_tools = True
        
        if has_tools:
            # Check loop prevention
            if state.get("tool_calls_count", 0) >= 10:
                return "__end__"
            return "tools"
        return "__end__"
    
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        }
    )
    
    return builder.compile()


# -----------------------------------------------------------------------
# CLI Runner
# -----------------------------------------------------------------------
def main():
    graph = build_graph()
    
    print("=" * 60)
    print("🤖 GLM Agent — LangGraph State Machine")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Graph: llm → router → [tools|end]")
    print("   Type 'exit' or Ctrl+C to quit")
    print("=" * 60)
    print()

    SYSTEM = (
        "Kamu adalah asisten AI berbasis GLM-4.7. "
        "Gunakan bahasa Indonesia untuk merespons. "
        "Kamu bisa menggunakan tool calculator, read_file, dan list_directory kalau perlu. "
        "Respon langsung, jelas, dan helpful."
    )

    # Initial state
    initial_state = {
        "messages": [{"role": "system", "content": SYSTEM}],
        "tool_calls_count": 0,
    }

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

        # Add user message to state
        initial_state["messages"] = initial_state["messages"] + [
            {"role": "user", "content": user_input}
        ]

        # Run graph
        print("\n🤖 Agent: ", end="", flush=True)
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 50})
            
            # Get last assistant message
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
            elif isinstance(last_msg, dict):
                content = last_msg.get("content", "(no content)")
            else:
                content = str(last_msg)
            
            print(content)
            
            # Update state for next turn (keep conversation history)
            initial_state = result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Reset tool_calls_count on error
            initial_state["tool_calls_count"] = 0

        print()


if __name__ == "__main__":
    main()