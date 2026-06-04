#!/usr/bin/env python3
"""
GLM Agent - OpenSpec-Enabled LangGraph Implementation
Spec-driven development with OpenSpec format.

OpenSpec Format:
    ## Requirement: [Name]
    System SHALL [behavior]

    #### Scenario: [Name]
    - **WHEN** [condition]
    - **THEN** [expected result]

LangGraph State Graph with:
- Typed State (AgentState)
- Message history with add_messages reducer
- Conditional edges (tool calling routing)
- Separate nodes: llm, tools, routing
- OpenSpec tools: create_spec, read_spec, validate_spec

Run:
    python3 glm_agent.py
"""

import json
import re
import sys
from typing import Annotated, Literal
from typing_extensions import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

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
    current_spec: str | None  # Active OpenSpec document being worked on


# -----------------------------------------------------------------------
# OpenSpec Format Constants
# -----------------------------------------------------------------------
OPENSPEC_TEMPLATE = """# {title}

## Purpose

{purpose}

## Requirements

### Requirement: {req_name}
System SHALL {req_behavior}

#### Scenario: {scenario_name}
- **WHEN** {when_condition}
- **THEN** {then_result}
"""

OPENSPEC_PATTERNS = {
    "requirement": r"^### Requirement: (.+)$",
    "scenario": r"^#### Scenario: (.+)$",
    "when": r"^\*\*WHEN\*\* (.+)$",
    "then": r"^\*\*THEN\*\* (.+)$",
    "shall": r"SHALL (.+)",
}


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
    },
    {
        "type": "function",
        "function": {
            "name": "create_spec",
            "description": "Create a new OpenSpec specification document from a description. Generates structured spec with Requirements and Scenarios in OpenSpec format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the specification"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Purpose/goal of the specification"
                    },
                    "requirements": {
                        "type": "string",
                        "description": "Comma-separated list of requirements to include"
                    }
                },
                "required": ["title", "purpose"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_spec",
            "description": "Validate an OpenSpec document structure. Checks for proper format: Requirements with SHALL, Scenarios with WHEN/THEN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec_content": {
                        "type": "string",
                        "description": "OpenSpec document content to validate"
                    }
                },
                "required": ["spec_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_spec",
            "description": "Save an OpenSpec document to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to save the spec"
                    },
                    "content": {
                        "type": "string",
                        "description": "OpenSpec document content"
                    }
                },
                "required": ["path", "content"]
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

def create_spec(title: str, purpose: str, requirements: str = "") -> str:
    """Generate an OpenSpec specification document."""
    lines = [f"# {title}", "", "## Purpose", "", purpose, "", "## Requirements", ""]
    
    if requirements:
        reqs = [r.strip() for r in requirements.split(",")]
        for i, req in enumerate(reqs, 1):
            lines.extend([
                f"### Requirement: {req}",
                f"System SHALL {req.lower()}.",
                "",
                f"#### Scenario: Primary behavior",
                f"- **WHEN** user interacts with the system",
                f"- **THEN** the system {req.lower()}",
                ""
            ])
    else:
        lines.extend([
            "### Requirement: TODO",
            "System SHALL define clear behavior.",
            "",
            "#### Scenario: Default scenario",
            "- **WHEN** TODO",
            "- **THEN** TODO",
            ""
        ])
    
    spec = "\n".join(lines)
    return f"📋 **OpenSpec Created:** `{title}`\n\n```markdown\n{spec}\n```"

def validate_spec(spec_content: str) -> str:
    """Validate OpenSpec document structure."""
    issues = []
    warnings = []
    
    # Check for required sections
    if "## Purpose" not in spec_content:
        issues.append("Missing '## Purpose' section")
    if "## Requirements" not in spec_content:
        issues.append("Missing '## Requirements' section")
    
    # Parse requirements
    reqs = re.findall(OPENSPEC_PATTERNS["requirement"], spec_content, re.MULTILINE)
    if not reqs:
        issues.append("No Requirements found (use '### Requirement: <name>')")
    
    # Check each requirement has SHALL
    for req in reqs:
        req_block = re.search(rf"### Requirement: {re.escape(req)}.*?(?=### |$)", spec_content, re.DOTALL)
        if req_block and "SHALL" not in req_block.group(0).upper():
            issues.append(f"Requirement '{req}' missing SHALL clause")
    
    # Parse scenarios
    scenarios = re.findall(OPENSPEC_PATTERNS["scenario"], spec_content, re.MULTILINE)
    for scenario in scenarios:
        sc_block = re.search(rf"#### Scenario: {re.escape(scenario)}.*?(?=#### |## |$)", spec_content, re.DOTALL)
        if sc_block:
            block = sc_block.group(0)
            if "**WHEN**" not in block:
                issues.append(f"Scenario '{scenario}' missing **WHEN** clause")
            if "**THEN**" not in block:
                issues.append(f"Scenario '{scenario}' missing **THEN** clause")
    
    # Build response
    if not issues and not warnings:
        return (
            f"✅ **OpenSpec Valid**\n"
            f"- Requirements: {len(reqs)}\n"
            f"- Scenarios: {len(scenarios)}\n"
            f"- Format: Compliant with OpenSpec 1.0"
        )
    elif not issues:
        return (
            f"⚠️ **OpenSpec Valid with warnings**\n"
            f"- Requirements: {len(reqs)}\n"
            f"- Scenarios: {len(scenarios)}\n"
            + "\n".join(f"  - {w}" for w in warnings)
        )
    else:
        return (
            f"❌ **OpenSpec Invalid**\n"
            + "\n".join(f"  - {i}" for i in issues)
        )

def save_spec(path: str, content: str) -> str:
    """Save OpenSpec document to file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"💾 **Spec saved:** `{path}`"
    except Exception as e:
        return f"❌ Cannot save spec: {e}"

TOOL_MAP = {
    "calculator": calculator,
    "read_file": read_file,
    "list_directory": list_directory,
    "create_spec": create_spec,
    "validate_spec": validate_spec,
    "save_spec": save_spec,
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
    
    api_messages = []
    for msg in messages:
        if hasattr(msg, "model"):
            api_messages.append(msg_to_dict(msg))
        elif isinstance(msg, dict):
            api_messages.append(msg)
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    choice = call_glm(api_messages)
    msg = choice.message
    
    if choice.finish_reason == "tool_calls":
        return {
            "messages": [msg_to_dict(msg)],
            "tool_calls_count": state.get("tool_calls_count", 0),
            "current_spec": state.get("current_spec"),
        }
    elif choice.finish_reason == "stop":
        content = extract_content(msg)
        return {
            "messages": [{"role": "assistant", "content": content}],
            "tool_calls_count": 0,
            "current_spec": state.get("current_spec"),
        }
    elif choice.finish_reason == "length":
        content = extract_content(msg)
        return {
            "messages": [{"role": "assistant", "content": content + "\n[response truncated]"}],
            "tool_calls_count": 0,
            "current_spec": state.get("current_spec"),
        }
    else:
        return {
            "messages": [{"role": "assistant", "content": f"[finish: {choice.finish_reason}]"}],
            "tool_calls_count": 0,
            "current_spec": state.get("current_spec"),
        }


def tools_node(state: AgentState) -> AgentState:
    """
    Tools Node: Executes tool calls from LLM response.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    tool_calls = None
    if isinstance(last_msg, dict):
        tool_calls = last_msg.get("tool_calls")
    elif hasattr(last_msg, "tool_calls"):
        tool_calls = last_msg.tool_calls
    
    if not tool_calls:
        return state
    
    new_messages = []
    for tc in tool_calls:
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
        
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": func_name,
            "content": str(result),
        })
    
    return {
        "messages": new_messages,
        "tool_calls_count": state.get("tool_calls_count", 0) + len(tool_calls),
        "current_spec": state.get("current_spec"),
    }


# -----------------------------------------------------------------------
# Build LangGraph
# -----------------------------------------------------------------------
def build_graph():
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(AgentState)
    
    builder.add_node("llm", llm_node, name="LLM")
    builder.add_node("tools", tools_node, name="Tools")
    
    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")
    
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
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
    print("🤖 GLM Agent — OpenSpec-Enabled LangGraph")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Tools: calculator, read_file, list_directory,")
    print(f"          create_spec, validate_spec, save_spec")
    print(f"   Graph: llm → [tools|end]")
    print("   Type 'exit' or Ctrl+C to quit")
    print("=" * 60)
    print()

    SYSTEM = (
        "Kamu adalah asisten AI berbasis GLM-4.7 dengan kemampuan OpenSpec.\n"
        "Gunakan bahasa Indonesia untuk merespons.\n\n"
        "**Tools yang tersedia:**\n"
        "- calculator: untuk perhitungan matematika\n"
        "- read_file: untuk membaca file lokal\n"
        "- list_directory: untuk melihat isi direktori\n"
        "- create_spec: untuk membuat dokumen OpenSpec baru\n"
        "- validate_spec: untuk memvalidasi format OpenSpec\n"
        "- save_spec: untuk menyimpan OpenSpec ke file\n\n"
        "**Format OpenSpec:**\n"
        "## Purpose\n\n[deskripsi tujuan]\n\n"
        "## Requirements\n\n"
        "### Requirement: [Nama]\n"
        "System SHALL [perilaku yang diharapkan]\n\n"
        "#### Scenario: [Nama Skenario]\n"
        "- **WHEN** [kondisi trigger]\n"
        "- **THEN** [hasil yang diharapkan]\n\n"
        "Respon langsung, jelas, dan helpful."
    )

    initial_state = {
        "messages": [{"role": "system", "content": SYSTEM}],
        "tool_calls_count": 0,
        "current_spec": None,
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

        initial_state["messages"] = initial_state["messages"] + [
            {"role": "user", "content": user_input}
        ]

        print("\n🤖 Agent: ", end="", flush=True)
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 50})
            
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
            elif isinstance(last_msg, dict):
                content = last_msg.get("content", "(no content)")
            else:
                content = str(last_msg)
            
            print(content)
            initial_state = result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            initial_state["tool_calls_count"] = 0

        print()


if __name__ == "__main__":
    main()
