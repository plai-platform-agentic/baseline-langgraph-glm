# baseline-langgraph-glm

Baseline LangGraph agent with GLM-4.7 as brain. Supports **OpenSpec** for spec-driven development.

## Overview

A LangGraph-based AI agent using GLM-4.7 (via ZAI) with tool calling and OpenSpec document support — spec-driven development for autonomous agent workflows.

## Features

- **LangGraph State Graph** — Agent orchestration with message history
- **GLM-4.7 Integration** — Uses ZAI API token from OpenClaw config
- **Function Calling** — Built-in tools: `calculator`, `read_file`, `list_directory`
- **OpenSpec Support** — `create_spec`, `validate_spec`, `save_spec` tools
- **Indonesian Language** — Optimized for Bahasa Indonesia responses
- **Hermes-inspired** — Follows the LLM→Tools→LLM loop pattern

## OpenSpec Format

OpenSpec is a structured specification format for spec-driven development:

```markdown
## Purpose

[What this system/component should do]

## Requirements

### Requirement: [Name]
System SHALL [expected behavior]

#### Scenario: [Scenario Name]
- **WHEN** [trigger condition]
- **THEN** [expected result]
```

## Quick Start

```bash
cd baseline-langgraph-glm
python3 glm_agent.py
```

### Prerequisites

- Python 3.11+
- OpenClaw auth token (auto-loaded from `~/.openclaw/`)
- GLM coding plan access

## Architecture

```
START → llm (GLM-4.7 brain)
         ↓
    [tool_calls?] → YES → tools (execute) → llm (loop)
                   → NO  → END (final response)
```

## OpenSpec Tools

| Tool | Description |
|------|-------------|
| `create_spec` | Generate new OpenSpec doc from description |
| `validate_spec` | Check OpenSpec format compliance |
| `save_spec` | Write OpenSpec to file |
| `read_file` | Read local files (specs, code) |
| `calculator` | Math expressions |
| `list_directory` | Browse filesystem |

## Project Structure

```
baseline-langgraph-glm/
├── glm_agent.py      # Main agent with OpenSpec tools
├── test_glm.py       # Connection test script
├── README.md         # This file
└── LICENSE           # MIT License
```

## Usage

### Basic Q&A

```bash
$ python3 glm_agent.py

============================================================
🤖 GLM Agent — OpenSpec-Enabled LangGraph
   Model: glm-4.7
   Tools: calculator, read_file, list_directory,
          create_spec, validate_spec, save_spec
============================================================

🧑 You: Halo, siapa kamu?
🤖 Agent: Halo! Saya adalah asisten AI berbasis GLM-4.7...

🧑 You: Berapa 15 * 7?
🤖 Agent: [calculator] → ✅ 15 × 7 = 105
```

### OpenSpec Workflow

```bash
🧑 You: Buat OpenSpec untuk sistem login
🤖 Agent: [create_spec] → 📋 OpenSpec Created: Login System
    (generates structured spec with Requirements & Scenarios)

🧑 You: Simpan spec ke ./specs/login.md
🤖 Agent: [save_spec] → 💾 Spec saved: ./specs/login.md

🧑 You: Validasi spec yang baru dibuat
🤖 Agent: [validate_spec] → ✅ OpenSpec Valid
    - Requirements: 3
    - Scenarios: 5
    - Format: Compliant with OpenSpec 1.0
```

## Configuration

The agent automatically loads GLM credentials from OpenClaw:
- Auth token: `~/.openclaw/agents/main/agent/auth-profiles.json`
- API config: `~/.openclaw/openclaw.json`

## License

MIT License
