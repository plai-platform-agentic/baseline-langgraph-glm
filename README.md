# baseline-langgraph-glm

Baseline LangGraph agent with GLM-4.7 as brain. Inspired by Hermes Agent architecture.

## Overview

A simple but functional LangGraph-based AI agent that uses GLM-4.7 (via ZAI coding plan) as the main brain, with tool calling capabilities.

## Features

- **LangGraph State Graph** - Agent orchestration with message history
- **GLM-4.7 Integration** - Uses ZAI API token from OpenClaw config
- **Function Calling** - Built-in tools: `calculator`, `read_file`
- **Indonesian Language** - Optimized for Bahasa Indonesia responses
- **Hermes-inspired** - Follows the LLM→Tools→LLM loop pattern

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

## Project Structure

```
baseline-langgraph-glm/
├── glm_agent.py      # Main agent - terminal Q&A
├── test_glm.py       # Connection test script
├── README.md         # This file
└── LICENSE           # MIT License
```

## Usage

```bash
$ python3 glm_agent.py

============================================================
🤖 GLM Agent — LangGraph Terminal Q&A (Baseline)
   Model: glm-4.7
   API: https://api.z.ai/api/coding/paas/v4
   Type 'exit' or Ctrl+C to quit
============================================================

🧑 You: Halo, siapa kamu?
🤖 Agent: Halo! Saya adalah asisten AI berbasis GLM-4.7...

🧑 You: Berapa 15 * 7?
🤖 Agent: [calculator] → ✅ 15 × 7 = 105

🧑 You: exit
👋 Goodbye!
```

## Configuration

The agent automatically loads GLM credentials from OpenClaw:
- Auth token: `~/.openclaw/agents/main/agent/auth-profiles.json`
- API config: `~/.openclaw/openclaw.json`

## License

MIT License