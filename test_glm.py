#!/usr/bin/env python3
"""
Test GLM connection using OpenClaw's token from auth-profiles.json
The GLM coding plan uses: https://api.z.ai/api/coding/paas/v4
"""

import json

# Load token from OpenClaw config
def get_zai_token():
    auth_file = "/root/.openclaw/agents/main/agent/auth-profiles.json"
    with open(auth_file) as f:
        data = json.load(f)
    return data["profiles"]["zai:default"]["key"]

def get_zai_base_url():
    config_file = "/root/.openclaw/openclaw.json"
    with open(config_file) as f:
        data = json.load(f)
    return data["models"]["providers"]["zai"]["baseUrl"]

token = get_zai_token()
base_url = get_zai_base_url()

print(f"✅ Loaded token: {token[:20]}...{token[-6:]}")
print(f"✅ Base URL: {base_url}")

# Test with OpenAI SDK (compatible with ZAI's OpenAI-like API)
from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url=base_url,
)

print("\n🔄 Testing GLM-4.7-flash...")
try:
    response = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[
            {"role": "system", "content": "Kamu assistant bilingual Indonesia/English. Respon singkat saja."},
            {"role": "user", "content": "Apa itu GLM? jawab singkat."}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    print(f"\n✅ SUCCESS!")
    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")