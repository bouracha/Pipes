import requests
import json
import os

def ping_claude_api():
    api_key = os.getenv("API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": "Hello, Claude! How are you today?"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    result = ping_claude_api()
    if result:
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                print(f"Claude says: {item.get('text')}")

        print("\nFull response:")
        print(json.dumps(result, indent=2))
