import os

from anthropic import Anthropic


def call_claude(
    prompt: str,
    model: str = "claude-3-5-haiku-latest",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """Call Claude API using the official SDK and return generated text."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable before running.")

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    text_chunks = [block.text for block in message.content if getattr(block, "type", "") == "text"]
    return "\n".join(text_chunks).strip()


if __name__ == "__main__":
    user_prompt = input("Enter prompt for Claude: ").strip()
    output = call_claude(user_prompt)
    print("\nClaude response:\n")
    print(output)
