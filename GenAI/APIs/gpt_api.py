import os

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


def call_gpt(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """Call OpenAI API using the official SDK and return generated text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable before running.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choices = response.choices
    if not choices:
        return ""

    content = choices[0].message.content or ""
    return content.strip()


if __name__ == "__main__":
    user_prompt = input("Enter prompt for GPT: ").strip()
    output = call_gpt(user_prompt)
    print("\nGPT response:\n")
    print(output)
