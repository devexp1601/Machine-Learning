import os

import google.generativeai as genai


def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_output_tokens: int = 256,
) -> str:
    """Call Gemini API using the official SDK and return generated text."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable before running.")

    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    client = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    response = client.generate_content(prompt)

    text = getattr(response, "text", "")
    return text.strip() if text else ""


if __name__ == "__main__":
    user_prompt = input("Enter prompt for Gemini: ").strip()
    output = call_gemini(user_prompt)
    print("\nGemini response:\n")
    print(output)
