import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def setup_client():
    """Initialize Gemini client with API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")
    
    genai.configure(api_key=api_key)


def extract_text(response):
    """Safely extract text from Gemini response."""
    try:
        return response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, KeyError):
        return ""


def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
) -> str:
    """Call Gemini API and return generated text."""
    
    setup_client()

    generation_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    model_client = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config
    )

    response = model_client.generate_content(prompt)

    # Extract text safely
    text = extract_text(response)

    # Debug info
    try:
        finish_reason = response.candidates[0].finish_reason
    except:
        finish_reason = "UNKNOWN"

    print(f"\n🔍 Finish reason: {finish_reason}")
    print(f"📦 Raw response: {response}\n")

    return text.strip()


if __name__ == "__main__":
    user_prompt = input("Enter prompt for Gemini: ").strip()

    if not user_prompt:
        print("❌ Prompt cannot be empty.")
        exit()

    output = call_gemini(user_prompt)

    print("\n🤖 Gemini Response:\n")
    print(output if output else "⚠️ No response generated.")