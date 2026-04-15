from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
import pyautogui
import time

load_dotenv()

# -------- TOOL --------
@tool
def create_folder(folder_name: str) -> str:
    """Create a folder"""
    time.sleep(1)
    pyautogui.hotkey('win', 'e')
    time.sleep(5)
    pyautogui.press('right')
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'shift', 'n')
    time.sleep(2)
    pyautogui.typewrite(folder_name)
    pyautogui.press('enter')
    return f"Folder '{folder_name}' created."

@tool
def create_file(file_name: str, file_extension: str, content: str) -> str:
    """Create a file"""

    with open(f"{file_name}.{file_extension}", "w") as f:
        f.write(content)  # Create a file with the specified content


@tool
def read_file(file_name: str, file_extension: str) -> str:
    """Read a file's content"""
    with open(f"{file_name}.{file_extension}", "r") as f:
        return f.read()

# -------- LLM --------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# -------- AGENT (NEW API) --------
agent = create_agent(
    model=llm,
    tools=[create_folder, create_file, read_file],
    system_prompt="You are an assistant that can use tools."
)

# -------- RUN --------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Read the python file named test and tell me what is the content in it."}
    ]
})

print(response)