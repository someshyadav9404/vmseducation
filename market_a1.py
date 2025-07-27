import json
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from duckduckgo_search import DDGS
import time # Import the time module

# --- 1. SETUP ---
API_KEY = "AIzaSyCY0UJj2iAnGirzqqRAMf8vhnFWyIb8g2w" 

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Google AI. Please check your API key. Details: {e}")
    exit()

# --- 2. DEFINE THE AGENT'S TOOLS ---

# TOOL 1: Web Search (with rate limit fix)
def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the results."""
    print(f"--- Calling Tool: web_search(query='{query}') ---")
    time.sleep(10) # Pause for 2 seconds to avoid being rate-limited.
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return json.dumps(results) if results else "No results found."

# TOOL 2: Read Webpage Content
def read_webpage_content(url: str) -> str:
    """Reads the primary text content from a webpage URL."""
    print(f"--- Calling Tool: read_webpage_content(url='{url}') ---")
    time.sleep(1) # Also a good idea to add a small delay here
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:8000]
    except requests.RequestException as e:
        return f"Error reading URL {url}: {e}"

# TOOL 3: Save to File
def save_to_file(filename: str, content: str) -> str:
    """Saves content to a file to remember information across steps."""
    print(f"--- Calling Tool: save_to_file(filename='{filename}') ---")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully saved to {filename}."
    except Exception as e:
        return f"Error saving file: {e}"

# --- 3. CREATE THE AGENT ---

SYSTEM_INSTRUCTION = """
You are a world-class market research analyst AI. Your goal is to execute a user's request by breaking it down into a step-by-step plan.
You must use the tools provided to gather information. Do not answer from your own knowledge.
For each step, think about what you need to do, choose the best tool, and execute it.
Observe the results from the tool and then think about the next step.
When gathering information about multiple items (like competitors), analyze one at a time.
Use the save_to_file tool to save your intermediate findings for each competitor.
Once you have gathered all the necessary information, synthesize it into a final, well-structured report that directly answers the user's original request.
Do not simply output the raw data from the tools.
"""

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[web_search, read_webpage_content, save_to_file],
    system_instruction=SYSTEM_INSTRUCTION
)

chat = model.start_chat(enable_automatic_function_calling=True)

def run_agent(prompt: str):
    """Runs the agentic loop."""
    print(f"User Goal: {prompt}\n" + "-"*20)
    response = chat.send_message(prompt)
    print("\n" + "-"*20 + "\n--- Final Report ---")
    print(response.text)

# --- 4. RUN THE AGENT WITH OUR COMPLEX GOAL ---

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("!!! ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual Google AI API key in the script.")
    else:
        user_goal = """
        Please act as a market research analyst.
        Our company is launching a new productivity app called 'FocusFlow'.
        Your task is to generate a competitive analysis report.
        
        The report must:
        1. Identify 2-3 of the main competitors in the productivity/note-taking space.
        2. For each competitor, find and summarize:
           - Their key features.
           - Their pricing model (e.g., free tier, subscription costs).
           - Common user complaints (search for reviews or Reddit comments).
        3. Conclude with a strategic recommendation for FocusFlow based on your findings.
        """
        run_agent(user_goal)