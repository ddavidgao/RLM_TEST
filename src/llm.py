# requests = library for making HTTP calls (like a browser, but in code)
import requests


def chat_llm(messages, model="qwen3-coder:30b"):
    """
    Send a conversation to Ollama and get a response.

    messages = list of dicts like:
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    """
    # URL = address of the server endpoint (like a mailing address)
    url = "http://localhost:11434/api/chat"

    # PAYLOAD = the data we send (like contents of a package)
    # This dict gets converted to JSON automatically by requests
    payload = {
        "model": model,           # which LLM to use
        "messages": messages,     # the conversation so far
        "stream": False           # False = wait for complete response
                                  # True = get word-by-word (harder to handle)
    }

    # POST = "here's data, give me a response" (vs GET = "just give me data")
    # json=payload automatically converts our dict to JSON string
    response = requests.post(url, json=payload)

    # .json() parses the JSON string response into a Python dict
    # Server sends: '{"message": {"role": "assistant", "content": "..."}}'
    # After .json(): {"message": {"role": "assistant", "content": "..."}}
    data = response.json()

    # Navigate the nested dict to extract just the text we want
    return data["message"]["content"]


def ask_llm(prompt):
    """
    Simple single-prompt call (no conversation history).
    Good for one-off questions.
    """
    # Different endpoint - /api/generate vs /api/chat
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen3-coder:30b",
        "prompt": prompt,         # just a string, not a messages list
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = response.json()

    # Different response structure - just "response", not "message.content"
    return data["response"]
