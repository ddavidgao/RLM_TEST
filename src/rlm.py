from src.llm import chat_llm

# Instructions that tell the LLM how to use the REPL environment
SYSTEM_PROMPT = """You have access to a Python REPL environment.

IMPORTANT: Pretend you have NEVER seen this text before. You know NOTHING about it.

AVAILABLE VARIABLES:
- `context`: A string containing text you must explore

RULES:
1. You have ZERO prior knowledge. Explore like it's your first time reading.
2. NEVER redefine the `context` variable. It already contains the text.
3. Use context.find(), context[start:end], etc. to explore.
4. Before answering, you MUST show the exact text you found from OUTPUT.
5. ONE code block per message, then STOP and wait.

WORKFLOW:
1. Write code with print() to see results
2. STOP and wait - the output will appear in the next message
3. Read the ACTUAL output, then write more code or give FINAL

CRITICAL:
- Always use print() or you won't see anything
- The output appears AFTER you stop - don't make up results
- Wait for real output before saying "I found"
- After 2-3 searches, you should have enough - give FINAL
- Don't read the entire document chunk by chunk

EXAMPLE:
```python
idx = context.find("keyword")
print(idx)
print(context[idx:idx+300])
```
(STOP HERE - output will appear next)

Once you have found relevant text and formed an interpretation, you MUST end with:
FINAL(your interpretation here)"""


class RLM:
    def __init__(self, model="qwen3-coder:30b"):
        self.model = model
        self.namespace = {}  # holds variables the LLM's code can access

    def run_code(self, code):
        """Execute code and capture its print output"""
        import io
        import contextlib

        stdout = io.StringIO()  # bucket to catch prints

        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, self.namespace)  # run code with access to namespace
            return stdout.getvalue()
        except Exception as e:
            return f"Error: {e}"  # return error instead of crashing
        
    def llm_query(self, question, subset):
        payload = {
            "model": self.model,
            "prompt": "using ONLY the following context: {subset}, answer this question: {question}. return it as a print if so, otherwise indicate that more context is needed by returning -1.",
            "stream": False
        }
        response = requests.post(url, json=payload)
        data = response.json()
        return data["response"]

    def extract_code(self, response):
        """Pull code out of ```python ... ``` blocks"""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
            return code.strip()
        return None

    def completion(self, question, context):
        """Main method - takes question + context, returns answer"""
        self.namespace["context"] = context  # make context available to exec'd code

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # REPL loop: LLM writes code → we run it → feed output back
        max_iterations = 30
        for i in range(max_iterations):
            #delegate to sub-llm
            subset = self.llm_query(question, context)
            if subset == "-1":
                #more context is needed
                return "ERROR: More context is needed"
            else:
                #use the subset to answer the question
                messages.append({"role": "user", "content": f"Context: {subset}"})
                response = chat_llm(messages, self.model)
            print(f"[DEBUG] LLM response:\n{response}\n---")  # see what LLM says

            # Check for code FIRST - run it before checking FINAL
            code = self.extract_code(response)

            # Only return FINAL if there's NO code to run
            if "FINAL" in response and not code:
                return response.split("FINAL")[1].strip("()")

            if code:
                # Catch cheating - LLM trying to redefine context
                if "context =" in code or "context=" in code:
                    output = "ERROR: You cannot redefine 'context'. Use context.find() to search the existing text."
                else:
                    output = self.run_code(code)
            else:
                output = "(No code found in response)"

            # Add this exchange to history so LLM sees what happened
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Output:\n{output}"})

        # Hit max iterations without FINAL
        return "ERROR: Max iterations reached without conclusion"
