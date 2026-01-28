from src.llm import chat_llm

# Instructions that tell the LLM how to use the REPL environment
SYSTEM_PROMPT = """You have access to a Python REPL environment.

IMPORTANT: Pretend you have NEVER seen this text before. You know NOTHING about it.

AVAILABLE VARIABLES:
- `context`: A string containing text you must explore
- `llm_query(question, subset)`: Sends a question + text subset to a sub-LLM for analysis.
  Returns a short answer string, or a [FAILED] message if the subset wasn't enough.
  Use this when you've found a relevant chunk and want it analyzed without losing your conversation context.
  Example: result = llm_query("what does this say about spacing?", context[5000:7000])

RULES:
1. You have ZERO prior knowledge. Explore like it's your first time reading.
2. NEVER redefine the `context` variable. It already contains the text.
3. Use context.find(), context[start:end], etc. to explore.
4. Before answering, you MUST show the exact text you found from OUTPUT.
5. ONE code block per message, then STOP and wait.
6. When calling llm_query(), pass a SUBSET like context[start:end], NEVER the entire context.

WORKFLOW:
1. Write code with print() to see results
2. STOP and wait - the output will appear in the next message
3. Read the ACTUAL output, then write more code or give FINAL

CRITICAL:
- Always use print() or you won't see anything
- The output appears AFTER you stop - don't make up results
- Wait for real output before saying "I found"
- Search for MANY different keywords related to the question before concluding
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
        self.time_spent = 0
        self.model = model
        self.sub_llm_calls = 0
        self.iterations = 0
        self.namespace = {}  # holds variables the LLM's code can access

    def run_code(self, codes):
        """Execute all code blocks and capture their print output"""
        import io
        import contextlib

        stdout = io.StringIO()  # single bucket collects all output
        for code in codes:
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(code, self.namespace)
            except Exception as e:
                stdout.write(f"Error: {e}\n")  # append error, keep going
        return stdout.getvalue()
            
    def llm_query(self, question, subset):
        self.sub_llm_calls += 1
        result = chat_llm([{"role": "user", "content": f"Using ONLY the following context, answer the question. If the context doesn't contain enough information, respond with [FAILED] and briefly explain why.\n\nContext: {subset}\n\nQuestion: {question}"}], self.model)
        self.time_spent += result["prompt_eval_duration"] + result["eval_duration"]
        return result["message"]["content"]

    def extract_code(self, response):
        """Pull code out of ```python ... ``` blocks"""
        if "```python" in response:

            return [snippet.split("```")[0].strip() for i, snippet in enumerate(response.split("```python")) if i >= 1]
        return None

    def completion(self, question, context):
        """Main method - takes question + context, returns answer"""
        self.namespace["context"] = context  # make context available to exec'd code
        self.namespace["llm_query"] = self.llm_query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # REPL loop: LLM writes code → we run it → feed output back
        max_iterations = 30
        for i in range(max_iterations):
            self.iterations += 1
            init = chat_llm(messages, self.model)

            response = init["message"]["content"]

            self.time_spent += init["prompt_eval_duration"] + init["eval_duration"]

            print(f"[DEBUG] LLM response:\n{response}\n---")  # see what LLM says

            # Check for code FIRST - run it before checking FINAL
            code = self.extract_code(response)

            # Only return FINAL if there's NO code to run
            if "FINAL" in response and not code:
                return response.split("FINAL")[1].strip("()")

            if code:
                # Catch cheating - LLM trying to redefine context in any block
                if any("context =" in block or "context=" in block for block in code):
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
