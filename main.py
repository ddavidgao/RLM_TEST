from src.rlm import RLM

rlm = RLM()

with open("gpo_manual.txt", "r", encoding="utf-8") as f:
    context = f.read()

print(f"Context size: {len(context)} characters")

# Hard question - requires synthesizing multiple rules from different sections
question = """I need to write a government document with this sentence:
"The 8x10 inch photograph shows the Navy's steamship Oregon docked at San Fransisco harbor."
According to this manual, what formatting errors does this sentence contain and how should it be corrected?"""

answer = rlm.completion(question, context)
print(f"\nFINAL ANSWER: {answer}")