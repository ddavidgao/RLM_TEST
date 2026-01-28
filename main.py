import json
import os
from datetime import datetime
from src.rlm import RLM
from src.llm import chat_llm

# --- BASELINE ANSWER (what the RLM should find) ---
EXPECTED_ERRORS = [
    "Use 'by' not 'x' for dimensions: '8 by 10' not '8x10'",
    "Names of vessels should be in italic: _Oregon_",
    "Naval vessels use 'U. S. S.' prefix: U. S. S. _Oregon_",
    "Misspelling: 'San Francisco' not 'San Fransisco'",
]

RESULTS_FILE = "run_results.json"


def score_answer(answer, expected_errors):
    """Use the LLM to check how many expected errors were found in the answer."""
    checklist = "\n".join(f"{i+1}. {e}" for i, e in enumerate(expected_errors))

    prompt = f"""You are a grader. Compare the ANSWER against the CHECKLIST of errors that should have been identified.

For each checklist item, respond with FOUND or MISSED. Then give a total score as SCORE: X/{len(expected_errors)}

CHECKLIST:
{checklist}

ANSWER:
{answer}"""

    result = chat_llm([{"role": "user", "content": prompt}], "qwen3-coder:30b")
    grading = result["message"]["content"]
    print(f"\n[GRADING]\n{grading}\n")

    # Parse score from response
    for line in grading.split("\n"):
        if "SCORE:" in line.upper():
            try:
                score_part = line.split(":")[-1].strip()
                found = int(score_part.split("/")[0])
                return found / len(expected_errors)
            except (ValueError, IndexError):
                pass
    return 0.0


def save_result(run_data):
    """Append run data to results JSON file."""
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    results.append(run_data)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def plot_results():
    """Generate scatter plot from all saved runs."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(RESULTS_FILE):
        print("No results to plot yet.")
        return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    if len(results) < 1:
        print("Need at least 1 run to plot.")
        return

    df = pd.DataFrame(results)

    # Color: oldest = red, newest = blue
    n = len(df)
    colors = [(1 - i/(max(n-1, 1)), 0, i/(max(n-1, 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["total_llm_calls"],
        df["score"],
        c=colors,
        s=100,
        edgecolors="black",
        linewidth=0.5,
    )

    # Label each point with run number
    for i, row in df.iterrows():
        ax.annotate(f"#{row['run_id']}", (row["total_llm_calls"], row["score"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Total LLM Calls (root + sub)")
    ax.set_ylabel("Score (errors found / total)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("RLM Performance: Effort vs Accuracy")
    ax.grid(True, alpha=0.3)

    # Legend for color
    ax.text(0.02, 0.98, "Red = older runs, Blue = newer runs",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            style="italic", color="gray")

    plt.tight_layout()
    plt.savefig("rlm_performance.png", dpi=150)
    print("Plot saved to rlm_performance.png")
    plt.show()


# --- RUN ---
rlm = RLM()

with open("gpo_manual.txt", "r", encoding="utf-8") as f:
    context = f.read()

print(f"Context size: {len(context)} characters")

question = """I need to write a government document with this sentence:
"The 8x10 inch photograph shows the Navy's steamship Oregon docked at San Fransisco harbor."
According to this manual, what formatting errors does this sentence contain and how should it be corrected?"""

answer = rlm.completion(question, context)
print(f"\nFINAL ANSWER: {answer}")

# Score the answer
score = score_answer(answer, EXPECTED_ERRORS)

# Calculate stats
total_calls = rlm.iterations + rlm.sub_llm_calls
time_seconds = rlm.time_spent / 1_000_000_000  # nanoseconds to seconds

# Determine run ID
run_id = 1
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        existing = json.load(f)
        run_id = len(existing) + 1

run_data = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "total_llm_calls": total_calls,
    "root_iterations": rlm.iterations,
    "sub_llm_calls": rlm.sub_llm_calls,
    "time_seconds": round(time_seconds, 2),
    "score": round(score, 2),
    "answer_preview": answer[:200],
}

save_result(run_data)

print(f"\n--- RUN #{run_id} STATS ---")
print(f"Root iterations: {rlm.iterations}")
print(f"Sub-LLM calls:   {rlm.sub_llm_calls}")
print(f"Total LLM calls: {total_calls}")
print(f"Time spent:       {time_seconds:.1f}s")
print(f"Score:            {score:.0%}")
print(f"Results saved to {RESULTS_FILE}")

# Plot all results
plot_results()
