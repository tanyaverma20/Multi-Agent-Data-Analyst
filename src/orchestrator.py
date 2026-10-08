# File: src/orchestrator.py
# Updated orchestrator to create A2ABus and wire agents. Use this orchestrator to start the pipeline
import os
import sys
import time
import pandas as pd

# ensure src is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "src" else BASE_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

# import bus and agents
from core.a2a_bus import A2ABus
from agents.profiler_agent import ProfilerAgent
from agents.eda_agent import EDAAgent
from agents.model_agent import ModelAgent
from agents.verifier_agent import VerifierAgent
from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent


def main():
    print("=== ORCHESTRATOR STARTED ===")

    memory = MemoryTools()
    file_tool = FileTools()

    # instantiate A2A bus with persistence into memory
    a2a = A2ABus(memory=memory, persist=True)

    # register agents
    profiler = ProfilerAgent(a2a_bus=a2a)
    eda = EDAAgent(output_dir="reports")
    model_agent = ModelAgent(a2a_bus=a2a)
    verifier = VerifierAgent(a2a_bus=a2a)
    notebook_agent = NotebookSynthesizerAgent(a2a_bus=a2a)

    # 1) Locate sample CSV
    candidates = [
        os.path.join(BASE_DIR, "sample.csv"),
        os.path.join(PROJECT_ROOT, "sample.csv"),
        os.path.join(PROJECT_ROOT, "data", "sample.csv"),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)

    if not path:
        print("Sample CSV not found in any of:", candidates)
        return

    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # 2) Run profiler (this will publish a message to eda)
    print("Running profiler...")
    profiler_out = profiler.run(df)
    print("Profiler:", profiler_out.get("status"))

    # 3) EDA agent run
    print("Running EDA agent...")
    eda_out = eda.run(df)
    print("EDA:", eda_out.get("status"))

    # Publish eda.completed to bus for ModelAgent
    a2a.publish(
        from_agent="eda",
        to="model",
        topic="eda.completed",
        payload=eda_out,
    )

    # 4) Model agent will be triggered by EDA (A2A). Fetch and run
    print("ModelAgent polling for messages (run if EDA completed)...")
    msgs = a2a.fetch("model", consume=True)
    ran_model = None
    for m in msgs:
        if m.get("topic") == "eda.completed":
            target = df.columns[-1]
            print("Training target chosen:", target)
            ran_model = model_agent.run(df, target_col=target)
            print("Model run status:", ran_model.get("status"))

    if not ran_model:
        print("Running model manually with last column as target...")
        target = df.columns[-1]
        ran_model = model_agent.run(df, target_col=target)

    # 5) Verifier agent polls and verifies
    print("Verifier polling messages and running verification...")
    verifier_poll = verifier.poll_messages_and_run()
    if not verifier_poll:
        mo = memory.load("model_output")
        if mo:
            verifier_res = verifier.run(mo)
            print("Verifier manual run:", verifier_res.get("status"))
    else:
        print("Verifier auto-run result:", verifier_poll.get("status"))

    # 6) Notebook generator: when all pieces are available, create notebook
    mem_all = memory.load_all()
    profiler_output = mem_all.get("profiler_output")
    eda_output = mem_all.get("eda_output")
    model_output = mem_all.get("model_output")
    verifier_output = mem_all.get("verifier_output")

    if profiler_output and eda_output and model_output and verifier_output:
        print("Generating notebook...")
        nb_res = notebook_agent.run(profiler_output, eda_output, model_output, verifier_output)
        print("Notebook result:", nb_res)
    else:
        print("Not all components available in memory; notebook generation skipped.")

    print("=== ORCHESTRATOR FINISHED ===")


if __name__ == "__main__":
    main()
