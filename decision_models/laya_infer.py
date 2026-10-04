"""compare laya on cpu and gpu"""

import gc
import os
import pprint
import time
import warnings

import laya
import torch

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # suppress loading bars
warnings.filterwarnings("ignore", message="laya:*")


MODEL = "convaiinnovations/laya"
TEXT = (
    "The company announced that its CFO will retire next quarter. "
    "The current controller will serve as interim CFO while the board searches "
    "for a permanent successor."
)
QUESTIONS = {
    "triage": {
        "type": "choice",
        "instructions": "Classify this filing excerpt for analyst review.",
        "criteria": {
            "routine": "Ordinary update with no clear investor impact.",
            "potentially_material": "Potentially significant event that merits review.",
            "unclear": "Not enough information to classify confidently.",
        },
    },
}

# NOTE: sync kernel often for accurate timing


def run(device, repeats=3):
    start = time.perf_counter()
    agent = laya.load(MODEL, device=device)
    load_seconds = time.perf_counter() - start

    agent.predict(TEXT, QUESTIONS)  # warm-up run, doesn't count towards timing
    if device == "cuda":
        torch.cuda.synchronize()  # wait for gpu to finish prediction

    timings = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = agent.predict(TEXT, QUESTIONS)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    mean_inference_ms = sum(timings) / repeats * 1000
    answer = result["answers"]["triage"]
    print(str(agent.device).upper())
    print(f"Model load: {load_seconds:.2f}s")
    print(f"Mean inference ({repeats} runs, after warm-up): {mean_inference_ms:.1f}ms")
    pprint.pprint(answer)

    # clean up before returning
    del agent
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return answer


if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

cpu = run("cpu")
gpu = run("cuda")
assert cpu["choice"] == gpu["choice"], "CPU and GPU predictions differ"
