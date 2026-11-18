"""End-to-end smoke test for the notebook.

Runs every code cell of landscape2art_cyclegan.ipynb with tiny synthetic data
(L2A_SMOKE_TEST=1). This covers data loading, the model, training, checkpoint/resume
and inference in about a minute on CPU, with no dataset or GPU needed.

    python tests/smoke_test.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no windows during tests

NOTEBOOK = Path(__file__).resolve().parents[1] / "landscape2art_cyclegan.ipynb"


def run_notebook(work_dir: Path, train: bool = True) -> dict:
    os.environ["L2A_SMOKE_TEST"] = "1"
    os.environ["L2A_WORK_DIR"] = str(work_dir)
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    ns: dict = {"__name__": "__main__"}
    for i, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if not train:
            src =src.replace("TRAIN             = True", "TRAIN             = False")
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            print(f"\n--- code cell {i} failed ---\n{src[:400]}\n")
            raise
    return ns


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        ns = run_notebook(work)

        # Training ran for every epoch and produced the expected artifacts.
        assert [h["epoch"] for h in ns["history"]] == list(range(1, ns["EPOCHS"] + 1)), ns["history"]
        for rel in ["checkpoints/last.pt", "history.csv", "losses.png", "gallery.jpg", "cycle.jpg",
                    f"generator_{ns['ART_STYLE']}.pt", "samples/epoch_001.jpg", "results/test_000.jpg"]:
            assert (work / rel).exists(), f"missing output: {rel}"

        assert ns["n_gray_removed"] == 2, f"grayscale filter removed {ns['n_gray_removed']} photos, expected 2"

        # A second run must resume from last.pt and not train again.
        ns2 = run_notebook(work)
        assert ns2["start_epoch"] == ns["EPOCHS"], "resume from checkpoint failed"
        assert len(ns2["history"]) == ns["EPOCHS"]

        # Evaluate-only mode: loads the checkpoint, skips training, still produces results.
        (work / "gallery.jpg").unlink()
        ns3 = run_notebook(work, train=False)
        assert ns3["start_epoch"] == ns["EPOCHS"] and len(ns3["history"]) == ns["EPOCHS"]
        assert (work / "gallery.jpg").exists(), "evaluate-only run did not produce results"

        # Evaluate-only without any checkpoint must fail loudly, not evaluate random weights.
        with tempfile.TemporaryDirectory() as empty:
            try:
                run_notebook(Path(empty), train=False)
                raise AssertionError("TRAIN=False without a checkpoint should raise")
            except FileNotFoundError:
                pass
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
