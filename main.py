"""Entry point.
  python main.py list
  python main.py run <id or prefix> ...   | python main.py run --all
  python main.py submit <id>              -> outputs/submission_<id>.csv
"""
import sys
import pandas as pd
from src.registry import EXPERIMENTS
from src.experiment import run, PRED, data
from src.data import ROOT


def main(a):
    cmd = a[0] if a else "list"
    if cmd == "list":
        for e in EXPERIMENTS:
            print(f"{e['id']:28s} {e['hypothesis']}")
    elif cmd == "run":
        for e in EXPERIMENTS:
            if "--all" in a or any(e["id"].startswith(x) for x in a[1:]):
                run(e)
    elif cmd == "submit":
        sub = pd.read_csv(PRED / f"test_{a[1]}.csv")
        te = data()[1]
        assert list(sub.ID) == list(te.ID) and sub.TARGET.notna().all() and list(sub.columns) == ["ID", "TARGET"]
        out = ROOT / "outputs" / f"submission_{a[1]}.csv"
        sub.to_csv(out, index=False)
        print("written", out, sub.shape)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
