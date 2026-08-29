"""Fill the `result` column of data/mlb/narrative/log.csv from games.csv for
any logged game that has since gone final. Idempotent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.compile import DATA  # noqa: E402

log_path = DATA / "narrative" / "log.csv"
if not log_path.exists():
    print("no log yet")
    sys.exit(0)
log = pd.read_csv(log_path)
games = pd.read_csv(DATA / "games.csv", usecols=["game_pk", "played", "result"])
res = games[games["played"]].set_index("game_pk")["result"]
need = log["result"].isna() & log["game_pk"].isin(res.index)
log.loc[need, "result"] = log.loc[need, "game_pk"].map(res)
log.to_csv(log_path, index=False)
print(f"settled {int(need.sum())} rows; {int(log['result'].isna().sum())} still open")
