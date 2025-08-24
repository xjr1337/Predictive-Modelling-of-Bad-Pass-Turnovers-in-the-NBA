# %% [markdown]
# # Interaction Module (Demo)
# 
# **Purpose.** Slice by player/team and inspect **bad-pass** risk curves with the trained model.
# 
# **What this notebook does**
# - Load saved Stage-2 model and feature encoders.
# - Provide quick filters (season, player, team).
# - Display: per-slice risk curves, threshold sensitivity snapshot, and notes on interpretation.
# - No training occurs here; this is a read-only inference/demo.

# %%
# Imports & global config 
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output
from functools import lru_cache

# Paths
OUTPUT_DIR = Path("frozen_interactive"); OUTPUT_DIR.mkdir(exist_ok=True)
STACKED_PATH = "stacked_rf_2023_final.joblib"     # trained Pipeline('prep','clf')
STAGE1_PATH  = "stage1_LGBM_full_2023.joblib"    # 2023 full LGBM

# Raw CSVs
SEASON_PATHS = {
    "2021": "nbastats_2021.csv",
    "2022": "nbastats_2022.csv",
    "2023": "nbastats_2023.csv",
}

# Feature columns 
NUM_COLS_BASE = ["PERIOD","sec_left_period","sec_left_game","bp_last20events",
                 "score_margin","prev_ev1","prev_ev2","prev_ev3"]
CAT_COLS_BASE = ["PLAYER1_TEAM_ID","score_bin","time_bin"]
X_COLS = NUM_COLS_BASE + CAT_COLS_BASE

# Event codes
TURNOVER = 5
BADPASS  = 1

# How many rows to show in the table
TOP_K = 20


# %%
def score2num(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().replace("'", "")
    return 0.0 if x.upper()=="TIE" else pd.to_numeric(x, errors="coerce")

def clock2sec(t):
    if pd.isna(t): return np.nan
    try:
        m, s = map(int, str(t).strip().split(":"))
        return m*60 + s
    except:
        return np.nan

def build_features_turnovers(df_all: pd.DataFrame) -> pd.DataFrame:
    """Build features on the full timeline first, then keep turnovers only."""
    df = df_all.copy()

    # previous event types (computed on ALL events, then filtered)
    for k in (1,2,3):
        df[f"prev_ev{k}"] = (
            df.groupby("GAME_ID")["EVENTMSGTYPE"].shift(k).fillna(0).astype(int)
        )

    # keep turnovers only
    df = df[df["EVENTMSGTYPE"] == TURNOVER].copy()

    # target: bad pass
    if "target" not in df.columns:
        df["target"] = (df["EVENTMSGACTIONTYPE"] == BADPASS).astype(int)

    # time/score features
    df["score_margin"]    = df["SCOREMARGIN"].apply(score2num)
    df["sec_left_period"] = df["PCTIMESTRING"].apply(clock2sec)
    df["sec_left_game"]   = df["sec_left_period"] + np.where(df["PERIOD"]<=4,(4-df["PERIOD"])*720,0)

    # bins
    df["score_bin"] = pd.cut(df["score_margin"], [-40,-15,-5,5,15,40],
                             labels=["<-15","-15~-5","-5~5","5~15",">15"])
    df["time_bin"]  = pd.cut(df["sec_left_period"], [0,60,180,360,720],
                             labels=["0-1min","1-3","3-6","6-12"])

    # order & rolling bad-pass count within turnovers
    df = df.sort_values(["GAME_ID","PERIOD","sec_left_period"], ascending=[True,True,False])
    df["bp_last20events"] = (
        df.groupby("GAME_ID")["target"].rolling(20, min_periods=1).sum()
          .reset_index(level=0, drop=True)
    )
    return df

def to_X_y_groups(df_turn: pd.DataFrame):
    X = df_turn[X_COLS].copy()
    for c in CAT_COLS_BASE:
        X[c] = X[c].astype("category")
    y = df_turn["target"].to_numpy().astype(int)
    groups = df_turn["GAME_ID"].to_numpy()
    return X, y, groups


# %%
USECOLS = None  # keep all columns

@lru_cache(maxsize=4)
def load_season(season: str):
    path = SEASON_PATHS[season]
    df_all = pd.read_csv(path, low_memory=False, usecols=USECOLS)
    df_turn = build_features_turnovers(df_all)

    # prefer name else id for player filtering
    pname = "PLAYER1_NAME" if "PLAYER1_NAME" in df_turn.columns else None
    pid   = "PLAYER1_ID"   if "PLAYER1_ID"   in df_turn.columns else None

    # player options
    if pname:
        players = sorted(df_turn[pname].dropna().astype(str).unique().tolist())
    elif pid:
        players = sorted(df_turn[pid].dropna().astype(str).unique().tolist())
    else:
        players = None
    return df_turn, players, pname, pid

@lru_cache(maxsize=1)
def load_models():
    stage1  = joblib.load(STAGE1_PATH)
    stacked = joblib.load(STACKED_PATH)
    return stage1, stacked

def predict_with_frozen(X: pd.DataFrame) -> np.ndarray:
    stage1, stacked = load_models()
    prior = stage1.predict_proba(X[X_COLS])[:, 1]
    X_stk = X.copy(); X_stk["oof_pred"] = prior
    return stacked.predict_proba(X_stk)[:, 1]

def plot_curves(y, p, prefix: Path):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title("ROC"); plt.tight_layout()
    plt.savefig(prefix.with_suffix(".roc.png"), dpi=160); plt.close()

    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(); plt.plot(rec, prec); plt.hlines(y.mean(), 0, 1, linestyles="--")
    plt.title("PR"); plt.tight_layout()
    plt.savefig(prefix.with_suffix(".pr.png"), dpi=160); plt.close()


# %%

import pandas as pd
import ipywidgets as W
from IPython.display import display, clear_output

#  NBA team ID mapping (full name & abbreviation) 
TEAM_ID_TO_NAME = {
    1610612737: "Atlanta Hawks",           1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",     1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",           1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",          1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",         1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",      1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",         1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",           1610612752: "New York Knicks",
    1610612753: "Orlando Magic",           1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",      1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",  1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",       1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",         1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",       1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",         1610612766: "Charlotte Hornets",
}
TEAM_ID_TO_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHA",
}

def _to_int64(series):
    """Robustly convert TEAM_ID-like series to Int64 (handles floats/scientific notation)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")

dd_season = W.Dropdown(options=list(SEASON_PATHS.keys()), value="2021", description="Season")
rb_scope  = W.RadioButtons(options=["By player","All season"], value="By player", description="Scope")
dd_player = W.Dropdown(options=["<All>"], value="<All>", description="Player")
btn_run   = W.Button(description="Run", button_style="primary")
out       = W.Output()

def refresh_options(change=None):
    season = dd_season.value
    df_turn, players, pname, pid = load_season(season)
    dd_player.options = ["<All>"] + (players or [])

dd_season.observe(refresh_options, names="value")
refresh_options()

def on_run_clicked(b):
    with out:
        clear_output()

        # 1) load data by season
        season = dd_season.value
        df_turn, players, pname, pid = load_season(season)

        # 2) filter by player if chosen
        df_sub = df_turn.copy()
        tag = f"{season}_all"
        if rb_scope.value == "By player" and dd_player.value != "<All>":
            if pname and pname in df_sub.columns:
                df_sub = df_sub[df_sub[pname].astype(str) == str(dd_player.value)]
            elif pid and pid in df_sub.columns:
                df_sub = df_sub[df_sub[pid].astype(str) == str(dd_player.value)]
            tag = f"{season}_player_{dd_player.value}"

        if len(df_sub) == 0:
            print("No rows after filtering. Please change the selection.")
            return

        # 3) predict
        X, y, _ = to_X_y_groups(df_sub)
        try:
            y_hat = predict_with_frozen(X)
        except Exception as e:
            import traceback as tb
            print("Prediction failed:", e)
            tb.print_exc()
            return

        # 4) metrics
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        auc   = roc_auc_score(y, y_hat) if len(np.unique(y)) > 1 else np.nan
        ap    = average_precision_score(y, y_hat) if len(np.unique(y)) > 1 else np.nan
        brier = brier_score_loss(y, y_hat)
        print(f"[{tag}] n={len(y)} | AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")

        # 5) save & curves (drop SCOREMARGIN)
        prefix = OUTPUT_DIR / f"interactive_{tag}"
        out_df = df_sub.copy()
        out_df["pred"] = y_hat
        out_df = out_df.drop(columns=["SCOREMARGIN"], errors="ignore")
        out_df.to_csv(prefix.with_suffix(".pred.csv"), index=False)
        plot_curves(y, y_hat, prefix)

        # 6) friendly display/CSV (rename, map team, pass type, rounding)
        rename_map = {
            "GAME_ID": "Game ID",
            "PERIOD": "Period",
            "PCTIMESTRING": "Time left in period",
            "target": "Target (0/1)",
            "pred": "Probability of bad pass",
        }
        friendly = out_df.rename(columns={k: v for k, v in rename_map.items() if k in out_df.columns})

        # Team mapping from PLAYER1_TEAM_ID
        if "PLAYER1_TEAM_ID" in out_df.columns:
            tid = _to_int64(out_df["PLAYER1_TEAM_ID"])
            friendly["Team"] = tid.map(TEAM_ID_TO_NAME).fillna(tid.astype("string"))
            friendly["Team abbr"] = tid.map(TEAM_ID_TO_ABBR)

        # Human-readable pass type
        if "Target (0/1)" in friendly.columns:
            friendly["Pass type"] = friendly["Target (0/1)"].map({1: "Bad pass turnover", 0: "Other turnover"})

        # Round probability
        if "Probability of bad pass" in friendly.columns:
            friendly["Probability of bad pass"] = friendly["Probability of bad pass"].astype(float).round(3)

        # Choose columns to show 
        show_cols = [c for c in [
            "Game ID", "Period", "Time left in period",
            "Team",  # or "Team abbr"
            "Pass type", "Probability of bad pass"
        ] if c in friendly.columns]

        # Save friendly CSV and display top-K
        friendly.to_csv(prefix.with_suffix(".display.csv"), index=False)
        key = "Probability of bad pass" if "Probability of bad pass" in friendly.columns else "pred"
        display(friendly[show_cols].sort_values(key, ascending=False).head(TOP_K))

# Bind the callback and render
btn_run.on_click(on_run_clicked)
display(W.HBox([dd_season, rb_scope, dd_player, btn_run]))
display(out)



