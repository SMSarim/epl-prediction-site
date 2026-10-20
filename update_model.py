"""
update_model.py - refresh results, refit the models, and save everything the site needs.

Run from the epl-site folder:   python update_model.py

Reads   data/matches.csv           (your historical data; created/updated by this script)
Writes  data/matches.csv           (history + newly downloaded results)
        epl_model.joblib           (refitted models + current Elo ratings)
        data/fixtures.csv          (upcoming fixtures, if the source is available)
        data/schedule.csv          (the full 38-matchweek schedule, if the source is available)
        data/meta.json             (when the model was last updated, etc.)

If any sanity check fails, nothing is overwritten and the script exits with code 1.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- configuration ----------------
HERE = Path(__file__).parent
DATA = HERE / "data"
MATCHES = DATA / "matches.csv"
MODEL = HERE / "epl_model.joblib"
FIXTURES = DATA / "fixtures.csv"
META = DATA / "meta.json"
SCHEDULE = DATA / "schedule.csv"

K, HOME_ADV, NEW_RATING, POISSON_SEASONS = 15, 60, 1400, 5
GAPFILL_YEARS = 5      # re-download incomplete seasons only if they are this recent

BASE_URL = os.environ.get("EPL_BASE_URL", "https://www.football-data.co.uk/mmz4281/{}/E0.csv")
FIXTURES_URL = os.environ.get("EPL_FIXTURES_URL", "https://www.football-data.co.uk/fixtures.csv")
SCHEDULE_URL = os.environ.get("EPL_SCHEDULE_URL",
                              "https://raw.githubusercontent.com/openfootball/football.json/master/{}/en.1.json")
TODAY = pd.Timestamp(os.environ.get("EPL_TODAY", str(pd.Timestamp.today().date())))

RENAME = {
    "Date": "MatchDate", "FTHG": "FullTimeHomeGoals", "FTAG": "FullTimeAwayGoals", "FTR": "FullTimeResult",
    "HTHG": "HalfTimeHomeGoals", "HTAG": "HalfTimeAwayGoals", "HTR": "HalfTimeResult",
    "HS": "HomeShots", "AS": "AwayShots", "HST": "HomeShotsOnTarget", "AST": "AwayShotsOnTarget",
    "HC": "HomeCorners", "AC": "AwayCorners", "HF": "HomeFouls", "AF": "AwayFouls",
    "HY": "HomeYellowCards", "AY": "AwayYellowCards", "HR": "HomeRedCards", "AR": "AwayRedCards",
}


class CheckFailed(Exception):
    pass


def label(start):                       # 2025 -> "2025/26"
    return f"{start}/{(start + 1) % 100:02d}"


def code(start):                        # 2025 -> "2526"
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def read_csv(src, **kw):
    if str(src).startswith("http"):
        kw["storage_options"] = {"User-Agent": "Mozilla/5.0"}
    return pd.read_csv(src, encoding="utf-8-sig", encoding_errors="replace", **kw)


# ---------------- data ----------------
def fetch_season(start, cols):
    raw = read_csv(BASE_URL.format(code(start))).rename(columns=RENAME)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    raw["MatchDate"] = pd.to_datetime(raw["MatchDate"], dayfirst=True, format="mixed")
    raw = raw.dropna(subset=["FullTimeResult", "FullTimeHomeGoals", "FullTimeAwayGoals"])
    raw = raw.reindex(columns=cols)           # keep only the columns we use (real files have ~100)
    raw["Season"] = label(start)
    raw[["FullTimeHomeGoals", "FullTimeAwayGoals"]] = raw[["FullTimeHomeGoals", "FullTimeAwayGoals"]].astype(int)
    return raw


def load_and_extend():
    old = pd.read_csv(MATCHES, parse_dates=["MatchDate"]).drop(columns=["match_id"], errors="ignore")
    cols = list(old.columns)
    starts = old.Season.str[:4].astype(int)
    last_start = int(starts.max())
    cur_start = TODAY.year if TODAY.month >= 7 else TODAY.year - 1
    counts = old.Season.value_counts()
    incomplete = {int(sn[:4]) for sn, n in counts.items()
                  if n < 380 and int(sn[:4]) >= cur_start - GAPFILL_YEARS}   # old seasons no longer matter
    required = set(range(last_start + 1, cur_start + 1)) | {cur_start}
    to_fetch = sorted(incomplete | set(range(last_start + 1, cur_start + 1)))

    frames, fetched = [], []
    for s in to_fetch:
        try:
            frames.append(fetch_season(s, cols))
            fetched.append(s)
            print(f"  downloaded {label(s)}: {len(frames[-1])} played matches (file had {int(counts.get(label(s), 0))})")
        except Exception as e:
            if s in required and not (s == cur_start and TODAY.month in (7, 8)):
                raise CheckFailed(f"could not download {label(s)}: {type(e).__name__}: {e}")
            print(f"  could not refresh {label(s)} ({type(e).__name__}); keeping what the file has")

    keep = old[~old.Season.isin([label(s) for s in fetched])]
    new = pd.concat([keep] + frames, ignore_index=True)
    new = new.drop_duplicates(["MatchDate", "HomeTeam", "AwayTeam"], keep="last")
    new = new.sort_values("MatchDate", kind="stable").reset_index(drop=True)
    new["match_id"] = new.index
    return old, new, [label(s) for s in fetched]


# ---------------- model ----------------
def add_elo2(df, k=K, home_adv=HOME_ADV, base=1500, new_rating=NEW_RATING):
    seasons = {s: i for i, s in enumerate(df.Season.unique())}
    ratings, last_seen, cur, h_elo, a_elo = {}, {}, None, [], []
    for r in df.itertuples():
        si = seasons[r.Season]
        cur = si
        for t in (r.HomeTeam, r.AwayTeam):
            last = last_seen.get(t)
            if last is None:
                ratings[t] = base if (si == 0 or new_rating is None) else new_rating
            elif last < si - 1 and new_rating is not None:      # absent last season -> reset
                ratings[t] = new_rating
            last_seen[t] = si
        rh, ra = ratings[r.HomeTeam], ratings[r.AwayTeam]
        h_elo.append(rh); a_elo.append(ra)
        exp_h = 1 / (1 + 10 ** ((ra - (rh + home_adv)) / 400))
        score_h = {"H": 1, "D": 0.5, "A": 0}[r.FullTimeResult]
        gd = abs(r.FullTimeHomeGoals - r.FullTimeAwayGoals)
        delta = k * (np.log(gd + 1) + 1) * (score_h - exp_h)
        ratings[r.HomeTeam], ratings[r.AwayTeam] = rh + delta, ra - delta
    out = pd.DataFrame({"match_id": df.match_id.values, "home_elo": h_elo, "away_elo": a_elo})
    return out, {"ratings": ratings, "last_seen": last_seen, "season_idx": cur}


def fit_models(new):
    elo, state = add_elo2(new)
    d = new.merge(elo, on="match_id")
    d["elo_diff"] = d.home_elo - d.away_elo
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(d[["elo_diff"]], d.FullTimeResult)
    recent = d[d.Season.isin(d.Season.unique()[-POISSON_SEASONS:])]
    mk = lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=0, max_iter=1000))
    pois_h = mk().fit(recent[["elo_diff"]], recent.FullTimeHomeGoals)
    pois_a = mk().fit(recent[["elo_diff"]], recent.FullTimeAwayGoals)
    return d, clf, pois_h, pois_a, state


# ---------------- checks ----------------
def run_checks(old, new, fetched, d, clf, state):
    problems = []
    core = ["HomeTeam", "AwayTeam", "FullTimeResult", "FullTimeHomeGoals", "FullTimeAwayGoals", "MatchDate", "Season"]
    if new[core].isna().any().any():
        problems.append("missing values in core columns")
    if len(new) < len(old):
        problems.append(f"row count dropped ({len(old)} -> {len(new)})")
    for s in fetched:
        sub = new[new.Season == s]
        if len(sub) > 380:
            problems.append(f"{s} has {len(sub)} matches (>380)")
        if sub.groupby(["HomeTeam", "AwayTeam"]).size().max() > 1:
            problems.append(f"{s} has a repeated home/away pairing")
        implied = np.where(sub.FullTimeHomeGoals > sub.FullTimeAwayGoals, "H",
                           np.where(sub.FullTimeHomeGoals < sub.FullTimeAwayGoals, "A", "D"))
        if (implied != sub.FullTimeResult.values).any():
            problems.append(f"{s}: result column disagrees with the goals")
    latest = new[new.Season == new.Season.iloc[-1]]
    clubs = set(latest.HomeTeam) | set(latest.AwayTeam)
    if len(latest) >= 30 and len(clubs) != 20:
        problems.append(f"latest season has {len(clubs)} clubs (expected 20) - check team-name spellings")
    tail = d.tail(380)
    ll = log_loss(tail.FullTimeResult, clf.predict_proba(tail[["elo_diff"]]), labels=clf.classes_)
    if not 0.80 <= ll <= 1.20:
        problems.append(f"log loss on the latest 380 matches is {ll:.3f} (expected 0.80-1.20)")
    return problems, ll


# ---------------- fixtures (optional) ----------------
def update_fixtures(state):
    try:
        fx = read_csv(FIXTURES_URL)
        if "Div" in fx.columns:
            fx = fx[fx["Div"] == "E0"]
        fx = fx.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
        fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed")
        fx = fx[fx["Date"] >= TODAY.normalize()]
        keep = [c for c in ["Date", "Time", "HomeTeam", "AwayTeam"] if c in fx.columns]
        fx = fx[keep].sort_values("Date")
        unknown = sorted((set(fx.HomeTeam) | set(fx.AwayTeam)) - set(state["ratings"]))
        if unknown:
            print("  WARNING: fixture team names not in the ratings (spelling?):", unknown)
        fx.to_csv(FIXTURES, index=False)
        print(f"  saved {len(fx)} upcoming fixtures")
    except Exception as e:                                      # non-fatal: the site still works without it
        print(f"  fixtures not updated ({type(e).__name__}: {e})")


# ---------------- full-season schedule (optional) ----------------
TEAM_MAP = {
    "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion FC": "Brighton", "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich", "Leeds United FC": "Leeds", "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves",
    "Leicester City FC": "Leicester", "Norwich City FC": "Norwich", "Luton Town FC": "Luton", "Burnley FC": "Burnley",
    "Sheffield United FC": "Sheffield United", "West Bromwich Albion FC": "West Brom",
}


def fd_name(raw, known):
    """Turn the schedule feed's club name into the spelling used in matches.csv."""
    if raw in TEAM_MAP:
        return TEAM_MAP[raw]
    c = re.sub(r"( AFC| FC)$", "", raw).strip()
    if c in known:
        return c
    for k in known:
        if c.startswith(k) or k.startswith(c):
            return k
    return c


def update_schedule(new):
    try:
        season = new.Season.iloc[-1]
        req = urllib.request.Request(SCHEDULE_URL.format(season.replace("/", "-")), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        cur = new[new.Season == season]
        known = set(cur.HomeTeam) | set(cur.AwayTeam)
        rows = [dict(Season=season, Round=int(re.search(r"\d+", m["round"]).group()), Date=m["date"], Time=m.get("time", ""),
                     HomeTeam=fd_name(m["team1"], known), AwayTeam=fd_name(m["team2"], known)) for m in data["matches"]]
        sc = pd.DataFrame(rows).sort_values(["Round", "Date", "Time"])
        unknown = sorted((set(sc.HomeTeam) | set(sc.AwayTeam)) - known)
        if unknown and len(cur) >= 30:
            print("  WARNING: schedule team names not in matches.csv (spelling?):", unknown)
        sc.to_csv(SCHEDULE, index=False)
        print(f"  saved {len(sc)} scheduled matches ({sc.Round.max()} matchweeks)")
    except Exception as e:                                      # non-fatal: the site falls back to fixtures.csv
        print(f"  schedule not updated ({type(e).__name__}: {e})")


# ---------------- main ----------------
def main():
    DATA.mkdir(exist_ok=True)
    print("Loading and extending results ...")
    old, new, fetched = load_and_extend()
    print("Fitting models ...")
    d, clf, pois_h, pois_a, state = fit_models(new)
    problems, ll = run_checks(old, new, fetched, d, clf, state)
    if problems:
        raise CheckFailed("; ".join(problems))

    tmp_csv, tmp_model = MATCHES.with_suffix(".tmp"), MODEL.with_suffix(".tmp")
    new.drop(columns=["match_id"]).to_csv(tmp_csv, index=False)
    meta = {
        "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "last_match_date": str(new.MatchDate.max().date()),
        "latest_season": new.Season.iloc[-1],
        "n_matches": int(len(new)),
        "recent_logloss_in_sample": round(float(ll), 3),
    }
    joblib.dump({"clf": clf, "pois_home": pois_h, "pois_away": pois_a, "state": state,
                 "config": {"k": K, "home_adv": HOME_ADV, "new_rating": NEW_RATING}, "meta": meta}, tmp_model)
    tmp_csv.replace(MATCHES); tmp_model.replace(MODEL)
    META.write_text(json.dumps(meta, indent=2))

    print("Fixtures ...")
    update_fixtures(state)
    print("Schedule ...")
    update_schedule(new)
    print("\nDone.", json.dumps(meta, indent=2))


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        print(f"\nUPDATE ABORTED, nothing was overwritten: {e}")
        sys.exit(1)
