import base64
import html as _html
import zlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

HERE = Path(__file__).parent
MODEL, MATCHES, FIXTURES = HERE / "epl_model.joblib", HERE / "data" / "matches.csv", HERE / "data" / "fixtures.csv"
SCHEDULE = HERE / "data" / "schedule.csv"      # full 38-matchweek schedule, written by update_model.py
BADGE_DIR = HERE / "assets" / "badges"          # optional: "<Team name>.png/.svg/.webp" here replaces the generated badge
MATCH_MINUTES = 110                              # after kick-off + this long, a game with no result is "awaiting result"
esc = _html.escape

st.set_page_config(page_title="EPL Predictor", page_icon="⚽", layout="wide")

# =====================================================================
# Styling: two-colour palette (green #1F8A5B, cream #FFFDD0), system fonts for a fast first paint
# =====================================================================
CSS = """
<style>
:root{--green:#1F8A5B;--cream:#FFFDD0;--deep:#0E3B27;--ink:#0F2A1D;--muted:#52705F;--line:rgba(31,138,91,.22);
--card:rgba(255,255,255,.72);--tan:#D6CF94;--font:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif}
.stApp{background:var(--cream);font-family:var(--font);color:var(--ink)}
.block-container{max-width:1320px;padding:5rem 2.5rem 3rem}
@media (max-width:760px){.block-container{padding:4.5rem 1rem 2rem}}
header[data-testid="stHeader"]{background:transparent}
.stAppDeployButton,[data-testid="stDecoration"]{display:none}
[data-baseweb="select"]>div{background:#fff}
#MainMenu,footer{visibility:hidden}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{padding:10px 18px;color:var(--muted)}
.stTabs [data-baseweb="tab"] p{font-size:1.02rem;font-weight:600}
.stTabs [aria-selected="true"]{color:var(--green)}
[data-baseweb="tab-highlight"]{background:var(--green)!important;height:3px}
.stButton>button{border-radius:10px;border:1px solid var(--line);background:var(--card);font-weight:600;transition:border-color .15s,color .15s}
.stButton>button:hover:not(:disabled){border-color:var(--green);color:var(--green)}

.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px}
.top h1{font-size:2rem;font-weight:800;margin:0;padding:0;letter-spacing:-.02em;color:var(--green)}
.pill{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:.82rem;color:var(--muted)}

.badge{display:inline-flex;align-items:center;justify-content:center;border-radius:50%;font-weight:800;letter-spacing:.02em;flex:none;
box-shadow:inset 0 0 0 2px rgba(255,255,255,.28),0 1px 3px rgba(0,0,0,.2)}
img.badge{object-fit:contain;box-shadow:none;border-radius:0}

.mw-head{margin:14px 0 4px;font-size:1.5rem;font-weight:800}
.mw-sub{color:var(--muted);font-size:.9rem}
.d-head{font-weight:700;color:var(--muted);margin:18px 0 8px;font-size:.95rem}
.m-card{display:grid;grid-template-columns:1fr 110px 1fr 280px;grid-template-areas:"home mid away prob";align-items:center;gap:12px;
background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 16px;margin-bottom:8px;
animation:rise .4s ease both;animation-delay:calc(var(--i)*35ms)}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.m-home{grid-area:home;display:flex;align-items:center;justify-content:flex-end;gap:10px;font-weight:600;text-align:right}
.m-away{grid-area:away;display:flex;align-items:center;gap:10px;font-weight:600}
.m-mid{grid-area:mid;text-align:center}.m-prob{grid-area:prob}
.lose{color:var(--muted);font-weight:500}
.score{font-size:1.4rem;font-weight:800}.ko{font-size:1.15rem;font-weight:800;color:var(--green)}
.note{font-size:.72rem;color:var(--muted);line-height:1.2;display:block}

.pbar{display:flex;height:8px;border-radius:6px;overflow:hidden;background:rgba(31,138,91,.1);animation:reveal .8s ease both}
.pbar span{height:100%}.pbar .h{background:var(--green)}.pbar .d{background:var(--tan)}.pbar .a{background:var(--deep)}
.plab{display:flex;justify-content:space-between;font-size:.76rem;font-weight:600;margin-top:4px;color:var(--muted)}
@keyframes reveal{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}

.hero{background:var(--green);color:var(--cream);border-radius:20px;padding:26px 28px 22px;margin:8px 0 14px;animation:rise .45s ease both}
.hero-teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:14px}
.ht{display:flex;flex-direction:column;align-items:center;text-align:center;gap:6px;font-size:1.3rem;font-weight:800}
.hmid{text-align:center;font-size:.85rem;opacity:.9}.hmid b{display:block;font-size:1.3rem}
.hero-nums{display:grid;grid-template-columns:1fr 1fr 1fr;text-align:center;margin:20px 0 10px}
.hero-nums b{display:block;font-size:2.5rem;line-height:1;font-weight:800}.hero-nums span{font-size:.85rem;opacity:.9}
.hero .pbar{height:12px;border-radius:8px;background:rgba(255,253,208,.15)}
.hero .pbar .h{background:var(--cream)}.hero .pbar .d{background:rgba(255,253,208,.4)}.hero .pbar .a{background:var(--deep)}
.verdict{margin-top:14px;font-size:.95rem}.scn{background:rgba(255,253,208,.18);border-radius:999px;padding:2px 10px;font-size:.78rem;margin-left:8px}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.tile .v{font-size:1.5rem;font-weight:800;color:var(--green)}.tile .l{color:var(--muted);font-size:.82rem}
.heat{border-collapse:separate;border-spacing:3px;width:100%;text-align:center;font-size:.8rem}
.heat th{color:var(--muted);font-weight:600;padding:4px}.heat td{border-radius:7px;padding:9px 2px}
.scroll{overflow-x:auto}

.f{display:inline-block;width:19px;height:19px;border-radius:50%;font-size:.65rem;font-style:normal;font-weight:700;line-height:19px;text-align:center;color:#fff;margin-right:3px}
.f.w{background:var(--green)}.f.d{background:#9BA89F}.f.l{background:#C0392B}
.tbl{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);font-size:.9rem}
.tbl th{color:var(--muted);font-weight:600;text-align:center;padding:10px 6px;border-bottom:1px solid var(--line)}
.tbl td{text-align:center;padding:8px 6px;border-bottom:1px solid rgba(31,138,91,.1);font-variant-numeric:tabular-nums}
.tbl td.cl,.tbl th.cl{text-align:left}.tbl td.cl{font-weight:600}.tbl td.pts{font-weight:800;color:var(--green)}
.tbl td.z{width:4px;padding:0}.z.ucl{background:var(--green)}.z.rel{background:#C0392B}
.clb{display:flex;align-items:center;gap:10px}
.key{display:flex;gap:18px;font-size:.8rem;color:var(--muted);margin-top:8px}
.key i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.sec{font-size:1.15rem;font-weight:800;margin:24px 0 8px}
.lb{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.lb-t{font-weight:800;margin-bottom:4px}
.lb-r{display:flex;align-items:center;gap:10px;padding:6px 0;border-top:1px solid rgba(31,138,91,.1);font-size:.9rem}
.lb-r:first-of-type{border-top:0}.lb-r .nm{flex:1}.lb-r b{color:var(--green)}

.about{max-width:720px;line-height:1.7}.about h3{margin:22px 0 4px}
.author{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 18px;margin:16px 0}
.author .badge{width:46px;height:46px;background:var(--green);color:var(--cream);font-size:.95rem}
.empty{border:1px dashed var(--line);border-radius:14px;padding:26px;text-align:center;color:var(--muted)}

@media (max-width:760px){
.m-card{grid-template-columns:1fr 70px 1fr;grid-template-areas:"home mid away" "prob prob prob";gap:8px 10px}
.hero{padding:18px 12px}.hero-nums b{font-size:1.8rem}.ht{font-size:1.05rem}.tbl{font-size:.8rem}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""


def H(s):
    """Render HTML. Lines are stripped and joined so Markdown never mistakes indentation for a code block."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


H(CSS)


# =====================================================================
# Data (cached; each cache refreshes only when its file changes)
# =====================================================================
def stamp(p):
    return p.stat().st_mtime if p.exists() else 0


@st.cache_resource
def load_model(stamp):
    return joblib.load(MODEL)


@st.cache_data
def load_matches(stamp):
    m = pd.read_csv(MATCHES, parse_dates=["MatchDate"]).sort_values("MatchDate", kind="stable").reset_index(drop=True)
    return m[m.Season == m.Season.iloc[-1]].reset_index(drop=True)          # the app only needs the current season


@st.cache_data
def load_fixtures(stamp):
    if not FIXTURES.exists():
        return pd.DataFrame(columns=["Date", "Time", "HomeTeam", "AwayTeam"])
    return pd.read_csv(FIXTURES, parse_dates=["Date"])


@st.cache_data
def load_schedule(stamp):
    if not SCHEDULE.exists():
        return pd.DataFrame(columns=["Season", "Round", "Date", "Time", "HomeTeam", "AwayTeam"])
    return pd.read_csv(SCHEDULE, parse_dates=["Date"], keep_default_na=False, dtype={"Time": str})


MS, FS, XS, SS = stamp(MATCHES), stamp(FIXTURES), stamp(MODEL), stamp(SCHEDULE)
M = load_model(XS)
clf, pois_h, pois_a, state, cfg = M["clf"], M["pois_home"], M["pois_away"], M["state"], M["config"]
season_df = load_matches(MS)
season = season_df.Season.iloc[-1]


def build_inferred(ms, fs):
    """Fallback when there is no schedule file: results plus the fixtures file, with matchweek = games the busier side has played."""
    rows, n = [], {}
    for r in load_matches(ms).itertuples():
        n[r.HomeTeam] = n.get(r.HomeTeam, 0) + 1
        n[r.AwayTeam] = n.get(r.AwayTeam, 0) + 1
        rows.append(dict(mw=min(max(n[r.HomeTeam], n[r.AwayTeam]), 38), date=r.MatchDate, time="", home=r.HomeTeam,
                         away=r.AwayTeam, hg=int(r.FullTimeHomeGoals), ag=int(r.FullTimeAwayGoals), played=True))
    base = {} if n and max(n.values()) >= 38 else dict(n)
    done = {(x["home"], x["away"], x["date"]) for x in rows}
    for f in load_fixtures(fs).sort_values("Date", kind="stable").itertuples():
        if (f.HomeTeam, f.AwayTeam, f.Date) in done:
            continue
        mw = max(base.get(f.HomeTeam, 0), base.get(f.AwayTeam, 0)) + 1
        base[f.HomeTeam] = base[f.AwayTeam] = mw
        t = getattr(f, "Time", "")
        rows.append(dict(mw=min(mw, 38), date=f.Date, time=t if isinstance(t, str) else "", home=f.HomeTeam,
                         away=f.AwayTeam, hg=None, ag=None, played=False))
    return rows


@st.cache_data
def build_standings(ms):
    df = load_matches(ms)
    h = df.assign(team=df.HomeTeam, gf=df.FullTimeHomeGoals, ga=df.FullTimeAwayGoals)
    a = df.assign(team=df.AwayTeam, gf=df.FullTimeAwayGoals, ga=df.FullTimeHomeGoals)
    b = pd.concat([h, a]).sort_values("MatchDate", kind="stable")
    b["pts"] = np.select([b.gf > b.ga, b.gf == b.ga], [3, 1], 0)
    b["res"] = np.select([b.gf > b.ga, b.gf == b.ga], ["W", "D"], "L")
    b["cs"] = (b.ga == 0).astype(int)
    b["w"], b["d"], b["l"] = (b.res == "W").astype(int), (b.res == "D").astype(int), (b.res == "L").astype(int)
    t = b.groupby("team").agg(P=("pts", "size"), W=("w", "sum"), D=("d", "sum"), L=("l", "sum"), GF=("gf", "sum"),
                              GA=("ga", "sum"), CS=("cs", "sum"), Pts=("pts", "sum"), Form=("res", lambda s: "".join(s.tail(5))))
    t["GD"] = t.GF - t.GA
    return t.sort_values(["Pts", "GD", "GF"], ascending=False)


@st.cache_data
def build_schedule(ms, fs, ss):
    """All 38 matchweeks. Each scheduled game takes its result from matches.csv if it is there (a home/away pairing happens once
    a season), otherwise it is upcoming; kick-off dates and times from fixtures.csv win when they are close to the schedule's."""
    sched = load_schedule(ss)
    if sched.empty or sched.Season.iloc[0] != load_matches(ms).Season.iloc[-1]:
        return build_inferred(ms, fs)
    res = {(r.HomeTeam, r.AwayTeam): r for r in load_matches(ms).itertuples()}
    fx = {(f.HomeTeam, f.AwayTeam): f for f in load_fixtures(fs).itertuples()}
    rows, seen = [], set()
    for s_ in sched.itertuples():
        key, r = (s_.HomeTeam, s_.AwayTeam), res.get((s_.HomeTeam, s_.AwayTeam))
        if r is not None:
            seen.add(key)
            rows.append(dict(mw=int(s_.Round), date=r.MatchDate, time="", home=key[0], away=key[1],
                             hg=int(r.FullTimeHomeGoals), ag=int(r.FullTimeAwayGoals), played=True))
            continue
        date, time = s_.Date, s_.Time or ""
        f = fx.get(key)
        if f is not None and abs((f.Date - date).days) <= 14:
            t = getattr(f, "Time", "")
            date, time = f.Date, (t if isinstance(t, str) and t else time)
        rows.append(dict(mw=int(s_.Round), date=date, time=time, home=key[0], away=key[1], hg=None, ag=None, played=False))
    if any(k not in seen for k in res):                 # a club name did not match the schedule, so do not hide games
        return build_inferred(ms, fs)
    return rows


schedule = build_schedule(MS, FS, SS)
upcoming = [x for x in schedule if not x["played"]]
max_mw = max((x["mw"] for x in schedule), default=1)


def now_uk():
    try:
        from zoneinfo import ZoneInfo
        return pd.Timestamp.now(tz=ZoneInfo("Europe/London")).tz_localize(None)
    except Exception:                               # no timezone database: UTC is within an hour of UK time
        return pd.Timestamp.now("UTC").tz_localize(None)


@lru_cache(maxsize=None)
def tz_for(name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return None


def viewer_tz():
    """The visitor's own time zone (from the browser), else this computer's. Returns (tzinfo or None, label)."""
    name = None
    try:
        name = st.context.timezone
    except Exception:
        pass
    tz = tz_for(name) if name else None
    if tz is None:
        tz = pd.Timestamp.now().astimezone().tzinfo
        name = pd.Timestamp.now().astimezone().tzname()
    return tz, name


def clock(ts):
    return ts.strftime("%I:%M %p").lstrip("0")          # 12-hour clock, e.g. 8:00 PM


def to_local(x, tz):
    """(kick-off in the viewer's time, text to show). Schedule times are UK time."""
    t = x["time"]
    if not t:
        return x["date"], ("" if x["played"] else "TBC")
    ko, uk = kickoff(x), tz_for("Europe/London")
    if tz is None or uk is None:
        return ko, clock(ko)
    try:
        lk = ko.tz_localize(uk).tz_convert(tz).tz_localize(None)
        return lk, clock(lk)
    except Exception:
        return ko, clock(ko)


def kickoff(x):
    try:
        return pd.Timestamp(f'{x["date"]:%Y-%m-%d} {x["time"]}')
    except Exception:                               # no time listed: assume midday
        return x["date"] + pd.Timedelta(hours=12)


def status(x, now):
    """played / upcoming / live (kicked off, no result yet) / pending (should be over, result not in the data yet)."""
    if x["played"]:
        return "played"
    ko = kickoff(x)
    if now < ko:
        return "upcoming"
    return "live" if now < ko + pd.Timedelta(minutes=MATCH_MINUTES) else "pending"


# the default matchweek is the earliest one with a game still to come, else the last one with a result
_now = now_uk()
_still = [x for x in upcoming if status(x, _now) in ("upcoming", "live")]
default_mw = min(x["mw"] for x in _still) if _still else max((x["mw"] for x in schedule if x["played"]), default=max_mw)

# =====================================================================
# Badges
# =====================================================================
TEAMS = {
    "Arsenal": ("ARS", "#EF0107", "#FFFFFF"), "Aston Villa": ("AVL", "#670E36", "#95BFE5"),
    "Bournemouth": ("BOU", "#DA291C", "#000000"), "Brentford": ("BRE", "#E30613", "#FFFFFF"),
    "Brighton": ("BHA", "#0057B8", "#FFFFFF"), "Burnley": ("BUR", "#6C1D45", "#99D6EA"),
    "Chelsea": ("CHE", "#034694", "#FFFFFF"), "Coventry": ("COV", "#59CBE8", "#0A2240"),
    "Crystal Palace": ("CRY", "#1B458F", "#C4122E"), "Everton": ("EVE", "#003399", "#FFFFFF"),
    "Fulham": ("FUL", "#FFFFFF", "#000000"), "Leeds": ("LEE", "#FFCD00", "#1D428A"),
    "Leicester": ("LEI", "#003090", "#FDBE11"), "Liverpool": ("LIV", "#C8102E", "#FFFFFF"),
    "Man City": ("MCI", "#6CABDD", "#1C2C5B"), "Man United": ("MUN", "#DA291C", "#FBE122"),
    "Newcastle": ("NEW", "#241F20", "#FFFFFF"), "Nott'm Forest": ("NFO", "#DD0000", "#FFFFFF"),
    "Sheffield United": ("SHU", "#EE2737", "#FFFFFF"), "Southampton": ("SOU", "#D71920", "#FFFFFF"),
    "Sunderland": ("SUN", "#EB172B", "#FFFFFF"), "Tottenham": ("TOT", "#132257", "#FFFFFF"),
    "West Ham": ("WHU", "#7A263A", "#1BB1E7"), "Wolves": ("WOL", "#FDB913", "#231F20"),
    "Ipswich": ("IPS", "#0044AA", "#FFFFFF"), "Luton": ("LUT", "#F78F1E", "#002D62"),
    "Norwich": ("NOR", "#00A650", "#FFF200"), "Watford": ("WAT", "#FBEE23", "#ED2127"),
    "West Brom": ("WBA", "#122F67", "#FFFFFF"), "Hull": ("HUL", "#F5A12D", "#000000"),
}


@lru_cache(maxsize=None)
def badge_img(team):
    for ext, mime in (("png", "image/png"), ("svg", "image/svg+xml"), ("webp", "image/webp")):
        p = BADGE_DIR / f"{team}.{ext}"
        if p.exists():
            return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    return None


@lru_cache(maxsize=None)
def badge(team, size=40):
    img = badge_img(team)
    if img:
        return f'<img class="badge" alt="{esc(team)}" src="{img}" style="width:{size}px;height:{size}px">'
    if team in TEAMS:
        code, bg, fg = TEAMS[team]
    else:
        code, bg, fg = team[:3].upper(), f"hsl({zlib.crc32(team.encode()) % 360} 50% 36%)", "#FFFFFF"
    edge = "box-shadow:inset 0 0 0 2px rgba(0,0,0,.14);" if bg.upper() == "#FFFFFF" else ""
    return (f'<span class="badge" title="{esc(team)}" style="width:{size}px;height:{size}px;background:{bg};color:{fg};'
            f'font-size:{size * 0.32:.0f}px;{edge}">{code}</span>')


# =====================================================================
# Model
# =====================================================================
def score_grid(mu_h, mu_a, max_goals=10):
    g = np.arange(max_goals + 1)
    grid = poisson.pmf(g, mu_h)[:, None] * poisson.pmf(g, mu_a)[None, :]
    return grid / grid.sum()


def get_rating(team):
    if team not in state["ratings"] or state["last_seen"][team] < state["season_idx"] - 1:
        return cfg["new_rating"]
    return state["ratings"][team]


def known(team):
    return team in state["ratings"]


@st.cache_data(max_entries=2000)
def predict(home, away, adj_h=0, adj_a=0, _xs=XS):
    rh, ra = get_rating(home) + adj_h, get_rating(away) + adj_a
    X = pd.DataFrame({"elo_diff": [rh - ra]})
    p = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    mu_h, mu_a = float(pois_h.predict(X)[0]), float(pois_a.predict(X)[0])
    g = score_grid(mu_h, mu_a)
    total = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    i, j = np.unravel_index(np.argmax(g), g.shape)
    return {"elo": (rh, ra), "home": float(p["H"]), "draw": float(p["D"]), "away": float(p["A"]), "xg": (mu_h, mu_a),
            "grid": g[:6, :6], "over25": float(g[total > 2].sum()), "btts": float(g[1:, 1:].sum()),
            "top": (f"{i}–{j}", float(g[i, j]))}


def pbar(ph, pdr, pa, labels=True):
    bar = (f'<div class="pbar"><span class="h" style="width:{ph * 100:.1f}%"></span>'
           f'<span class="d" style="width:{pdr * 100:.1f}%"></span><span class="a" style="width:{pa * 100:.1f}%"></span></div>')
    if labels:
        bar += f'<div class="plab"><span>{ph:.0%}</span><span>{pdr:.0%}</span><span>{pa:.0%}</span></div>'
    return bar


def dots(form):
    return "".join(f'<i class="f {c.lower()}">{c}</i>' for c in form)


# =====================================================================
# Header + tabs
# =====================================================================
meta = M.get("meta", {})
through = pd.to_datetime(meta.get("last_match_date", season_df.MatchDate.max())).strftime("%d %b %Y")
H(f'<div class="top"><h1>Premier League Predictor</h1><span class="pill">Results through {through}</span></div>')
tab_matches, tab_predict, tab_table, tab_about = st.tabs(["Matches", "Predictions", "Table & stats", "About"])


# ---------------- Tab 1: Matches ----------------
@st.fragment
def matches_view():
    st.session_state.setdefault("mw", default_mw)
    st.session_state.mw = min(max(st.session_state.mw, 1), max_mw)

    def step(d):
        st.session_state.mw = min(max(st.session_state.mw + d, 1), max_mw)

    def to_current():
        st.session_state.mw = default_mw

    c1, c2, c3, c4 = st.columns([1, 2, 1, 1.4], vertical_alignment="bottom")
    c1.button("‹ Previous", on_click=step, args=(-1,), disabled=st.session_state.mw <= 1, width="stretch")
    c2.selectbox("Matchweek", range(1, max_mw + 1), key="mw", format_func=lambda x: f"Matchweek {x}")
    c3.button("Next ›", on_click=step, args=(1,), disabled=st.session_state.mw >= max_mw, width="stretch")
    c4.button("Current week", on_click=to_current, disabled=st.session_state.mw == default_mw, width="stretch")

    mw, now = st.session_state.mw, now_uk()
    tz, tzname = viewer_tz()
    week = []
    for x in schedule:
        if x["mw"] == mw:
            ld, lt = to_local(x, tz)
            week.append(dict(x, ld=ld, lt=lt))
    week.sort(key=lambda x: (x["ld"], x["home"]))
    if not week:
        H('<div class="empty">No matches listed for this matchweek yet.</div>')
        return
    d0, d1 = week[0]["ld"].normalize(), week[-1]["ld"].normalize()
    span = f"{d0:%a %d %b}" + (f" – {d1:%a %d %b}" if d1 != d0 else "")
    H(f'<div class="mw-head">Matchweek {mw}</div><div class="mw-sub">{span} · kick-off times in your local time ({esc(str(tzname))})</div>')

    out, last, waiting = [], None, False
    for i, x in enumerate(week):
        day = x["ld"].normalize()
        if day != last:
            out.append(f'<div class="d-head">{day:%A %d %B}</div>')
            last = day
        s, hl, al, prob = status(x, now), "", "", ""
        if s == "played":
            mid = f'<span class="score">{x["hg"]}–{x["ag"]}</span>'
            hl, al = ("", "lose") if x["hg"] > x["ag"] else ("lose", "") if x["hg"] < x["ag"] else ("", "")
        elif s == "upcoming":
            mid = f'<span class="ko">{esc(x["lt"]) or "TBC"}</span>'
            if known(x["home"]) and known(x["away"]):
                p = predict(x["home"], x["away"])
                prob = pbar(p["home"], p["draw"], p["away"])
        else:
            waiting = waiting or s == "pending"
            mid = f'<span class="note">{"Kicked off" if s == "live" else "Awaiting result"}</span>'
        out.append(f'<div class="m-card" style="--i:{min(i, 12)}">'
                   f'<div class="m-home {hl}"><span>{esc(x["home"])}</span>{badge(x["home"], 34)}</div>'
                   f'<div class="m-mid">{mid}</div>'
                   f'<div class="m-away {al}">{badge(x["away"], 34)}<span>{esc(x["away"])}</span></div>'
                   f'<div class="m-prob">{prob}</div></div>')
    H("".join(out))
    if waiting:
        st.caption("Finished games show a score once the results data is updated (run update_model.py).")


with tab_matches:
    matches_view()


# ---------------- Tab 2: Predictions ----------------
@st.fragment
def predict_view():
    clubs = sorted(t for t, s in state["last_seen"].items() if s == state["season_idx"])
    if st.session_state.get("home") not in clubs or st.session_state.get("away") not in clubs:
        nxt = next((x for x in _still if x["home"] in clubs and x["away"] in clubs), None)
        st.session_state.home, st.session_state.away = (nxt["home"], nxt["away"]) if nxt else (clubs[0], clubs[1])
    st.session_state.setdefault("adj_h", 0)
    st.session_state.setdefault("adj_a", 0)

    def swap():
        s = st.session_state
        s.home, s.away, s.adj_h, s.adj_a = s.away, s.home, s.adj_a, s.adj_h

    def reset():
        st.session_state.adj_h = st.session_state.adj_a = 0

    c1, c2, c3 = st.columns([5, 1, 5], vertical_alignment="bottom")
    c1.selectbox("Home team", clubs, key="home")
    c2.button("⇄", on_click=swap, help="Swap home and away", width="stretch")
    c3.selectbox("Away team", clubs, key="away")
    home, away = st.session_state.home, st.session_state.away
    if home == away:
        st.warning("Pick two different teams.")
        return

    with st.expander("What-if: change team strength"):
        st.caption("Lower a team's Elo to test a key injury, raise it for strong form. About 20 points is a small change.")
        e1, e2, e3 = st.columns([5, 5, 2], vertical_alignment="bottom")
        e1.slider(f"{home}", -100, 100, step=5, key="adj_h")
        e2.slider(f"{away}", -100, 100, step=5, key="adj_a")
        e3.button("Reset", on_click=reset, width="stretch")

    ah, aa = st.session_state.adj_h, st.session_state.adj_a
    r = predict(home, away, ah, aa)
    best = max(r["home"], r["draw"], r["away"])
    if r["draw"] == best:
        verdict = "A draw is the most likely result."
    else:
        fav = home if r["home"] == best else away
        verdict = f"{'Model favours' if best >= .5 else 'Slight edge to'} {esc(fav)} ({best:.0%})."
    scn = f'<span class="scn">{esc(home)} {ah:+d}, {esc(away)} {aa:+d} Elo</span>' if (ah or aa) else ""
    H(f"""
    <div class="hero"><div class="hero-teams">
    <div class="ht">{badge(home, 72)}{esc(home)}</div>
    <div class="hmid">expected goals<b>{r['xg'][0]:.1f} – {r['xg'][1]:.1f}</b></div>
    <div class="ht">{badge(away, 72)}{esc(away)}</div></div>
    <div class="hero-nums"><div><b>{r['home']:.0%}</b><span>Home win</span></div>
    <div><b>{r['draw']:.0%}</b><span>Draw</span></div><div><b>{r['away']:.0%}</b><span>Away win</span></div></div>
    {pbar(r['home'], r['draw'], r['away'], labels=False)}
    <div class="verdict">{verdict}{scn}</div></div>
    """)
    H(f'<div class="tiles"><div class="tile"><div class="v">{r["top"][0]}</div><div class="l">Most likely score ({r["top"][1]:.0%})</div></div>'
      f'<div class="tile"><div class="v">{r["over25"]:.0%}</div><div class="l">Over 2.5 goals</div></div>'
      f'<div class="tile"><div class="v">{r["btts"]:.0%}</div><div class="l">Both teams score</div></div></div>')

    with st.expander("Full scoreline grid"):
        g = r["grid"]
        mx, body = g.max(), ""
        for i in range(6):
            cells = "".join(f'<td style="background:rgba(31,138,91,{0.07 + 0.93 * g[i, j] / mx:.2f});'
                            f'color:{"#FFFDD0" if g[i, j] / mx > .55 else "#0F2A1D"}">{g[i, j]:.1%}</td>' for j in range(6))
            body += f"<tr><th>{i}</th>{cells}</tr>"
        H(f'<div class="scroll"><table class="heat"><tr><th></th>{"".join(f"<th>{j}</th>" for j in range(6))}</tr>{body}</table></div>')
        st.caption(f"Rows: {home} goals. Columns: {away} goals.")


with tab_predict:
    predict_view()

# ---------------- Tab 3: Table & stats ----------------
with tab_table:
    tbl = build_standings(MS)
    n_teams = len(tbl)
    rows = []
    for pos, (team, r) in enumerate(tbl.iterrows(), 1):
        zone = "ucl" if pos <= 4 else "rel" if n_teams >= 18 and pos > n_teams - 3 else ""
        rows.append(f'<tr><td class="z {zone}"></td><td>{pos}</td><td class="cl"><div class="clb">{badge(team, 26)}{esc(team)}</div></td>'
                    f'<td>{r.P}</td><td>{r.W}</td><td>{r.D}</td><td>{r.L}</td><td>{r.GD:+d}</td><td class="pts">{r.Pts}</td>'
                    f'<td style="white-space:nowrap">{dots(r.Form)}</td></tr>')
    H('<div class="scroll"><table class="tbl"><thead><tr><th class="z"></th><th>#</th><th class="cl">Club</th><th>P</th><th>W</th>'
      '<th>D</th><th>L</th><th>GD</th><th>Pts</th><th>Form</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
      '<div class="key"><span><i style="background:var(--green)"></i>Top 4</span>'
      '<span><i style="background:#C0392B"></i>Bottom 3</span></div>')

    d = season_df
    tot = d.FullTimeHomeGoals + d.FullTimeAwayGoals
    tiles = [(len(d), "Matches played"), (f"{tot.mean():.2f}", "Goals per match"),
             (f"{(d.FullTimeResult == 'H').mean():.0%}", "Home wins"),
             (f"{((d.FullTimeHomeGoals > 0) & (d.FullTimeAwayGoals > 0)).mean():.0%}", "Both teams score")]
    H('<div class="sec">Season stats</div><div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles) + "</div>")

    def board(title, df, col, asc=False):
        top = df.sort_values([col, "GD"], ascending=[asc, not asc]).head(3)
        body = "".join(f'<div class="lb-r">{badge(t, 24)}<span class="nm">{esc(t)}</span><b>{int(r[col])}</b></div>' for t, r in top.iterrows())
        return f'<div class="lb"><div class="lb-t">{title}</div>{body}</div>'

    b1, b2, b3 = st.columns(3)
    b1.markdown(board("Most goals scored", tbl, "GF"), unsafe_allow_html=True)
    b2.markdown(board("Fewest goals conceded", tbl, "GA", True), unsafe_allow_html=True)
    b3.markdown(board("Most clean sheets", tbl, "CS"), unsafe_allow_html=True)

# ---------------- Tab 4: About ----------------
with tab_about:
    H("""
    <div class="about">
    <p>This site estimates the outcome of Premier League matches. It turns past results into win, draw and loss probabilities,
    expected goals and likely scorelines for any pairing of clubs.</p>
    <div class="author"><span class="badge">SMS</span><div><b>Syed Muhammad Sarim</b><br>Designer and developer</div></div>
    <h3>How it works</h3>
    <p>Each club has an Elo rating that rises with good results and falls with bad ones, including an allowance for home advantage.
    The gap between two ratings feeds a logistic regression for win, draw and loss chances, and Poisson regressions for the goals each
    side is expected to score. Those goal estimates give the likelihood of every scoreline.</p>
    <h3>Data</h3>
    <p>Results and fixtures come from football-data.co.uk. The schedule lists kick-off times in UK time; the site converts them to your own time zone. Matchweek numbers are worked out
    from the games each club has played.</p>
    <h3>Limits</h3>
    <p>The model only knows past results. It cannot see injuries, line-ups or managers. Treat the numbers as a starting point, not a
    certainty. This is not betting advice, and the site is not affiliated with the Premier League or any club.</p>
    </div>
    """)
