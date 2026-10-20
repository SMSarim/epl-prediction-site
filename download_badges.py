"""
download_badges.py - fetch club crests into assets/badges/ so the site shows real badges.

Run from the epl-site folder:   python download_badges.py

Tries two public crest sources for each club (football-data.org first, then the Premier League's own badge server) and saves
assets/badges/<club name as used in matches.csv>.svg or .png. The site picks these files up automatically; any club without a
file keeps its coloured code badge.

It also writes assets/badges/preview.html. OPEN IT and check every crest matches its label: the IDs below come from memory of the
two sources' numbering, and a wrong one would show the wrong club. Delete a wrong file (the site falls back to the code badge)
or drop in your own image named exactly like the club, e.g. "Man City.png".

Crests are trademarks of the clubs. Fine for a personal or learning project; ask the clubs before using them on a public site.
"""
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "assets" / "badges"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "image/svg+xml,image/png,image/*;q=0.8"}

# club name in matches.csv -> (football-data.org team id, Premier League badge code)
IDS = {
    "Arsenal": (57, 3), "Aston Villa": (58, 7), "Bournemouth": (1044, 91), "Brentford": (402, 94), "Brighton": (397, 36),
    "Burnley": (328, 90), "Chelsea": (61, 8), "Coventry": (1076, None), "Crystal Palace": (354, 31), "Everton": (62, 11),
    "Fulham": (63, 54), "Hull": (322, 88), "Ipswich": (349, 40), "Leeds": (341, 2), "Leicester": (338, 13),
    "Liverpool": (64, 14), "Man City": (65, 43), "Man United": (66, 1), "Newcastle": (67, 4), "Nott'm Forest": (351, 17),
    "Sheffield United": (356, 49), "Southampton": (340, 20), "Sunderland": (71, 56), "Tottenham": (73, 6),
    "West Ham": (563, 21), "Wolves": (76, 39), "Luton": (389, 102), "Norwich": (68, 45), "Watford": (346, 57),
    "West Brom": (74, 35),
}


def sources(fd_id, pl_code):
    s = [f"https://crests.football-data.org/{fd_id}.svg", f"https://crests.football-data.org/{fd_id}.png"]
    if pl_code is not None:
        s.append(f"https://resources.premierleague.com/premierleague/badges/100/t{pl_code}.png")
    return s


def kind(data):
    """'svg' / 'png' if the bytes really are that image type, else None (blocks HTML error pages)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    head = data[:600].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in data[:2000].lower()):
        return "svg"
    return None


def get(url, tries=3):
    for n in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and n < tries - 1:                  # rate limited: wait, then retry
                time.sleep(5 * (n + 1))
                continue
            raise


def have(team):
    return next((OUT / f"{team}.{e}" for e in ("svg", "png", "webp") if (OUT / f"{team}.{e}").exists()), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for team, (fd_id, pl_code) in IDS.items():
        if have(team):
            continue
        last = "no source worked"
        for url in sources(fd_id, pl_code):
            try:
                data = get(url)
                ext = kind(data)
                if ext:
                    (OUT / f"{team}.{ext}").write_bytes(data)
                    print(f"  saved {team}  <- {url}")
                    break
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        else:
            print(f"  could not get {team} ({last})")
        time.sleep(0.6)

    got = [(t, have(t)) for t in IDS if have(t)]
    cells = "".join(f'<div class="c"><img src="{urllib.parse.quote(p.name)}"><span>{t}</span></div>' for t, p in got)
    (OUT / "preview.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Badge check</title><style>body{font-family:sans-serif;display:flex;flex-wrap:wrap;gap:14px;padding:20px}"
        ".c{width:130px;text-align:center}img{width:80px;height:80px;object-fit:contain}span{display:block;margin-top:6px}</style>"
        "<h2 style='width:100%'>Check each crest matches its label</h2>" + cells, encoding="utf-8")
    print(f"\nDone: {len(got)}/{len(IDS)} badges. Open {OUT / 'preview.html'} to check them, then restart the site.")


if __name__ == "__main__":
    main()
