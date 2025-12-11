import csv
import re
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup

BASE = "https://www.formula1.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; F1ProjectScraper/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def get_race_result_paths(year: int):
  
    url = f"{BASE}/en/results/{year}/races"
    html = get_html(url)

    pattern = rf'(/en/results/{year}/races/\d+/[a-z-]+/race-result)'
    paths = re.findall(pattern, html)

    seen = set()
    ordered = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    if not ordered:
        raise RuntimeError(f"Could not find any race-result URLs for {year}. Check the regex or site layout.")

    return ordered


def find_table_by_header(soup: BeautifulSoup, header_keywords):
 
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        header_text = " ".join(th.get_text(" ", strip=True) for th in header_cells)
        if all(k in header_text for k in header_keywords):
            return table
    return None


def scrape_starting_grid(url: str):
 
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    table = find_table_by_header(soup, ["Pos", "Driver", "Time"])
    if table is None:
        print(f"[WARN] No starting grid table found for {url}")
        return {}

    results = {}
    body = table.find("tbody") or table
    for row in body.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cols) < 5:
            continue
        # Expected columns (approx): Pos, No, Driver, Team, Time
        pos = cols[0]
        number = cols[1]
        grid_time = cols[4]  # can be blank
        results[number] = {
            "grid_position": pos,
            "grid_time": grid_time,
        }
    return results


def scrape_race_result(url: str, year: int, round_idx: int):
  
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # Race name
    race_header = soup.find("h1") or soup.find("h2")
    race_name = race_header.get_text(" ", strip=True) if race_header else f"Round {round_idx}"

    # Extract slug from URL
    m = re.search(r"/races/\d+/([a-z-]+)/race-result", url)
    race_slug = m.group(1) if m else "unknown"

    table = find_table_by_header(soup, ["Pos", "Driver", "Laps"])
    if table is None:
        print(f"[WARN] No race result table found for {url}")
        return {
            "race_meta": {
                "year": year,
                "round": round_idx,
                "race_name": race_name,
                "race_slug": race_slug,
                "total_laps": None,
            },
            "per_driver": {},
        }

    race_meta = {
        "year": year,
        "round": round_idx,
        "race_name": race_name,
        "race_slug": race_slug,
        "total_laps": None,
    }

    per_driver = {}
    body = table.find("tbody") or table
    winner_number = None
    total_laps = None

    for row in body.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if not cols:
            continue

        
        if "No results available" in " ".join(cols): #hopefully handles the issue with Imola 23 missing data and if tehres other stuff like that too
            per_driver = {}
            winner_number = None
            total_laps = None
            break

        
        if len(cols) < 7:
            continue

        
        pos_raw = cols[0]
        number = cols[1]
        driver_name_full = cols[2]
        team_name = cols[3]
        laps = cols[4]
        time_or_status = cols[5]
        points = cols[6] if len(cols) > 6 else "0"

        tokens = driver_name_full.split()
        driver_code = tokens[-1] if len(tokens) >= 2 else ""
        driver_name = " ".join(tokens[:-1]) if len(tokens) >= 2 else driver_name_full

        try:
            final_position = int(pos_raw)
        except ValueError:
            final_position = None  

        if final_position == 1:
            winner_number = number
            try:
                total_laps = int(laps)
            except ValueError:
                total_laps = None

        per_driver[number] = {
            "final_position_raw": pos_raw,
            "final_position": final_position,
            "driver_number": number,
            "driver_name": driver_name,
            "driver_code": driver_code,
            "team_name": team_name,
            "laps_completed": laps,
            "time_or_status": time_or_status,
            "race_time": time_or_status,
            "points": points,
        }

    # Attach winner flag
    for num, info in per_driver.items():
        info["is_winner"] = 1 if num == winner_number else 0

    race_meta["total_laps"] = total_laps
    return {
        "race_meta": race_meta,
        "per_driver": per_driver,
    }


def scrape_pit_stops(url: str):
    
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    table = find_table_by_header(soup, ["Stops", "Lap", "Time"])
    if table is None:
        print(f"[WARN] No pit stop table found for {url}")
        return {}

    body = table.find("tbody") or table

    data = defaultdict(
        lambda: {
            "pit_stops": 0,
            "first_pit_lap": None,
            "last_pit_lap": None,
            "total_pit_time": 0.0,
        }
    )

    for row in body.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cols) < 6:
            continue
        # Roughly: Stops, No, Driver, Team, Lap, Time of Day, Time, Total
        lap_str = cols[4]
        time_str = cols[6] if len(cols) > 6 else ""
        number = cols[1]

        try:
            lap = int(lap_str)
        except ValueError:
            lap = None

        try:
            pit_time = float(time_str)
        except ValueError:
            pit_time = 0.0

        info = data[number]
        info["pit_stops"] += 1
        if lap is not None:
            if info["first_pit_lap"] is None or lap < info["first_pit_lap"]:
                info["first_pit_lap"] = lap
            if info["last_pit_lap"] is None or lap > info["last_pit_lap"]:
                info["last_pit_lap"] = lap
        info["total_pit_time"] += pit_time

    result = {}
    for num, info in data.items():
        info["total_pit_time"] = f"{info['total_pit_time']:.3f}"
        result[num] = info

    return result


def scrape_fastest_laps(url: str):
    
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    table = find_table_by_header(soup, ["Pos", "Lap", "Avg. Speed"])
    if table is None:
        # Older style header fallback
        table = find_table_by_header(soup, ["Fastest Laps", "Avg. Speed"])
    if table is None:
        print(f"[WARN] No fastest laps table found for {url}")
        return {}

    body = table.find("tbody") or table

    data = {}
    for row in body.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cols) < 8:
            continue
        # Expected: Pos, No, Driver, Team, Lap, Time of Day, Time, Avg. Speed
        pos = cols[0]
        number = cols[1]
        lap_str = cols[4]
        lap_time = cols[6]
        avg_speed = cols[7]

        data[number] = {
            "fastest_lap_rank": pos,
            "fastest_lap_lap": lap_str,
            "fastest_lap_time": lap_time,
            "fastest_lap_avg_speed": avg_speed,
        }

    return data


def scrape_year(year: int):

    race_result_paths = get_race_result_paths(year)
    print(f"\n Year {year}: found {len(race_result_paths)} race-result URLs")
    for p in race_result_paths:
        print("  ", p)

    all_rows = []

    for round_idx, rel_path in enumerate(race_result_paths, start=1):
        race_result_url = BASE + rel_path
        base_prefix = rel_path.rsplit("/", 1)[0]

        starting_grid_url = f"{BASE}{base_prefix}/starting-grid"
        pit_stop_url = f"{BASE}{base_prefix}/pit-stop-summary"
        fastest_laps_url = f"{BASE}{base_prefix}/fastest-laps"

        print(f"\n[Year {year} Round {round_idx}] Scraping:")
        print("  Race result    :", race_result_url)
        print("  Starting grid  :", starting_grid_url)
        print("  Pit stops      :", pit_stop_url)
        print("  Fastest laps   :", fastest_laps_url)

        rr = scrape_race_result(race_result_url, year, round_idx)
        race_meta = rr["race_meta"]
        per_driver = rr["per_driver"]

        grid_info = scrape_starting_grid(starting_grid_url)
        pit_info = scrape_pit_stops(pit_stop_url)
        fl_info = scrape_fastest_laps(fastest_laps_url)

        for number, base_info in per_driver.items():
            row = {
                "year": race_meta["year"],
                "round": race_meta["round"],
                "race_slug": race_meta["race_slug"],
                "race_name": race_meta["race_name"],
                "total_laps": race_meta["total_laps"],
                "driver_number": base_info["driver_number"],
                "driver_code": base_info["driver_code"],
                "driver_name": base_info["driver_name"],
                "team_name": base_info["team_name"],
                "grid_position": "",
                "grid_time": "",
                "final_position_raw": base_info["final_position_raw"],
                "final_position": base_info["final_position"],
                "laps_completed": base_info["laps_completed"],
                "time_or_status": base_info["time_or_status"],
                "race_time": base_info["race_time"],
                "points": base_info["points"],
                "pit_stops": "",
                "first_pit_lap": "",
                "last_pit_lap": "",
                "total_pit_time": "",
                "fastest_lap_rank": "",
                "fastest_lap_lap": "",
                "fastest_lap_time": "",
                "fastest_lap_avg_speed": "",
                "is_winner": base_info["is_winner"],
            }

            if number in grid_info:
                row["grid_position"] = grid_info[number]["grid_position"]
                row["grid_time"] = grid_info[number]["grid_time"]

            if number in pit_info:
                row["pit_stops"] = pit_info[number]["pit_stops"]
                row["first_pit_lap"] = pit_info[number]["first_pit_lap"]
                row["last_pit_lap"] = pit_info[number]["last_pit_lap"]
                row["total_pit_time"] = pit_info[number]["total_pit_time"]

            if number in fl_info:
                row["fastest_lap_rank"] = fl_info[number]["fastest_lap_rank"]
                row["fastest_lap_lap"] = fl_info[number]["fastest_lap_lap"]
                row["fastest_lap_time"] = fl_info[number]["fastest_lap_time"]
                row["fastest_lap_avg_speed"] = fl_info[number]["fastest_lap_avg_speed"]

            all_rows.append(row)

        time.sleep(1.0)  # don't hammer the server coz it KEEPS DYING ON ME

    return all_rows


def main():
    # Years we want to scrape
    years = [2021, 2025, 2024, 2023, 2022, 2020, 2019, 2018, 2017, 2016,2015, 2014]

    rows = []
    for year in years:
        rows.extend(scrape_year(year))

    out_file = "f1_raw_2014-2025.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_file}")
    else:
        print("it broke")


if __name__ == "__main__":
    main()
