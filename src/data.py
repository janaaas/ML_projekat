"""Ucitavanje i osnovno ciscenje sirovih CSV datoteka."""

import pandas as pd

from src.config import KEPT_GAME_TYPES, RAW_DATA_PATH

# Statistika samog meca. Ove kolone se NE smeju koristiti kao atributi za
# predvidjanje tog istog meca - dostupne su tek posto je mec odigran.
GAME_STAT_COLUMNS = [
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
]

# Kolone bez informacije: GAME_STATUS_TEXT je konstanta "Final", a TEAM_ID_home
# i TEAM_ID_away su identicni sa HOME_TEAM_ID odnosno VISITOR_TEAM_ID.
DEGENERATE_COLUMNS = ["GAME_STATUS_TEXT", "TEAM_ID_home", "TEAM_ID_away"]


def load_games():
    """Ucitava games.csv i vraca tabelu meceva sortiranu po datumu.

    Ocekivane kolone: GAME_DATE_EST, GAME_ID, SEASON, HOME_TEAM_ID,
    VISITOR_TEAM_ID, PTS_home, PTS_away, HOME_TEAM_WINS i statistika meca.
    """
    df_games = pd.read_csv(RAW_DATA_PATH / "games.csv", parse_dates=["GAME_DATE_EST"])
    return df_games.sort_values("GAME_DATE_EST").reset_index(drop=True)


def load_teams():
    """Ucitava teams.csv sa nazivima i skracenicama timova."""
    return pd.read_csv(RAW_DATA_PATH / "teams.csv")


def load_rankings():
    """Ucitava ranking.csv sa pozicijama timova po datumima.

    Tabela je snimak stanja po danu, pa spajanje stanja strogo pre datuma
    meca daje legitimne pred-mecevske atribute.
    """
    return pd.read_csv(RAW_DATA_PATH / "ranking.csv", parse_dates=["STANDINGSDATE"])


def game_type(df_games):
    """Vraca prvu cifru GAME_ID, koja kodira tip meca.

    1 pripremni, 2 regularna sezona, 4 plej-of, 5 plej-in.
    """
    return df_games["GAME_ID"].astype(str).str.zfill(8).str[0]


def clean_games(df_games):
    """Ciscenje tabele meceva u cetiri koraka.

    1. Izbacuju se pripremne utakmice. Igraju se rezervnim sastavima i
       njihova statistika ne opisuje snagu tima.
    2. Uklanjaju se duplikati po GAME_ID. Uklanjanje po celom redu ovde ne
       radi, jer se parovi razlikuju samo u ispisu decimale
       (0.343 naspram 0.3429999999999999).
    3. Izbacuju se mecevi bez upisane statistike - imaju popunjen ishod, ali
       prazne statisticke kolone, i nedostaju iz games_details.csv.
    4. Uklanjaju se kolone bez informacije.
    """
    df_clean = df_games[game_type(df_games).isin(KEPT_GAME_TYPES)]
    df_clean = df_clean.drop_duplicates(subset="GAME_ID", keep="first")
    df_clean = df_clean.dropna(subset=GAME_STAT_COLUMNS)
    df_clean = df_clean.drop(columns=DEGENERATE_COLUMNS, errors="ignore")
    return df_clean.reset_index(drop=True)
