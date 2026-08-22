"""Centralna podesavanja projekta.

Sve konstante se definisu ovde i nigde drugde. Ako se random seed
prepise u svesci, rezultati prestaju da budu reproducibilni, a upravo
to se proverava kada se sveske ponovo pokrenu.
"""

from pathlib import Path

# --- Putanje ---------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw"
FIGURES_PATH = ROOT_DIR / "reports" / "figures"
RESULTS_FILE = ROOT_DIR / "reports" / "rezultati.csv"

# pripremljeni tenzori iz sveske 02; sveska 04 ih samo ucitava umesto da
# ponovo gradi sekvence
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"

# skaler iz sveske 02 i istrenirani modeli iz sveske 04
MODELS_PATH = ROOT_DIR / "models"

# --- Reproducibilnost ------------------------------------------------------

RANDOM_STATE = 7

# --- Tipovi meca -----------------------------------------------------------
# Prva cifra GAME_ID kodira tip meca. Pripremne utakmice se igraju rezervnim
# sastavima i unose sum, pa se izbacuju u clean_games.

GAME_TYPE_PRESEASON = "1"
GAME_TYPE_REGULAR = "2"
GAME_TYPE_PLAYOFFS = "4"
GAME_TYPE_PLAY_IN = "5"

KEPT_GAME_TYPES = (GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFFS, GAME_TYPE_PLAY_IN)

# Sezona 2019/20 je prekinuta 11.3.2020. i nastavljena u "mehuru" u Orlandu:
# neutralni tereni, bez publike. Prednost domaceg terena tamo prakticno
# nestaje, pa se mecevi pre i posle ovog datuma porede odvojeno.

SEASON_SUSPENSION_DATE = "2020-03-11"

# --- Hronoloska podela -----------------------------------------------------
# Vrednost oznacava godinu pocetka sezone: 2016 je sezona 2016/17.
# Podela je hronoloska, jer bi nasumicna znacila treniranje na buducnosti.
#
# Skup pokriva sezone 2003/04 do 22.12.2022. Sezona 2022/23 je nepotpuna i
# ne koristi se. Sezone 2019/20 i 2020/21 su pod uticajem pandemije, pa se
# drze van glavnog toka i analiziraju odvojeno u svesci 05 - time test ostaje
# normalna sezona sa publikom, a anomalija postaje test robusnosti.

LAST_TRAIN_SEASON = 2016
LAST_VALIDATION_SEASON = 2018
TEST_SEASON = 2021

ANOMALY_SEASONS = (2019, 2020)
UNUSED_SEASONS = (2022,)

# --- Atributi i sekvence ---------------------------------------------------

SEQUENCE_LENGTH = 10
ROLLING_WINDOWS = (5, 10)

# sirove statistike meca iz ugla jednog tima - osnova i za pokretne
# proseke i za sirovi ulaz u sekvence
RAW_STAT_COLUMNS = ["PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]

# atributi koji ulaze u sekvencu jednog proslog meca - sirove vrednosti,
# ne pokretni proseci, da bi mreza sama naucila vremenski obrazac
SEQUENCE_FEATURE_COLUMNS = RAW_STAT_COLUMNS + ["WON", "IS_HOME"]

# kalendarski atributi tima na dan tekuceg meca - poznati unapred (raspored
# je poznat pre poceta), pa ne traze shift(1) kao pokretni proseci
CONTEXT_CALENDAR_COLUMNS = ["REST_DAYS", "GAMES_LAST_7_DAYS", "IS_SEASON_START"]

# koristi se kao podrazumevana vrednost u add_head_to_head - mora da postoji
# da bi se ceo modul features.py ucitao, i pre nego sto Jana implementira
# telo te funkcije
HEAD_TO_HEAD_GAMES = 5

# --- Ucitavanje paketica ---------------------------------------------------
# Polazna vrednost. Memorija nije ogranicenje 
# nego broj koraka gradijenta po epohi: 18092 primera daje 283 koraka.

BATCH_SIZE = 64

# --- Elo rejting -------------------------------------------------------
# Vrednosti prate FiveThirtyEight-ovu metodologiju za NBA; K=20, prednost domaceg terena ~100 poena na Elo skali,
# regresija ka proseku 1/3 na pocetku svake nove sezone.

ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_BONUS = 100
ELO_SEASON_REGRESSION = 1 / 3
