"""Centralna podesavanja projekta.

Sve konstante se definisu ovde i nigde drugde. Ako se seme slucajnosti
prepise u svesci, rezultati prestaju da budu reproducibilni, a upravo
to se proverava kada se sveske ponovo pokrenu.
"""

from pathlib import Path

# --- Putanje ---------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw"
FIGURES_PATH = ROOT_DIR / "reports" / "figures"

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

HEAD_TO_HEAD_GAMES = 5

# --- Elo rejting -----------------------------------------------------------
# Ove konstante uvozi features.py (potpisi funkcija compute_elo i
# add_head_to_head), pa moraju da postoje da bi se ceo modul ucitao - i pre
# nego sto Jana implementira telo tih funkcija.

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_BONUS = 100.0
ELO_SEASON_REGRESSION = 0.25  # regresija ka proseku izmedju sezona
