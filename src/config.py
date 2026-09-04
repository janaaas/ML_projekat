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

# pojedinacna semena, po jedan red na (model, skup, seed); rezultati.csv drzi
# samo prosek
SEED_RESULTS_FILE = ROOT_DIR / "reports" / "rasipanje-semena.csv"

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
# neutralni tereni, bez publike. Ocekivanje je bilo da prednost domaceg terena
# tamo nestane, pa se mecevi pre i posle ovog datuma porede odvojeno.
#
# Sveska 05-02 je to ocekivanje oborila: M0, koji je doslovna mera te prednosti,
# pada svega 0.8 procentnih poena (55.1% -> 54.3%), a razlika nije statisticki
# znacajna (z = 0.37, p = 0.71). Prednost domaceg terena, dakle, ne nestaje;
# da li uopste slabi, ovi podaci ne mogu da kazu. Datum ostaje granica poredjenja
# jer M1-M5 tu ipak gube 5-6 poena - ali iz drugog razloga, najverovatnije zbog
# poremecenog ritma sezone koji kvari pokretne proseke i Elo.

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

# atributi tima na dan tekuceg meca koji ne ulaze u sekvencu vec direktno u
# kontekst - kalendarski (poznati unapred, ne traze shift(1)) i agregatni
# (elo, dueli, tabela - vec su pred-meceve po konstrukciji funkcije koja ih
# racuna). *_MISSING kolone prate atribute koji ponekad nemaju istoriju (prvi
# duel sa protivnikom, pocetak sezone bez snimka tabele) - popunjavaju se u
# add_head_to_head i add_standings_features, sa MISSING_RATE_FILL nize
CONTEXT_CALENDAR_COLUMNS = [
    "REST_DAYS", "GAMES_LAST_7_DAYS", "IS_SEASON_START",
    "ELO",
    "HEAD_TO_HEAD_WIN_PCT", "HEAD_TO_HEAD_WIN_PCT_MISSING",
    "STANDINGS_WIN_PCT", "STANDINGS_WIN_PCT_MISSING",
    "STANDINGS_HOME_WIN_PCT", "STANDINGS_HOME_WIN_PCT_MISSING",
    "STANDINGS_ROAD_WIN_PCT", "STANDINGS_ROAD_WIN_PCT_MISSING",
]

# agregatni atributi za klasicne modele M1 i M2 - isti CONTEXT_CALENDAR_COLUMNS
# plus pokretni proseci. Nikad sirove RAW_STAT_COLUMNS/WON vrednosti bez
# pomeraja - to bi bila statistika samog meca koji se predvidja
AGGREGATE_FEATURE_COLUMNS = CONTEXT_CALENDAR_COLUMNS + [
    f"{column}_ROLLING_{window}"
    for window in ROLLING_WINDOWS
    for column in RAW_STAT_COLUMNS + ["WON"]
]

# vrednost kojom se popunjavaju NaN u procentima pobeda bez dovoljno istorije
# (prvi duel sa protivnikom, pocetak sezone pre prvog snimka tabele) - 0.5 jer
# "nema podataka" nije ni prednost ni mana, uz pratecu _MISSING zastavicu koja
# modelu kaze da je ta polovina izmisljena, a ne izmerena ravnoteza
MISSING_RATE_FILL = 0.5

# koristi se kao podrazumevana vrednost u add_head_to_head - mora da postoji
# da bi se ceo modul features.py ucitao, i pre nego sto Jana implementira
# telo te funkcije
HEAD_TO_HEAD_GAMES = 5

# --- Ucitavanje paketica ---------------------------------------------------
# Polazna vrednost. Memorija nije ogranicenje 
# nego broj koraka gradijenta po epohi: 18092 primera daje 283 koraka.

BATCH_SIZE = 64

# --- Pretraga hiperparametara ----------------------------------------------
# Ista mreza vrednosti za sve tri rekurentne mreze.
#
# Izabrane vrednosti se OVDE ne upisuju - one su rezultat merenja i smeju da se
# razlikuju po mrezi. Ovde stoji prostor pretrage, u svesci ishod.

HIDDEN_SIZE_GRID = (32, 64, 128)
DROPOUT_GRID = (0.0, 0.15, 0.3)
LEARNING_RATE_GRID = (3e-4, 1e-3, 3e-3)

# --- Semena za ponovljeno treniranje ---------------------------------------
# Inicijalizacija tezina i redosled paketica su slucajni, pa jedno seme moze
# prividno da izdvoji jednu mrezu. Zato se svaka mreza trenira vise puta i prijavljuje prosek.
#
# Ista semena za sve tri mreze.

SEEDS = (7, 17, 27, 37, 47)

# --- Dvostepena pretraga ----------------------------------------------------
N_FINALISTS = 5

# --- Elo rejting -------------------------------------------------------
# Vrednosti prate FiveThirtyEight-ovu metodologiju za NBA; K=20, prednost domaceg terena ~100 poena na Elo skali,
# regresija ka proseku 1/3 na pocetku svake nove sezone.

ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_BONUS = 100
ELO_SEASON_REGRESSION = 1 / 3
