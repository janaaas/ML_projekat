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
