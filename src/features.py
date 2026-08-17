"""Izvedeni atributi: pokretne statistike, kalendar, Elo, medjusobni dueli.

Funkcije compute_elo, add_head_to_head i add_standings_features pise Jana -
dogovoreno u planu, odeljak "Podela posla".

Pravilo koje vazi za svaku funkciju u ovom modulu: atribut za mec g sme
da zavisi iskljucivo od meceva odigranih strogo pre g. U praksi to znaci
obavezan .shift(1) pre svakog .rolling(...).
"""

import pandas as pd

from src.config import HEAD_TO_HEAD_GAMES, RAW_STAT_COLUMNS, ROLLING_WINDOWS


def build_team_view(df_games):
    """Pretvara tabelu meceva u tabelu iz ugla tima.

    Svaki mec daje dva reda, jedan za domacina i jedan za goste. Ovaj
    oblik cini racunanje pokretnih statistika i sekvenci pravolinijskim.
    """
    home_view = pd.DataFrame({
        "GAME_ID": df_games["GAME_ID"],
        "GAME_DATE_EST": df_games["GAME_DATE_EST"],
        "SEASON": df_games["SEASON"],
        "TEAM_ID": df_games["HOME_TEAM_ID"],
        "OPPONENT_TEAM_ID": df_games["VISITOR_TEAM_ID"],
        "IS_HOME": 1,
        "WON": df_games["HOME_TEAM_WINS"],
    })
    for column in RAW_STAT_COLUMNS:
        home_view[column] = df_games[f"{column}_home"]

    away_view = pd.DataFrame({
        "GAME_ID": df_games["GAME_ID"],
        "GAME_DATE_EST": df_games["GAME_DATE_EST"],
        "SEASON": df_games["SEASON"],
        "TEAM_ID": df_games["VISITOR_TEAM_ID"],
        "OPPONENT_TEAM_ID": df_games["HOME_TEAM_ID"],
        "IS_HOME": 0,
        # gost pobedjuje tacno kada domacin ne pobedjuje - u nba nema nereseno
        "WON": 1 - df_games["HOME_TEAM_WINS"],
    })
    for column in RAW_STAT_COLUMNS:
        away_view[column] = df_games[f"{column}_away"]

    df_team_view = pd.concat([home_view, away_view], ignore_index=True)
    return df_team_view.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)


def add_rolling_stats(df_team_view, windows=ROLLING_WINDOWS):
    """Dodaje pokretne proseke po timu za zadate prozore.

    Vrednosti se pomeraju za jedan mec unazad, tako da nijedan atribut
    ne koristi ishod meca na koji se odnosi. Redosled je uvek
    .shift(1) pa .rolling(...) - obrnuto znaci curenje podataka.
    """
    df_team_view = df_team_view.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)
    grouped = df_team_view.groupby("TEAM_ID")

    # datum poslednjeg odigranog meca pre ovog - koristi se kao gornja
    # granica u sanitarnoj proveri assert_no_leakage
    df_team_view["LAST_GAME_DATE"] = grouped["GAME_DATE_EST"].shift(1)

    rolling_columns = RAW_STAT_COLUMNS + ["WON"]
    for window in windows:
        for column in rolling_columns:
            df_team_view[f"{column}_ROLLING_{window}"] = grouped[column].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df_team_view


def add_calendar_features(df_team_view):
    """Dodaje dane odmora, broj meceva u poslednjih 7 dana i oznaku nove sezone."""
    df_team_view = df_team_view.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)

    # dani odmora od prethodnog meca istog tima; prvi mec tima nema prethodni
    df_team_view["REST_DAYS"] = df_team_view.groupby("TEAM_ID")["GAME_DATE_EST"].diff().dt.days

    # broj meceva u poslednjih 7 dana, bez tekuceg (closed="left" iskljucuje
    # desnu ivicu prozora, a to je upravo tekuci mec)
    def _games_last_7_days(group):
        counts = group.set_index("GAME_DATE_EST")["GAME_ID"].rolling("7D", closed="left").count()
        return pd.Series(counts.values, index=group.index)

    df_team_view["GAMES_LAST_7_DAYS"] = (
        df_team_view.groupby("TEAM_ID", group_keys=False).apply(_games_last_7_days)
    )

    # oznaka prvog meca tima u sezoni - vazi i za sam prvi mec tima u skupu
    df_team_view["IS_SEASON_START"] = (
        df_team_view.groupby("TEAM_ID")["SEASON"].transform(lambda s: s != s.shift(1))
    ).astype(int)

    return df_team_view


def compute_elo(df_games):
    """Racuna Elo rejting oba tima pre svakog meca.

    Racunanje ide hronoloski kroz sve meceve, pa je po konstrukciji
    uzrocno. Kao atribut se koristi rejting PRE meca, a tek zatim se
    rejting azurira. Izmedju sezona rejtinzi se delimicno vracaju ka
    proseku, jer se sastavi timova menjaju.
    """
    # TODO
    raise NotImplementedError


def add_head_to_head(df_games, n_last=HEAD_TO_HEAD_GAMES):
    """Dodaje procenat pobeda protiv istog protivnika u poslednjih n duela."""
    # TODO
    raise NotImplementedError


def add_standings_features(df_games, df_rankings):
    """Dodaje stanje na tabeli na dan meca, iz ranking.csv.

    ranking.csv je snimak stanja po danu, pa se spaja stanje sa datumom
    STRIKTNO pre GAME_DATE_EST - stanje na sam dan meca vec moze da sadrzi
    ishod tog meca.

    Kolone HOME_RECORD i ROAD_RECORD su tekst oblika "28-8" i rastavljaju
    se na broj pobeda i broj poraza, odnosno na procenat pobeda.
    """
    # TODO
    raise NotImplementedError
