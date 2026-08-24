"""Izvedeni atributi: pokretne statistike, kalendar, Elo, medjusobni dueli.

Funkcije compute_elo, add_head_to_head i add_standings_features pise Jana -
dogovoreno u planu, odeljak "Podela posla".

Pravilo koje vazi za svaku funkciju u ovom modulu: atribut za mec g sme
da zavisi iskljucivo od meceva odigranih strogo pre g. U praksi to znaci
obavezan .shift(1) pre svakog .rolling(...).
"""

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATE_FEATURE_COLUMNS,
    ELO_HOME_BONUS,
    ELO_INITIAL,
    ELO_K,
    ELO_SEASON_REGRESSION,
    GAME_TYPE_REGULAR,
    HEAD_TO_HEAD_GAMES,
    MISSING_RATE_FILL,
    RAW_STAT_COLUMNS,
    ROLLING_WINDOWS,
)


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
    #
    # min_periods=0 je obavezno. Za prozor zadat vremenskim rasponom pandas
    # podrazumeva min_periods=1, pa kada je prethodni mec stariji od sedam dana
    # - a to je svaki prvi mec sezone, gde je razmak oko pet meseci - prozor je
    # prazan i count() vraca NaN umesto nule. Ispravno je nula: tim zaista
    # nije odigrao nijedan mec u prethodnih sedam dana.
    def _games_last_7_days(group):
        counts = (
            group.set_index("GAME_DATE_EST")["GAME_ID"]
            .rolling("7D", closed="left", min_periods=0)
            .count()
        )
        return pd.Series(counts.values, index=group.index)

    df_team_view["GAMES_LAST_7_DAYS"] = (
        df_team_view.groupby("TEAM_ID", group_keys=False).apply(_games_last_7_days)
    )

    # oznaka prvog meca tima u sezoni - vazi i za sam prvi mec tima u skupu
    df_team_view["IS_SEASON_START"] = (
        df_team_view.groupby("TEAM_ID")["SEASON"].transform(lambda s: s != s.shift(1))
    ).astype(int)

    return df_team_view


def compute_elo(df_team_view):
    """Racuna Elo rejting oba tima pre svakog meca.

    Racunanje ide hronoloski kroz sve meceve, pa je po konstrukciji
    uzrocno. Kao atribut se koristi rejting PRE meca, a tek zatim se
    rejting azurira. Izmedju sezona rejtinzi se delimicno vracaju ka
    proseku, jer se sastavi timova menjaju.

    Prednost domaceg terena (ELO_HOME_BONUS) ulazi u racunanje ocekivanog
    ishoda pri azuriranju - standardna Elo praksa. Sama kolona ELO ipak ne
    sadrzi taj bonus kao stalni pomeraj, vec cist rejting tima - IS_HOME je
    zaseban atribut koji modelu vec govori ko je domacin u tekucem mecu.
    """
    df_team_view = df_team_view.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)
    row_by_game_team = {
        (game_id, team_id): row_index
        for row_index, (game_id, team_id) in enumerate(
            zip(df_team_view["GAME_ID"], df_team_view["TEAM_ID"])
        )
    }
    # jedan red po mecu (perspektiva domacina) je dovoljan da se dodje do
    # oba tima preko TEAM_ID/OPPONENT_TEAM_ID - gost bi samo duplirao meceve
    home_games = df_team_view[df_team_view["IS_HOME"] == 1].sort_values(["GAME_DATE_EST", "GAME_ID"])

    current_elo = {}
    last_season_played = {}
    elo_pre_game = np.full(len(df_team_view), np.nan)

    def _rating_before_game(team_id, season):
        rating = current_elo.get(team_id, ELO_INITIAL)
        if last_season_played.get(team_id) not in (None, season):
            rating = ELO_INITIAL + (1 - ELO_SEASON_REGRESSION) * (rating - ELO_INITIAL)
        last_season_played[team_id] = season
        return rating

    for game in home_games.itertuples():
        home_id, away_id = game.TEAM_ID, game.OPPONENT_TEAM_ID
        home_rating = _rating_before_game(home_id, game.SEASON)
        away_rating = _rating_before_game(away_id, game.SEASON)

        elo_pre_game[row_by_game_team[(game.GAME_ID, home_id)]] = home_rating
        elo_pre_game[row_by_game_team[(game.GAME_ID, away_id)]] = away_rating

        expected_home = 1 / (1 + 10 ** (-(home_rating + ELO_HOME_BONUS - away_rating) / 400))
        actual_home = game.WON
        change = ELO_K * (actual_home - expected_home)

        current_elo[home_id] = home_rating + change
        current_elo[away_id] = away_rating - change

    df_team_view["ELO"] = elo_pre_game
    return df_team_view


def add_head_to_head(df_team_view, n_last=HEAD_TO_HEAD_GAMES):
    """Dodaje procenat pobeda protiv istog protivnika u poslednjih n duela.

    Prvi duel sa nekim protivnikom nema istoriju (NaN). Popunjava se sa
    MISSING_RATE_FILL, uz HEAD_TO_HEAD_WIN_PCT_MISSING koja modelu kaze da je
    ta vrednost izmisljena, ne izmerena ravnoteza - inace bi 0.5 izgledalo
    kao "podjednaki timovi", sto ovde niko nije izmerio.
    """
    df_team_view = df_team_view.sort_values(
        ["TEAM_ID", "OPPONENT_TEAM_ID", "GAME_DATE_EST"]
    ).reset_index(drop=True)
    grouped = df_team_view.groupby(["TEAM_ID", "OPPONENT_TEAM_ID"])

    # isti obrazac kao pokretne statistike - shift(1) pa rolling, tako da
    # duel g zavisi iskljucivo od ranijih duela sa istim protivnikom
    win_pct = grouped["WON"].transform(
        lambda s: s.shift(1).rolling(n_last, min_periods=1).mean()
    )
    df_team_view["HEAD_TO_HEAD_WIN_PCT_MISSING"] = win_pct.isna().astype(int)
    df_team_view["HEAD_TO_HEAD_WIN_PCT"] = win_pct.fillna(MISSING_RATE_FILL)
    return df_team_view.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)


def add_standings_features(df_team_view, df_rankings):
    """Dodaje stanje na tabeli na dan meca, iz ranking.csv.

    ranking.csv je snimak stanja po danu, pa se spaja stanje sa datumom
    STRIKTNO pre GAME_DATE_EST - stanje na sam dan meca vec moze da sadrzi
    ishod tog meca.

    Kolone HOME_RECORD i ROAD_RECORD su tekst oblika "28-8" i rastavljaju
    se na broj pobeda i broj poraza, odnosno na procenat pobeda.

    Mecevi pre prvog snimka tabele u sezoni nemaju stanje (NaN); STANDINGS_
    HOME_WIN_PCT i STANDINGS_ROAD_WIN_PCT dodatno mogu biti NaN i posle toga,
    dok tim jos nema odigran nijedan mec kod kuce ili u gostima (0/0). Sve tri
    se popunjavaju sa MISSING_RATE_FILL, svaka sa svojom *_MISSING zastavicom.
    """
    df_rankings = df_rankings.copy()
    season_id = df_rankings["SEASON_ID"].astype(str)

    # SEASON_ID pocinje istom cifrom kao GAME_ID (1 pripremni, 2 regularna
    # sezona) - pripremne meceve vec izbacujemo u clean_games jer ih igraju
    # rezervni sastavi, pa isti razlog vazi i za stanje na tabeli iz njih.
    df_rankings = df_rankings[season_id.str[0] == GAME_TYPE_REGULAR]

    # preostale cifre nose godinu pocetka sezone, isto kodiranje kao SEASON
    # u games.csv. Spajanje mora da postuje granicu sezone - inace bi prvi
    # mecevi nove sezone preuzeli stanje sa kraja prethodne, koje vise ne
    # vazi jer se tabela svake sezone resetuje.
    df_rankings["SEASON"] = df_rankings["SEASON_ID"].astype(str).str[-4:].astype(int)

    home_wins, home_losses = _split_record(df_rankings["HOME_RECORD"])
    road_wins, road_losses = _split_record(df_rankings["ROAD_RECORD"])
    df_rankings["STANDINGS_HOME_WIN_PCT"] = home_wins / (home_wins + home_losses)
    df_rankings["STANDINGS_ROAD_WIN_PCT"] = road_wins / (road_wins + road_losses)
    df_rankings = df_rankings.rename(columns={"W_PCT": "STANDINGS_WIN_PCT"})

    standings_columns = ["STANDINGS_WIN_PCT", "STANDINGS_HOME_WIN_PCT", "STANDINGS_ROAD_WIN_PCT"]
    df_rankings_sorted = df_rankings.sort_values("STANDINGSDATE")[
        ["TEAM_ID", "SEASON", "STANDINGSDATE"] + standings_columns
    ]

    df_team_view = df_team_view.sort_values("GAME_DATE_EST").reset_index(drop=True)

    # merge_asof zahteva IDENTICAN tip za "by" kolone na obe strane. SEASON
    # ovde gore prolazi kroz .astype(int), a taj je platformski zavisan (na
    # Windowsu daje int32, na macOS/Linuxu int64) - pa spajanje puca sa
    # "incompatible merge keys" iako su vrednosti iste. Eksplicitan int64
    # na obe strane to uklanja bez obzira na platformu.
    join_dtypes = {"TEAM_ID": "int64", "SEASON": "int64"}
    df_team_view = df_team_view.astype(join_dtypes)
    df_rankings_sorted = df_rankings_sorted.astype(join_dtypes)

    df_merged = pd.merge_asof(
        df_team_view,
        df_rankings_sorted,
        left_on="GAME_DATE_EST",
        right_on="STANDINGSDATE",
        by=["TEAM_ID", "SEASON"],
        direction="backward",
        allow_exact_matches=False,
    )
    for column in standings_columns:
        df_merged[f"{column}_MISSING"] = df_merged[column].isna().astype(int)
        df_merged[column] = df_merged[column].fillna(MISSING_RATE_FILL)

    return (
        df_merged.drop(columns="STANDINGSDATE")
        .sort_values(["TEAM_ID", "GAME_DATE_EST"])
        .reset_index(drop=True)
    )


def build_aggregate_features(df_team_view, df_games, indices):
    """Gradi ravnu tabelu agregatnih atributa za M1 i M2, po mecu.

    Uzima tacno mecevi iz indices - isti skup redova koji vraca
    build_sequences - tako da klasicni modeli i mreze budu ocenjeni nad
    identicnim mecevima. Spaja se preko (GAME_ID, TEAM_ID), ne preko
    IS_HOME - ta kolona je posle skaliranja realan broj, ne vise 0/1.
    Svaka kolona iz AGGREGATE_FEATURE_COLUMNS se udvostrucuje sa HOME_/AWAY_
    prefiksom; imena kolona ostaju citljiva radi tumacenja koeficijenata
    logisticke regresije.
    """
    games = df_games.loc[indices]
    by_game_team = df_team_view.set_index(["GAME_ID", "TEAM_ID"])[AGGREGATE_FEATURE_COLUMNS]

    home_features = (
        by_game_team.loc[list(zip(games["GAME_ID"], games["HOME_TEAM_ID"]))]
        .add_prefix("HOME_")
        .reset_index(drop=True)
    )
    away_features = (
        by_game_team.loc[list(zip(games["GAME_ID"], games["VISITOR_TEAM_ID"]))]
        .add_prefix("AWAY_")
        .reset_index(drop=True)
    )
    return pd.concat([home_features, away_features], axis=1)


def _split_record(record_column):
    """Rastavlja tekst oblika '28-8' na dva niza brojeva: pobede i porazi."""
    wins_losses = record_column.str.split("-", expand=True).astype(int)
    return wins_losses[0], wins_losses[1]
