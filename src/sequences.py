"""Gradjenje sekvenci prethodnih meceva i podela na skupove."""

import numpy as np

from sklearn import preprocessing

import torch
from torch.utils.data import Dataset

from src.config import (
    ANOMALY_SEASONS,
    CONTEXT_CALENDAR_COLUMNS,
    LAST_TRAIN_SEASON,
    LAST_VALIDATION_SEASON,
    SEQUENCE_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TEST_SEASON,
    UNUSED_SEASONS,
)

# sve kolone koje skaliranje mora da pokrije - i one koje idu u sekvence i
# one koje idu u kontekst, jer fit_scaler radi na ravnoj tabeli pre nego
# sto se ista podeli na sekvence i kontekst
SCALED_COLUMNS = SEQUENCE_FEATURE_COLUMNS + CONTEXT_CALENDAR_COLUMNS


def build_sequences(df_team_view, df_games, length=SEQUENCE_LENGTH):
    """Gradi tenzore sekvenci za oba tima na svakom mecu.

    Vraca petorku (seq_domacin, seq_gost, kontekst, labele, indeksi):
    tenzore oblika (broj_meceva, length, broj_atributa) za domacina i goste,
    matricu kontekstnih atributa meca, oznake (HOME_TEAM_WINS) i indekse
    redova iz df_games koji su prezileli filter. Mecevi bez dovoljno
    istorije se izbacuju.

    Sekvenca meca g sme da sadrzi iskljucivo meceve odigrane strogo pre g.
    Krace sekvence se dopunjuju nulama na POCETKU, ne na kraju, da bi
    poslednji vremenski korak uvek bio najskoriji stvarni mec.
    """
    df_team_view = df_team_view.sort_values(["TEAM_ID", "GAME_DATE_EST"]).reset_index(drop=True)
    team_history = {
        team_id: group.reset_index(drop=True) for team_id, group in df_team_view.groupby("TEAM_ID")
    }
    # pozicija svakog meca unutar niza meceva tog tima (0 = prvi mec u skupu)
    position_within_team = df_team_view.groupby("TEAM_ID").cumcount()
    position_lookup = dict(
        zip(zip(df_team_view["GAME_ID"], df_team_view["TEAM_ID"]), position_within_team)
    )

    home_sequences, away_sequences, context_rows = [], [], []
    labels, surviving_indices = [], []

    for row_index, game in df_games.iterrows():
        home_id, away_id, game_id = game["HOME_TEAM_ID"], game["VISITOR_TEAM_ID"], game["GAME_ID"]
        home_position = position_lookup[(game_id, home_id)]
        away_position = position_lookup[(game_id, away_id)]

        # mec bez ijednog odigranog meca pre sebe nema istoriju - sekvenca
        # bi bila sva od nula i ne nosi signal, pa se izbacuje
        if home_position == 0 or away_position == 0:
            continue

        home_row = team_history[home_id].iloc[home_position]
        away_row = team_history[away_id].iloc[away_position]

        home_sequences.append(_team_sequence(team_history[home_id], home_position, length))
        away_sequences.append(_team_sequence(team_history[away_id], away_position, length))
        context_rows.append(np.concatenate([
            home_row[CONTEXT_CALENDAR_COLUMNS].to_numpy(dtype=np.float32),
            away_row[CONTEXT_CALENDAR_COLUMNS].to_numpy(dtype=np.float32),
        ]))
        labels.append(game["HOME_TEAM_WINS"])
        surviving_indices.append(row_index)

    return (
        np.stack(home_sequences),
        np.stack(away_sequences),
        np.stack(context_rows),
        np.array(labels, dtype=np.float32),
        np.array(surviving_indices),
    )


def _team_sequence(team_games, position, length):
    """Vraca sekvencu duzine `length` sa poslednjih `position` meceva tima.

    Popunjava nulama na pocetku ako je isteklo manje od `length` meceva.
    """
    start = max(0, position - length)
    window = team_games.iloc[start:position][SEQUENCE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if len(window) < length:
        padding = np.zeros((length - len(window), len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32)
        window = np.vstack([padding, window])
    return window


def chronological_split(df_games, indices):
    """Deli indekse na trening, validaciju i test po sezonama.

    Podela je hronoloska: nasumicna bi znacila da model trenira na
    buducnosti i predvidja proslost.

    Granice su u config.py. Sezone iz ANOMALY_SEASONS (pandemija) i
    UNUSED_SEASONS (nepotpuna sezona) ne ulaze ni u jedan od tri skupa -
    anomalne se analiziraju odvojeno u svesci 05, kao test robusnosti.

    Vraca cetiri niza indeksa - pozicije UNUTAR `indices`, spremne za
    direktno secenje tenzora koje vraca build_sequences (npr.
    home_sequences[trening]).
    """
    seasons = df_games.loc[indices, "SEASON"].to_numpy()
    positions = np.arange(len(indices))

    is_anomaly = np.isin(seasons, ANOMALY_SEASONS)
    is_unused = np.isin(seasons, UNUSED_SEASONS)
    is_excluded = is_anomaly | is_unused

    train = positions[(seasons <= LAST_TRAIN_SEASON) & ~is_excluded]
    validation = positions[(seasons > LAST_TRAIN_SEASON) & (seasons <= LAST_VALIDATION_SEASON) & ~is_excluded]
    test = positions[(seasons == TEST_SEASON) & ~is_excluded]
    anomaly = positions[is_anomaly]

    return train, validation, test, anomaly


def fit_scaler(X_train):
    """Prilagodjava StandardScaler iskljucivo na trening skupu.

    X_train je ravna tabela (df_team_view ogranicen na trening sezone).
    Skaliranje ide PRE build_sequences, na kolonama iz SCALED_COLUMNS, tako
    da i sekvence i kontekst izadju vec skalirani sa istim scalerom.
    Isti scaler se posle koristi za transform nad validacijom i testom -
    prilagodjavanje na celom skupu je curenje podataka.
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train[SCALED_COLUMNS])
    return scaler


def assert_no_leakage(df_features, feature_source_date_columns, game_date_column):
    """Sanitarna provera: nijedan atribut ne sme poticati iz buducnosti.

    Za svaki mec i za svaku prosledjenu izvornu kolonu potvrdjuje da je
    najkasniji datum iz kojeg poticu atributi strogo manji od datuma tog
    meca. Ovo je dokaz ispravnosti na koji se moze pokazati na odbrani, pa
    se poziva u svesci 02 i ostaje u njoj.

    feature_source_date_columns moze biti jedno ime kolone ili lista imena -
    LAST_GAME_DATE pokriva pokretne statistike i Elo (obe se azuriraju posle
    svakog odigranog meca, istim datumom), LAST_HEAD_TO_HEAD_DATE
    medjusobne duele, STANDINGS_SOURCE_DATE stanje na tabeli.
    """
    if isinstance(feature_source_date_columns, str):
        feature_source_date_columns = [feature_source_date_columns]

    for source_date_column in feature_source_date_columns:
        # NaT (nema izvornog datuma - prvi mec tima, prvi duel sa
        # protivnikom ili mec pre prvog snimka tabele) je u poredjenju uvek
        # False, sto je ispravno - nema atributa znaci nema ni curenja
        is_leaking = df_features[source_date_column] >= df_features[game_date_column]
        n_leaking = int(is_leaking.sum())
        assert n_leaking == 0, (
            f"{n_leaking} meceva ima '{source_date_column}' iz buducnosti ili sa istog dana"
        )


class MatchSequenceDataset(Dataset):
    """PyTorch Dataset nad sekvencama meceva.

    Vraca cetvorku (sekvenca_domacin, sekvenca_gost, kontekst, oznaka),
    gde su prve tri torch tenzori tipa float32, a oznaka float32 skalar.

    Nasledjuje torch.utils.data.Dataset i implementira __len__ i __getitem__.
    Oblik tenzora mora da se poklapa sa izlazom build_sequences.
    """

    def __init__(self, home_sequences, away_sequences, context, labels):
        self.home_sequences = torch.as_tensor(home_sequences, dtype=torch.float32)
        self.away_sequences = torch.as_tensor(away_sequences, dtype=torch.float32)
        self.context = torch.as_tensor(context, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.home_sequences[index], self.away_sequences[index], self.context[index], self.labels[index]
