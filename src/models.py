"""Arhitekture rekurentnih mreza nad sekvencama meceva.

Jedna klasa pokriva RNN, LSTM i GRU, jer se arhitektura ne menja - menja se
samo tip celije.
"""

import torch
import torch.nn as nn

CELL_TYPES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class MatchPredictor(nn.Module):
    """Dvograna mreza sa deljenim tezinama nad sekvencama oba tima.

    Ista celija obradjuje obe sekvence, jer je "forma tima" isti pojam bez
    obzira ko je domacin - time se broj parametara prepolovljuje.

    Prednost domaceg terena zato pada na izlazni sloj: prvih hidden_size
    brojeva spojenog vektora je uvek domacin, pa dobija druge tezine. Kontekst
    nema kolonu o tome ko je domacin - IS_HOME u sekvenci se odnosi na PROSLE
    meceve.

    Parametar cell ("rnn", "lstm", "gru") daje redom modele M3, M4 i M5.

    forward vraca sirove logite oblika (B,) za BCEWithLogitsLoss; sigmoid ide
    tek pre racunanja metrika.
    """

    def __init__(self, n_features, n_context, cell="lstm", hidden_size=64, dropout=0.3):
        super().__init__()
        if cell not in CELL_TYPES:
            raise ValueError(f"cell mora biti iz {sorted(CELL_TYPES)}, dobijeno {cell!r}")

        # batch_first=True jer DataLoader daje (B, L, F), a torch podrazumeva
        # (L, B, F) - bez ovoga bi sloj velicinu paketica protumacio kao
        # duzinu sekvence
        self.encoder = CELL_TYPES[cell](
            input_size=n_features, hidden_size=hidden_size, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * hidden_size + n_context, 1)

    def forward(self, seq_home, seq_away, context):
        # dropout pogadja samo ono sto je mreza naucila iz sekvenci.
        h_home = self.dropout(self._encode(seq_home))
        h_away = self.dropout(self._encode(seq_away))

        combined = torch.cat([h_home, h_away, context], dim=1)
        return self.output(combined).squeeze(-1)

    def _encode(self, sequence):
        """Vraca poslednje skriveno stanje jedne sekvence, oblika (B, hidden_size).

        Uzima out[:, -1, :] umesto h_n jer nn.LSTM vraca (output, (h_n, c_n)),
        a nn.RNN i nn.GRU (output, h_n) - indeksiranje izlaza drzi jednu granu
        koda za sve tri celije.
        """
        out, _ = self.encoder(sequence)
        return out[:, -1, :]
