# Predviđanje ishoda NBA mečeva rekurentnim neuronskim mrežama

Projekat iz kursa **Mašinsko učenje**, Matematički fakultet, Univerzitet u
Beogradu, letnji semestar 2025/2026.

## Članovi tima

| Ime | Indeks | GitHub |
|---|---|---|
| Jana Stojanović | 1066/2023 | [@janaaas](https://github.com/janaaas) |
| Mina Velebit | 1101/2024 | [@minavelebit](https://github.com/minavelebit) |
| Mina Kovandžić | 1031/2024 | [@minakovandzic](https://github.com/minakovandzic) |

## Problem

Predviđamo pobednika NBA meča **pre nego što je odigran**, na osnovu sekvenci
prethodno odigranih utakmica oba tima. Statistika samog meča u trenutku
predviđanja ne postoji, pa se svi atributi grade isključivo iz mečeva
odigranih pre onog koji se predviđa.

## Plan

Poredimo nekoliko modela, od trivijalnog ka složenom:

- **M0** — „uvek pobeđuje domaćin", trivijalna donja granica
- **M1** — logistička regresija nad agregatnim atributima
- **M2** — XGBoost nad istim atributima
- **M3, M4, M5** — rekurentne mreže (obična RNN, LSTM, GRU) nad sekvencama
  prethodnih mečeva, u PyTorch-u

## Skup podataka

**NBA games**, autor Nathan Lauga —
[kaggle.com/datasets/nathanlauga/nba-games](https://www.kaggle.com/datasets/nathanlauga/nba-games)

Podaci se ne čuvaju u repozitorijumu; uputstvo za preuzimanje je u
[data/raw/README.md](data/raw/README.md).

## Struktura

```
src/         zajednički kod (učitavanje podataka, atributi, modeli, treniranje)
notebooks/   sveske sa analizom i eksperimentima, redom 01–05
data/        podaci (ne prate se u git-u)
models/      sačuvani istrenirani modeli
reports/     rezultati i metrike
```

## Podešavanje okruženja

```bash
conda env create -f environment.yml
conda activate ml-projekat
```

Alternativa preko pip-a:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

