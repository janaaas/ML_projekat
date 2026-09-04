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
prethodno odigranih utakmica oba tima.

Ključno ograničenje je da statistika samog meča — poeni, procenti šuta,
skokovi — u trenutku predviđanja ne postoji. Zato se svi atributi grade
isključivo iz mečeva odigranih **strogo pre** onog koji se predviđa.

Poredimo šest modela, od trivijalnog ka složenom, da bi se videlo koliko svaki
sledeći korak zaista donosi:

| Oznaka | Model | Uloga |
|---|---|---|
| M0 | „uvek pobeđuje domaćin" | trivijalna donja granica |
| M1 | logistička regresija | klasični baseline |
| M2 | XGBoost | jak tabelarni model |
| M3 | obična RNN | ablacija — pokazuje čemu služe kapije |
| M4 | LSTM | glavni model projekta |
| M5 | GRU | poređenje arhitektura |

Mreže su u PyTorch-u. Arhitektura (`src/models.py`, `MatchPredictor`) je
dvograna sa deljenim težinama: ista rekurentna ćelija obrađuje sekvencu
domaćina i sekvencu gosta, pa se završna skrivena stanja spajaju sa
kontekstnim atributima meča i vode u izlazni sloj. Jedan primer je par
sekvenci oblika `(10, 8)` — poslednjih deset mečeva svakog tima, po osam
sirovih statistika — plus kontekstni vektor od 24 vrednosti (Elo, kalendar,
međusobni dueli, stanje na tabeli, za oba tima).

Hiperparametre biramo dvostepenom pretragom na validaciji, posebno po mreži:

| Model | `hidden_size` | `dropout` | `learning_rate` |
|---|---|---|---|
| M3 RNN | 64 | 0.15 | 0.003 |
| M4 LSTM | 32 | 0.30 | 0.003 |
| M5 GRU | 32 | 0.30 | 0.003 |

Svaka mreža se zatim trenira **pet puta**, sa seedovima iz `config.SEEDS`, jer
su početne težine i redosled paketića slučajni. Prijavljujemo prosek; za
grafike i matrice konfuzije služi prolaz sa seedom `RANDOM_STATE`.

## Skup podataka

**NBA games**, autor Nathan Lauga —
[kaggle.com/datasets/nathanlauga/nba-games](https://www.kaggle.com/datasets/nathanlauga/nba-games)

Podaci se **ne čuvaju u repozitorijumu**; uputstvo za preuzimanje je u
[data/raw/README.md](data/raw/README.md).

Skup pokriva 5.10.2003 – 22.12.2022, oko 26.600 mečeva, i više se ne ažurira.
Koristimo `games.csv`, `ranking.csv` i `teams.csv`. Posle čišćenja (izbacivanje
pripremnih utakmica, duplikata i mečeva bez upisane statistike) ostaje **24.848
mečeva**, od kojih 24.830 ima dovoljno istorije da se za oba tima sastavi
sekvenca. `games_details.csv` i `players.csv` nismo
koristile — prvi je ostao van obima, a drugi pokriva samo sezone 2009–2019.

Klase su blago neuravnotežene u korist domaćina (59.1% / 40.9%) — dovoljno da
„uvek pobeđuje domaćin" bude netrivijalna donja granica, ali ne toliko da
zahteva balansiranje skupa.

### Podela na skupove

Podela je **hronološka** — nasumična bi značila da model trenira na
budućnosti i predviđa prošlost.

| Skup | Sezone | Broj mečeva |
|---|---|---|
| Trening | 2003/04 – 2016/17 | 18.092 |
| Validacija | 2017/18 – 2018/19 | 2.624 |
| Test | 2021/22 | 1.323 |
| Izdvojeno (kovid) | 2019/20 – 2020/21 | 2.314 |

Sezone 2019/20 i 2020/21 **ne ulaze** ni u jedan od tri skupa. Prva je
prekinuta 11.3.2020. i završena u „mehuru" u Orlandu na neutralnim terenima bez
publike, a druga je delom igrana pred praznim tribinama. Pošto je prednost
domaćeg terena najjači pojedinačni signal u ovom problemu, te sezone koristimo
odvojeno u svesci 05-02, kao test robusnosti. Sezona 2022/23 je nepotpuna i ne
koristi se.

Važno je šta „ne ulaze" ovde znači: te sezone su izuzete kao **mečevi koje
predviđamo**, ali ne i kao prošlost iz koje se računaju atributi. Sekvenca test
meča iz 2021/22 sadrži ranije mečeve ta dva tima, među njima i one iz 2020/21,
a Elo rejting teče kroz ceo period neprekidno. Tako i treba: u trenutku
predviđanja model sme da koristi sve što se dotad odigralo. Izbačeno je samo
ono što bi pokvarilo merenje — pandemijski mečevi kao primeri za treniranje i
kao mečevi na kojima se modeli ocenjuju.

Za test je uzeta 2021/22 kao poslednja kompletna sezona u skupu odigrana u
normalnom formatu — sa publikom i bez neutralnih terena. I ona pokazuje niži
udeo pobeda domaćina od istorijskog (54.8% naspram 59.9% na treningu). Zašto,
iz ovog skupa se ne može pouzdano utvrditi: sveska 01 pokazuje da je pad počeo
još pre „mehura", pa ga odsustvo publike ne objašnjava u celosti, a nepotpuna
sezona 2022/23 nagoveštava povratak na uobičajen nivo. To ostaje otvoreno
pitanje, ne zaključak.

Za podelu je važnija praktična posledica: donju granicu (model M0) računamo
**na samoj test sezoni**, a ne kao istorijsku vrednost. Time niži nivo ne kvari
poređenje — svaki model se meri prema onome što je na tom istom skupu značilo
„ne znati ništa osim da domaćin češće pobeđuje".

### Kako je sprečeno curenje podataka

Ovo je najopasniji deo problema, pa ga izdvajamo:

- **Statistika samog meča nije atribut.** `PTS_home`, `FG_PCT_home` i slične
  kolone korelišu sa ishodom do 0.43 upravo zato što su njegova *posledica*.
  U model ulaze samo njihove pomerene i pokretne verzije.
- **`shift(1)` pre svakog `rolling(...)`**, tako da pokretni prosek za meč `g`
  koristi isključivo mečeve odigrane strogo pre `g`.
- **Skaler se prilagođava samo na trening skupu** (`sequences.fit_scaler`), pa
  se istim skalerom transformišu validacija i test.
- **Sanitarna provera** `sequences.assert_no_leakage` u svesci 02 potvrđuje da
  za svaki meč najkasniji datum iz kojeg potiču atributi prethodi datumu tog
  meča.
- **Test skup se dodiruje tačno jednom**, u svesci 05-01. Sve odluke o
  modelima donete su na validaciji, a modeli su pre toga zamrznuti.
- **Prag za uzbunu:** tačnost preko 70% smatramo znakom curenja, ne uspehom.
  Nijedan model ga ne prelazi.

## Redosled pregledanja

| Sveska | Sadržaj | Autor |
|---|---|---|
| [01](notebooks/01-eksplorativna-analiza.ipynb) | eksplorativna analiza, balans klasa, prednost domaćeg terena, koji signali su legitimni | Jana |
| [02](notebooks/02-priprema-podataka-i-atributi.ipynb) | izvedeni atributi, Elo, sekvence, hronološka podela i provera curenja | Mina V. |
| [03](notebooks/03-osnovni-modeli.ipynb) | M0, M1, M2 | Jana i Mina V. |
| [04-01](notebooks/04-01-rnn.ipynb) | M3 obična RNN | Jana |
| [04-02](notebooks/04-02-lstm.ipynb) | M4 LSTM, uz ablaciju „sa sekvencama naspram bez njih" | Mina K. |
| [04-03](notebooks/04-03-gru.ipynb) | M5 GRU | Mina V. |
| [05-01](notebooks/05-01-ocena-na-testu.ipynb) | ocena svih šest modela na testu — **jedini dodir test skupa** | Mina K. |
| [05-02](notebooks/05-02-poredjenje-modela.ipynb) | objedinjeni rezultati, statistička značajnost, ROC, kalibracija, kovid sezone, **zaključak projekta** | Jana |
| [05-03](notebooks/05-03-analiza-gresaka.ipynb) | analiza grešaka po tipovima mečeva | Mina V. |

Svaka mreža ima svoju svesku da bi se moglo raditi paralelno; sve tri dele
istu arhitekturu i istu petlju treniranja iz `src/`.

Trojka 05-01/02/03 ujedno je **demo deo projekta**: ništa se u njoj ne trenira,
nego se sačuvani modeli i skaler učitavaju iz `models/` i koriste kroz funkcije
iz `src/`, čime se pokriva funkcionalnost tog paketa.

## Rezultati

Metrike svih modela upisuju se u `reports/rezultati.csv`. Rezultati na **test
skupu** (za mreže prosek po pet seedova):

| Model | Tačnost | ROC-AUC | Log-gubitak | Brier |
|---|---|---|---|---|
| M0 domacin | 0.5480 | 0.5000 | 0.6940 | 0.2504 |
| M1 log. reg. | 0.6417 | 0.6805 | 0.6518 | 0.2288 |
| M2 XGBoost | 0.6395 | 0.6789 | 0.6487 | 0.2275 |
| M3 RNN | 0.6499 | 0.6791 | 0.6444 | 0.2257 |
| M4 LSTM | 0.6444 | 0.6807 | 0.6444 | 0.2259 |
| M5 GRU | 0.6402 | 0.6793 | 0.6447 | 0.2260 |

Gornja granica predvidivosti u NBA ligi je oko 70%: jači tim pobeđuje u 68–72%
mečeva, pa ni model koji savršeno prepoznaje jači tim ne može više od toga.
Objavljeni modeli nad ovim skupom kreću se u rasponu 61–67%, a naša ciljna
zona bila je 63–67%. Svih pet netrivijalnih modela je unutar nje.

**Šta smo našle:**

- **Lestvica se zaustavlja posle prvog koraka.** Svih pet netrivijalnih modela
  je ubedljivo iznad M0 (McNemarov test, p < 0.001 u svih pet parova), ali
  **međusobno nema nijednog statistički značajnog para** — najbliži granici je
  M4 naspram M5 sa p = 0.057, i to bez korekcije za deset poređenja.
- **Glavni nalaz projekta je negativan, i to je u redu.** Arhitektonska
  složenost (RNN → LSTM → GRU) ne donosi ništa merljivo preko logističke
  regresije nad agregatnim atributima. Nalaz stoji na tri oslonca: nijedan par
  nije značajan, modeli daju isti odgovor na preko 90% mečeva, a poredak se
  menja čim se promeni metrika ili seed. Pravi dobitak došao je iz **atributa**
  (Elo, stanje na tabeli, međusobni dueli), ne iz izbora modela.
- **Modeli su nagnuti ka domaćinu** jače nego što test sezona opravdava — učili
  su na periodu sa 59.9% pobeda domaćina, a ocenjuju se na sezoni sa 54.8%.
  Verovatnoće su upotrebljive kao gruba procena, ali bi ih pre ozbiljne upotrebe
  trebalo dodatno kalibrisati.
- **Provera robusnosti dala je nalaz suprotan pretpostavci.** U periodu posle
  prekida sezone 2019/20 svih pet modela gubi 5.0–5.9 procentnih poena, ali M0
  — doslovna mera prednosti domaćeg terena — pada svega 0.8 poena, i ta razlika
  nije statistički značajna (p = 0.71). Prednost domaćeg terena, dakle, ne
  nestaje bez publike; verovatniji uzrok pada je poremećen ritam sezone, koji
  kvari pokretne proseke i Elo a M0 ne dotiče.
- **Greške nisu nasumične.** Analiza po tipovima mečeva pokazuje da svi modeli
  greše sistematski tamo gde je ishod suštinski neizvesniji. Na mečevima sa
  malom razlikom u Elo rejtingu tačnost na testu pada sa 70.7–71.8% na
  57.1–59.1% — razlika višestruko veća od intervala poverenja za te grupe. Kad
  favorit izgubi (482 od 1.323 test meča), tačnost je 8.7–15.8%, ali ta podela
  je delom definiciona: meri koliko model odstupa od Elo favorita, a ne
  popravljivu grešku. Odmor pre meča i početak sezone ne daju stabilan nalaz —
  predznak se menja između validacije i testa.

**Ograničenja.** Test je jedna sezona (1.323 meča), pa su intervali poverenja
široki oko ±2.6 procentna poena i ne bi razdvojili ni razlike od dva poena.
Zaključak „mreže ne pomažu" važi za ovaj skup, ovaj skup atributa i ovaj
horizont predviđanja — ne uopšteno.

## Struktura repozitorijuma

```
src/config.py      putanje, random seed, granice sezona, Elo parametri
src/data.py        učitavanje i čišćenje
src/features.py    pokretne statistike, kalendar, Elo, dueli, tabela
src/sequences.py   građenje sekvenci, podela, PyTorch Dataset
src/models.py      arhitektura mreža (MatchPredictor)
src/train.py       petlja treniranja sa ranim zaustavljanjem
src/evaluate.py    metrike i zbirni zapis rezultata
src/plotting.py    jednoobrazan izgled grafika

notebooks/         sveske 01–05, redosled pregledanja iznad
data/raw/          sirovi CSV fajlovi (van gita, vidi uputstvo)
data/processed/    atributi.csv i sekvence.npz iz sveske 02 (van gita)
models/            sačuvani modeli i skaler
reports/           rezultati i predviđanja
reports/figures/   izvezeni grafici za prezentaciju
prezentacija/      slajdovi za odbranu
```

U `reports/` stoje tri fajla:

| Fajl | Sadržaj |
|---|---|
| `rezultati.csv` | metrike po modelu i skupu; za mreže prosek po pet seedova |
| `rasipanje-semena.csv` | pojedinačne metrike po (model, skup, seed) |
| `predvidjanja-test.csv` | verovatnoće svih šest modela po test meču, za dalju analizu |

Grafike za slajdove sveska 05-02 izvozi u `reports/figures/` pozivom
`plotting.save_figure`, da se ne bi prekopiravali rukom iz sveske.

Sve konstante projekta — random seed, granice sezona, dužina sekvence,
prozori pokretnih statistika, Elo parametri — su u `src/config.py` i nigde
drugde.

## Podešavanje okruženja

**Preporučeni način je conda.** To je jedini način koji radi na svim
platformama, uključujući Intel Mac, za koji PyTorch više ne objavljuje pip
pakete:

```bash
conda env create -f environment.yml
conda activate ml-projekat
```

Alternativa preko pip-a radi na Linux-u, Windows-u i Apple Silicon Mac-u:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verzije su zakucane da bi se izlazi svesaka poklopili. Okruženje koristi
isključivo `conda-forge` kanal i paket `pytorch-cpu`, bez CUDA zavisnosti.

Zatim preuzeti podatke po uputstvu iz [data/raw/README.md](data/raw/README.md)
i pokrenuti sveske redom. Sveska 02 upisuje `data/processed/`, sveske 03 i 04
upisuju `models/`, a sveske 05 samo čitaju — pa se, ako su modeli već u
repozitorijumu, posle 02 može odmah na 05.

Sve sveske se izvršavaju na običnom procesoru, bez grafičke kartice. Najduže
traju sveske 04 — po 14 do 15 minuta svaka, jer pretraga hiperparametara i pet
seedova znače osamdeset do sto dvadeset istreniranih mreža po svesci.

## Sačuvani modeli

Istrenirani modeli i prateći skaler čuvaju se u `models/` i **komituju se u
repozitorijum** — svi su ispod nekoliko megabajta, pa nije potreban spoljni
link.

| Fajl | Sadržaj |
|---|---|
| `scaler.joblib` | `StandardScaler`, prilagođen samo na trening skupu |
| `log_reg.joblib` | M1 logistička regresija |
| `m2_xgboost.joblib` | M2 XGBoost |
| `rnn_seed{7,17,27,37,47}.pt` | M3, po jedan model za svaki seed |
| `lstm_seed{7,17,27,37,47}.pt` | M4, po jedan model za svaki seed |
| `gru_seed{7,17,27,37,47}.pt` | M5, po jedan model za svaki seed |

Mreže se čuvaju kao `state_dict`; `hidden_size` se pri učitavanju čita iz samih
težina, pa se rekonstrukcija ne oslanja na prepisane konstante. M0 nema fajl —
predviđa konstantnu osnovnu stopu izračunatu na trening skupu.

## Literatura

- Rios et al. (2025), *Long-Sequence LSTM Modeling for NBA Game Outcome
  Prediction Using a Novel Multi-Season Dataset*, arXiv:2512.08591
- *Forecasting NCAA Basketball Outcomes with Deep Learning: LSTM vs
  Transformer*, arXiv:2508.02725
- *The application of AI techniques in predicting game outcomes of professional
  basketball league: a systematic review*, PLOS One (2025)
- FiveThirtyEight, *How Our NBA Predictions Work* — metodologija Elo rejtinga
- Materijali sa vežbi