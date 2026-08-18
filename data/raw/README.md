# Sirovi podaci

Podaci se ne cuvaju u repozitorijumu. Preuzmite ih pre pokretanja svesaka.

**Skup:** NBA games data, autor Nathan Lauga
**Adresa:** https://www.kaggle.com/datasets/nathanlauga/nba-games

**Opseg:** sezone 2003/04 do 22.12.2022, oko 26.600 meceva. Podnaslov na
Kaggle-u i dalje navodi "2004 season to dec 2020" — to je zastarelo, aktuelna
verzija skupa (v10, objavljena 23.12.2022) sadrzi i sezone 2020/21, 2021/22 i
pocetak 2022/23. Skup se vise ne azurira.

## Preuzimanje kroz pregledac

Otvorite adresu iznad, kliknite Download i raspakujte arhivu u ovaj folder.

## Preuzimanje kroz kaggle alat

    pip install kaggle
    # kaggle.json token se preuzima sa kaggle.com/settings, sekcija API
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download -d nathanlauga/nba-games -p data/raw --unzip

## Ocekivane datoteke

Preuzimaju se sve, komandom iznad.

    games.csv         
    ranking.csv         
    teams.csv           
    games_details.csv   
    players.csv         
