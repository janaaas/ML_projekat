"""Metrike i zbirni zapis rezultata.

Svi modeli upisuju rezultate u isti reports/rezultati.csv sa istim
kolonama. 

Metrike se ne svode na tacnost. Klase su blizu ravnoteze
(oko 59% naspram 41%), pa tacnost sama ne razlikuje model koji pogadja
sigurne meceve od modela koji je dobro kalibrisan.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import calibration
from sklearn import metrics as sklearn_metrics

from src.config import RESULTS_FILE
from src import plotting

RESULT_COLUMNS = ["model", "skup", "tacnost", "roc_auc", "log_loss", "brier"]


def compute_metrics(y_true, y_proba, threshold=0.5):
    """Racuna tacnost, ROC-AUC, log-loss i Brier skor.

    Vraca recnik sa kljucevima iz RESULT_COLUMNS, bez polja model i skup.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "tacnost": sklearn_metrics.accuracy_score(y_true, y_pred),
        "roc_auc": sklearn_metrics.roc_auc_score(y_true, y_proba),
        "log_loss": sklearn_metrics.log_loss(y_true, y_proba),
        "brier": sklearn_metrics.brier_score_loss(y_true, y_proba),
    }


def append_results(model_name, dataset_name, metrics):
    """Dodaje red u reports/rezultati.csv.

    Ako za isti (model, skup) vec postoji red - npr. posle ponovnog
    izvrsavanja sveske - taj red se zamenjuje, ne duplira.
    """
    row = {"model": model_name, "skup": dataset_name, **metrics}

    if RESULTS_FILE.exists():
        df_results = pd.read_csv(RESULTS_FILE)
        is_same_row = (df_results["model"] == model_name) & (df_results["skup"] == dataset_name)
        df_results = df_results[~is_same_row]
    else:
        df_results = pd.DataFrame(columns=RESULT_COLUMNS)

    df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_results[RESULT_COLUMNS].to_csv(RESULTS_FILE, index=False)


def compute_class_report(y_true, y_pred):
    """Preciznost, odziv i F1 po klasi."""
    return sklearn_metrics.classification_report(
        y_true, y_pred, target_names=["gost", "domacin"], digits=3
    )


def plot_confusion_matrix(y_true, y_pred, title):
    """Crta matricu konfuzije u dogovorenom izgledu."""
    matrix = sklearn_metrics.confusion_matrix(y_true, y_pred)

    plotting.new_figure(title, "Predvidjeno", "Stvarno")
    axes = plt.gca()
    axes.imshow(matrix, cmap="Blues")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes.text(j, i, str(matrix[i, j]), ha="center", va="center")

    axes.set_xticks([0, 1])
    axes.set_xticklabels(["gost", "domacin"])
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["gost", "domacin"])
    plt.show()


def plot_calibration_curve(y_true, y_proba, model_name):
    """Crta kalibracioni dijagram.

    Model koji je tacan 65% vremena treba i da bude siguran oko 65%.
    Odstupanje od dijagonale znaci da su verovatnoce precenjene ili
    potcenjene, sto se ne vidi ni iz tacnosti ni iz ROC-AUC.
    """
    # calibration_curve vraca (prob_true, prob_pred) - u tom redosledu
    prob_true, prob_pred = calibration.calibration_curve(y_true, y_proba, n_bins=10)

    plotting.new_figure(f"Kalibracija - {model_name}", "Predvidjena verovatnoca", "Stvarna verovatnoca")
    plt.plot([0, 1], [0, 1], linestyle="--", color=plotting.COLOR_WARNING, label="Savrsena kalibracija")
    plt.plot(prob_pred, prob_true, marker="o", color=plotting.model_color(model_name), label=model_name)
    plt.legend()
    plt.show()
