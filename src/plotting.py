"""Jednoobrazan izgled grafika kroz ceo projekat."""

from matplotlib import pyplot as plt

from src.config import FIGURES_PATH

# Boje za grafike u eksplorativnoj analizi (sveska 01).
COLOR_HOME = "#2a78d6"
COLOR_AWAY = "#eb6834"
COLOR_HIGHLIGHT = "#4a3aa7"
COLOR_WARNING = "#e34948"

FIGURE_SIZE = (8, 6)


def new_figure(title, x_label, y_label, figsize=FIGURE_SIZE):
    """Otvara novu figuru sa naslovom i obelezenim osama."""
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def save_figure(file_name, dpi=150):
    """Cuva tekuci grafik u reports/figures/ za upotrebu u prezentaciji."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / file_name, dpi=dpi, bbox_inches="tight")
