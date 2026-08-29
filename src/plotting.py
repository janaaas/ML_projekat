"""Jednoobrazan izgled grafika kroz ceo projekat."""

from matplotlib import pyplot as plt

from src.config import FIGURES_PATH

# Boje za grafike u eksplorativnoj analizi (sveska 01).
COLOR_HOME = "#2a78d6"
COLOR_AWAY = "#eb6834"
COLOR_HIGHLIGHT = "#4a3aa7"
COLOR_WARNING = "#e34948"

FIGURE_SIZE = (8, 6)

# krive ucenja idu u dva podgrafika jedan pored drugog, pa su sire
LEARNING_CURVE_SIZE = (15, 5)

# Redosled i boje modela su fiksirani i koriste se u svim zbirnim grafikama,
# da isti model ima istu boju u svakoj svesci.
MODEL_ORDER = ["M0 domacin", "M1 log. reg.", "M2 XGBoost", "M3 RNN", "M4 LSTM", "M5 GRU"]

MODEL_COLORS = {
    "M0 domacin": "#9e9e9e",
    "M1 log. reg.": "#4c72b0",
    "M2 XGBoost": "#dd8452",
    "M3 RNN": "#937860",
    "M4 LSTM": "#c44e52",
    "M5 GRU": "#55a868",
}


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


def model_color(model_name):
    """Vraca dogovorenu boju modela, sivu ako model nije u spisku."""
    return MODEL_COLORS.get(model_name, "#9e9e9e")


def plot_learning_curves(history, model_name):
    """Crta gubitak i tacnost kroz epohe, u dva podgrafika."""
    epochs = range(1, len(history["train_loss"]) + 1)
    color = model_color(model_name)

    plt.figure(figsize=LEARNING_CURVE_SIZE)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], color=color, label="Train Loss")
    plt.plot(epochs, history["valid_loss"], color=color, linestyle="--", label="Validation Loss")
    plt.title(f"{model_name} - gubitak")
    plt.xlabel("Epoha")
    plt.ylabel("Gubitak")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], color=color, label="Train Accuracy")
    plt.plot(epochs, history["valid_accuracy"], color=color, linestyle="--", label="Validation Accuracy")
    plt.title(f"{model_name} - tacnost")
    plt.xlabel("Epoha")
    plt.ylabel("Tacnost")
    plt.legend()

    plt.tight_layout()
    plt.show()
