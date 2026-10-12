"""Gambar berbasis kode dan eksperimen variansi untuk Bab 7.

__author__ = "Dr. Eng. Farrikh Alzami, M.Kom"
__email__ =  "alzami@dsn.dinus.ac.id"

Keluaran.
  Gambar 7.3  Kurva entropy, Gini, dan misclassification error (dua kelas).
  Gambar 7.8  Batas keputusan pohon tunggal, bagging, dan Random Forest.
  Tabel hasil eksperimen variansi (rata-rata dan simpangan baku dari 30 pengulangan).

Data pada skrip ini sintetis (sklearn.datasets.make_moons) dan hanya dipakai
untuk menunjukkan perilaku model, bukan untuk membandingkan kinerja pada data nyata.

Cara menjalankan di Google Colab.
  Unggah file ini, lalu jalankan: !python 02_ensemble_2d.py
Gambar disimpan di folder FIGURES_DIR.
"""

import os
import platform

import matplotlib

matplotlib.use("Agg")  # backend non-interaktif agar aman di server dan Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

SEED = 42
FIGURES_DIR = os.environ.get("FIGURES_DIR", "figures_bab7")
N_TRAIN = 200
NOISE = 0.30
N_REPEATS = 30
N_TEST = 5000
N_TREES = 100


def set_style():
    """Gaya gambar sederhana, ramah cetak hitam-putih."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def save_figure(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gambar disimpan: {path}")


# ---------------------------------------------------------------------------
# Gambar 7.3. Kurva impurity
# ---------------------------------------------------------------------------
def plot_impurity_curves(filename="gambar_7_3_kurva_impurity.png"):
    p = np.linspace(0.0005, 0.9995, 999)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    gini = 2 * p * (1 - p)
    misclass = 1 - np.maximum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(p, entropy, color="black", linestyle="-", linewidth=1.6, label="Entropy")
    ax.plot(p, gini, color="dimgray", linestyle="--", linewidth=1.6, label="Gini index")
    ax.plot(p, misclass, color="gray", linestyle=":", linewidth=2.0, label="Misclassification error")
    ax.set_xlabel("Proporsi kelas positif (p)")
    ax.set_ylabel("Nilai impurity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, loc="upper center", ncol=3, columnspacing=1.0, handlelength=2.0)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, filename)

    print("Nilai maksimum pada p = 0,5 "
          f"(entropy = {-(0.5 * np.log2(0.5) * 2):.4f}, Gini = {2 * 0.5 * 0.5:.4f}, "
          f"misclassification = {1 - 0.5:.4f})")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_models(seed):
    """Tiga model. Tiap pohon pada model ensemble tumbuh penuh (tanpa pruning)."""
    return {
        "Pohon tunggal": DecisionTreeClassifier(random_state=seed),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=seed),
            n_estimators=N_TREES,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_TREES,
            max_features=1,  # 1 dari 2 fitur pada setiap node
            random_state=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Gambar 7.8. Batas keputusan
# ---------------------------------------------------------------------------
def plot_decision_boundaries(filename="gambar_7_8_batas_keputusan.png"):
    X, y = make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=SEED)
    models = build_models(SEED)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    region_cmap = ListedColormap(["#d9d9d9", "#ffffff"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    labels = ["(a) Pohon tunggal", "(b) Bagging", "(c) Random Forest"]
    for ax, (name, model), label in zip(axes, models.items(), labels):
        model.fit(X, y)
        zz = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=region_cmap)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=0.8)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="black", marker="o", s=14, label="Kelas 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], facecolors="white", edgecolors="black",
                   marker="^", s=20, linewidths=0.8, label="Kelas 1")
        ax.set_title(label)
        ax.set_xlabel("Fitur 1")
        print(f"{name}: akurasi latih = {accuracy_score(y, model.predict(X)):.4f}")
    axes[0].set_ylabel("Fitur 2")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Eksperimen variansi
# ---------------------------------------------------------------------------
def variance_experiment():
    """Latih tiap model pada 30 sampel latih berbeda, uji pada satu himpunan uji tetap.

    Dua ukuran dilaporkan.
      Akurasi uji. Rata-rata dan simpangan baku dari 30 pengulangan.
      Ketidakstabilan prediksi. Rata-rata p(1 - p) di seluruh titik uji, dengan p
      proporsi model (dari 30 pengulangan) yang memprediksi kelas 1 pada titik itu.
      Nilai 0 berarti semua pengulangan memprediksi sama, nilai 0,25 berarti terbelah rata.
    """
    X_test, y_test = make_moons(n_samples=N_TEST, noise=NOISE, random_state=10_000)
    names = list(build_models(SEED).keys())
    accuracies = {n: [] for n in names}
    predictions = {n: [] for n in names}

    for rep in range(N_REPEATS):
        X_tr, y_tr = make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=rep)
        for name, model in build_models(SEED + rep).items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_test)
            accuracies[name].append(accuracy_score(y_test, pred))
            predictions[name].append(pred)

    rows = []
    for name in names:
        acc = np.array(accuracies[name])
        p = np.mean(np.vstack(predictions[name]), axis=0)
        rows.append({
            "Model": name,
            "Akurasi uji (rata-rata)": acc.mean(),
            "Akurasi uji (simpangan baku)": acc.std(ddof=1),
            "Akurasi uji (minimum)": acc.min(),
            "Akurasi uji (maksimum)": acc.max(),
            "Ketidakstabilan prediksi": float(np.mean(p * (1 - p))),
        })
    table = pd.DataFrame(rows).set_index("Model")
    print(f"\nEksperimen variansi. {N_REPEATS} pengulangan, n latih = {N_TRAIN}, "
          f"n uji = {N_TEST}, noise = {NOISE}")
    print(table.round(4).to_string())

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "hasil_eksperimen_variansi.csv")
    table.to_csv(out)
    print(f"Tabel disimpan: {out}")
    return table


def main():
    print(f"Python {platform.python_version()} | numpy {np.__version__} | "
          f"pandas {pd.__version__} | scikit-learn {sklearn.__version__} | "
          f"matplotlib {matplotlib.__version__}")
    set_style()
    plot_impurity_curves()
    plot_decision_boundaries()
    variance_experiment()


if __name__ == "__main__":
    main()
