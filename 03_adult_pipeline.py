"""Decision Tree dan Random Forest pada dataset adult (UCI) tanpa kebocoran data.

__author__ = "Dr. Eng. Farrikh Alzami, M.Kom"
__email__ =  "alzami@dsn.dinus.ac.id"

Alur.
  1. Muat data, buang kolom dan baris yang tidak dipakai, bagi data latih dan uji.
  2. Semua preprocessing (imputasi, one-hot, scaling) berada di dalam Pipeline,
     sehingga di-fit hanya pada data latih (dan hanya pada lipatan latih saat
     validasi silang).
  3. Decision Tree. Pohon penuh, pohon dangkal, pre-pruning, cost-complexity pruning.
  4. Random Forest. Baseline dengan skor OOB, lalu tuning dengan RandomizedSearchCV.
  5. Pembanding. Logistic Regression.
  6. Evaluasi pada data uji, confusion matrix, MDI dan permutation importance.
  7. Diagnostik tambahan (bagian 11 dan 12).
       a. Ukuran pohon (kedalaman dan jumlah leaf).
       b. Uji hipotesis skor OOB bersifat pesimistis.
       c. Asosiasi antar fitur kategorikal (Cramer V).
       d. Permutation importance gabungan untuk fitur yang saling berasosiasi.
       e. Pembagian data berulang untuk tiga model akhir.

Prasyarat. Python 3.9 atau lebih baru, scikit-learn 1.2 atau lebih baru, pandas,
numpy, matplotlib. Versi library dicetak di awal untuk dicatat.

Cara menjalankan.
  Google Colab.
    1. Unggah 03_adult_pipeline.py dan adult.data (atau pasang Google Drive).
    2. Atur lokasi data lewat variabel lingkungan DATA_PATH, contoh:
         %env DATA_PATH=/content/drive/My Drive/Colab Notebooks/dataset/adult.data
    3. Jalankan: !python 03_adult_pipeline.py
  Terminal.
    DATA_PATH=/lokasi/adult.data python 03_adult_pipeline.py

    DATA_PATH="/Users/alzami/Downloads/hadla buku/kode/adult/adult.data" python 03_adult_pipeline.py


Variabel lingkungan opsional.
  QUICK=1           Grid dan jumlah pohon lebih kecil, untuk uji coba.
  FIGURES_DIR=...   Folder keluaran (default figures_bab7).
  REPEAT_SPLITS=N   Jumlah pembagian data berulang pada bagian 12
                    (default 10, 3 pada mode cepat, 0 untuk melewati).

Skrip mencari data berurutan di DATA_PATH, ./adult.data, lalu mengunduh dari URL UCI.
"""

import itertools
import json
import os
import platform
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
SEED = 42
QUICK = os.environ.get("QUICK", "0") == "1"
FIGURES_DIR = os.environ.get("FIGURES_DIR", "figures_bab7")
REPEAT_SPLITS = int(os.environ.get("REPEAT_SPLITS", "3" if QUICK else "10"))
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
DROP_COLUMNS = ["fnlwgt", "education"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race",
               "sex", "native_country"]
TARGET = "income"
POSITIVE_LABEL = ">50K"
CLASS_NAMES = ["<=50K", ">50K"]

# Kelompok fitur untuk permutation importance gabungan (bagian 11d).
PERMUTATION_GROUPS = [
    ["marital_status"],
    ["relationship"],
    ["marital_status", "relationship"],
    ["sex"],
    ["marital_status", "relationship", "sex"],
]


def log(message=""):
    print(message, flush=True)


def header(text):
    log("\n" + "=" * 72)
    log(text)
    log("=" * 72)


def save_figure(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Gambar disimpan: {path}")


def save_table(table, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    table.to_csv(path)
    log(f"Tabel disimpan: {path}")


def set_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def timed(label, start):
    log(f"[waktu] {label}: {time.time() - start:.1f} detik")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """Muat adult.data dari DATA_PATH, ./adult.data, atau URL UCI."""
    candidates = [p for p in [os.environ.get("DATA_PATH"), "adult.data"] if p]
    source = next((p for p in candidates if os.path.exists(p)), None)
    if source is None:
        source = UCI_URL
        log(f"File lokal tidak ditemukan, mengunduh dari {UCI_URL}")
    log(f"Sumber data: {source}")
    return pd.read_csv(source, header=None, names=COLUMNS, sep=",",
                       skipinitialspace=True, na_values="?")


def prepare_data(raw):
    """Buang kolom dan duplikat, lalu bagi data latih dan uji (stratified)."""
    header("1. DATASET")
    log(f"Ukuran data mentah: {raw.shape[0]} baris, {raw.shape[1]} kolom")
    counts = raw[TARGET].value_counts()
    for label in CLASS_NAMES:
        log(f"  Kelas {label}: {counts[label]} ({counts[label] / len(raw):.4f})")
    missing = raw.isna().sum()
    log(f"Nilai hilang per kolom: {missing[missing > 0].to_dict()}")
    log(f"Baris dengan minimal satu nilai hilang: {int(raw.isna().any(axis=1).sum())}")

    summary = pd.DataFrame({
        "tipe": ["numerik" if c in NUMERIC else "kategorikal" if c in CATEGORICAL
                 else "dibuang" if c in DROP_COLUMNS else "target" for c in COLUMNS],
        "nilai_unik": [raw[c].nunique() for c in COLUMNS],
        "nilai_hilang": [int(raw[c].isna().sum()) for c in COLUMNS],
    }, index=COLUMNS)
    log("\nRingkasan atribut")
    log(summary.to_string())
    save_table(summary, "tabel_ringkasan_atribut.csv")

    # Duplikat diperiksa pada data mentah (15 kolom, termasuk fnlwgt). Jika diperiksa
    # setelah fnlwgt dibuang, ribuan baris dari orang yang berbeda tampak kembar.
    n_dup = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().reset_index(drop=True)
    df = raw.drop(columns=DROP_COLUMNS)
    n_same = int(df.duplicated().sum())
    log(f"\nBaris duplikat penuh dibuang: {n_dup}. Sisa data: {len(df)} baris")
    log(f"Baris dengan fitur identik setelah fnlwgt dan education dibuang: {n_same}. "
        "Baris ini berasal dari orang berbeda, sehingga dipertahankan.")

    X = df[NUMERIC + CATEGORICAL]
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    log(f"Data latih: {len(X_train)} baris (kelas positif {y_train.mean():.4f})")
    log(f"Data uji  : {len(X_test)} baris (kelas positif {y_test.mean():.4f})")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing dan model
# ---------------------------------------------------------------------------
def make_categorical_transformer():
    """Nilai hilang menjadi kategori 'Missing', lalu one-hot.

    Kategori yang frekuensinya di bawah 1 persen digabung menjadi satu kategori
    'infrequent'. Aturan ini dipelajari hanya dari data latih.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=0.01, sparse_output=False)),
    ])


def make_preprocessor(scale_numeric):
    """Preprocessor untuk pohon (tanpa scaling) atau model linear (dengan scaling)."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("cat", make_categorical_transformer(), CATEGORICAL),
    ])


def make_pipeline(model, scale_numeric=False):
    return Pipeline([("prep", make_preprocessor(scale_numeric)), ("model", model)])


def clean_names(names):
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def warn_if_on_grid_edge(best_params, grid):
    """Beri peringatan jika nilai terpilih adalah nilai terkecil atau terbesar pada grid."""
    for name, values in grid.items():
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        chosen = best_params[name]
        if len(numeric) >= 3 and chosen in (min(numeric), max(numeric)):
            log(f"Peringatan: {name} = {chosen} berada di tepi grid {values}. "
                "Perlebar grid dan ulangi.")


def log_tree_size(model, label):
    """Cetak kedalaman dan jumlah leaf satu pohon di dalam pipeline."""
    tree = model.named_steps["model"]
    log(f"{label}: kedalaman = {tree.get_depth()}, jumlah leaf = {tree.get_n_leaves()}")


def log_forest_size(model, label):
    """Cetak ukuran rata-rata pohon penyusun Random Forest."""
    trees = model.named_steps["model"].estimators_
    depths = np.array([t.get_depth() for t in trees])
    leaves = np.array([t.get_n_leaves() for t in trees])
    log(f"{label}: kedalaman pohon rata-rata = {depths.mean():.1f} "
        f"(minimum {depths.min()}, maksimum {depths.max()}); "
        f"jumlah leaf rata-rata = {leaves.mean():.0f}")


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def evaluate(model, X, y):
    """Metrik pada satu himpunan data. Kelas positif adalah '>50K'."""
    pred = model.predict(X)
    score = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred),
        "f1_pos": f1_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, score),
        "avg_precision": average_precision_score(y, score),
    }


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------
def tree_baseline(X_train, y_train, X_test, y_test):
    header("2. DECISION TREE TANPA BATAS (BASELINE)")
    start = time.time()
    model = make_pipeline(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"))
    model.fit(X_train, y_train)
    tree = model.named_steps["model"]
    train_metrics, test_metrics = evaluate(model, X_train, y_train), evaluate(model, X_test, y_test)
    log(f"Kedalaman = {tree.get_depth()}, jumlah leaf = {tree.get_n_leaves()}")
    log(f"Akurasi latih = {train_metrics['accuracy']:.4f}; akurasi uji = {test_metrics['accuracy']:.4f}")
    log(f"ROC-AUC latih = {train_metrics['roc_auc']:.4f}; ROC-AUC uji = {test_metrics['roc_auc']:.4f}")
    timed("baseline", start)
    return model


def plot_shallow_tree(X_train, y_train, X_test, y_test, depth=3,
                      filename="gambar_7_9_pohon_dangkal.png"):
    """Pohon dangkal untuk visualisasi. class_weight tidak dipakai agar jumlah sampel apa adanya."""
    header(f"3. POHON DANGKAL (KEDALAMAN {depth}) UNTUK VISUALISASI")
    model = make_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=SEED))
    model.fit(X_train, y_train)
    names = clean_names(model.named_steps["prep"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(16, 7))
    nodes = plot_tree(model.named_steps["model"], feature_names=names, class_names=CLASS_NAMES,
                      rounded=True, filled=False, impurity=True, fontsize=7, ax=ax)
    # scikit-learn kadang mencetak hitungan bulat sebagai "12592.0" akibat galat
    # pembulatan float. Akhiran ".0" dibuang agar format seragam.
    for node in nodes:
        node.set_text(re.sub(r"(?<=\d)\.0(?=[,\]])", "", node.get_text()))
    save_figure(fig, filename)
    metrics = evaluate(model, X_test, y_test)
    log(f"Akurasi uji pohon kedalaman {depth} = {metrics['accuracy']:.4f}; "
        f"ROC-AUC uji = {metrics['roc_auc']:.4f}")
    return model


def tree_pre_pruning(X_train, y_train, cv):
    header("4. PRE-PRUNING DENGAN GRID SEARCH")
    start = time.time()
    grid = {
        "model__max_depth": [3, 5, 7, 10, 15, None],
        "model__min_samples_leaf": [1, 5, 20, 50, 100, 200, 400],
        "model__criterion": ["gini", "entropy"],
    }
    if QUICK:
        grid = {"model__max_depth": [5, 10], "model__min_samples_leaf": [5, 20],
                "model__criterion": ["gini"]}
    search = GridSearchCV(
        make_pipeline(DecisionTreeClassifier(random_state=SEED, class_weight="balanced")),
        grid, cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log(f"Parameter terbaik: {search.best_params_}")
    log(f"ROC-AUC validasi silang terbaik: {search.best_score_:.4f}")
    top = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)[
        ["param_model__criterion", "param_model__max_depth", "param_model__min_samples_leaf",
         "mean_test_score", "std_test_score"]]
    log("Lima konfigurasi teratas")
    log(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    warn_if_on_grid_edge(search.best_params_, grid)
    log_tree_size(search.best_estimator_, "Pohon pre-pruning terpilih")
    timed("pre-pruning", start)
    return search.best_estimator_, search.best_params_


def tree_post_pruning(X_train, y_train, cv, filename="gambar_7_5_jalur_pruning.png"):
    header("5. COST-COMPLEXITY PRUNING")
    start = time.time()
    prep = make_preprocessor(scale_numeric=False)
    Xt = prep.fit_transform(X_train)
    path = DecisionTreeClassifier(random_state=SEED, class_weight="balanced"
                                  ).cost_complexity_pruning_path(Xt, y_train)
    positive = path.ccp_alphas[path.ccp_alphas > 0]
    log(f"Jumlah alpha pada jalur pruning pohon penuh: {len(path.ccp_alphas)}")
    log(f"Rentang alpha positif: {positive.min():.2e} sampai {positive.max():.2e}")

    alphas = np.unique(np.concatenate([[0.0], np.logspace(-6, -2, 9 if QUICK else 17)]))
    rows = []
    for alpha in alphas:
        model = make_pipeline(DecisionTreeClassifier(
            random_state=SEED, class_weight="balanced", ccp_alpha=alpha))
        res = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc",
                             return_train_score=True, n_jobs=-1)
        full_fit = model.fit(X_train, y_train).named_steps["model"]
        rows.append({
            "ccp_alpha": alpha,
            "auc_train": res["train_score"].mean(),
            "auc_cv_mean": res["test_score"].mean(),
            "auc_cv_std": res["test_score"].std(ddof=1),
            "n_leaves": full_fit.get_n_leaves(),
        })
    table = pd.DataFrame(rows)
    log(table.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    best = table.loc[table["auc_cv_mean"].idxmax()]
    best_alpha = float(best["ccp_alpha"])
    log(f"\nAlpha terbaik (ROC-AUC validasi silang tertinggi): {best_alpha:.3e}, "
        f"leaf = {int(best['n_leaves'])}, AUC CV = {best['auc_cv_mean']:.4f}")
    if best_alpha in (alphas[0], alphas[-1]):
        log("Peringatan: alpha terbaik berada di tepi grid. Perlebar rentang grid.")
    save_table(table.set_index("ccp_alpha"), "hasil_ccp_alpha.csv")

    plot = table[table["ccp_alpha"] > 0]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(plot["ccp_alpha"], plot["auc_train"], color="dimgray", linestyle="--",
            marker="s", markersize=3.5, label="Data latih")
    ax.plot(plot["ccp_alpha"], plot["auc_cv_mean"], color="black", linestyle="-",
            marker="o", markersize=3.5, label="Validasi silang")
    ax.fill_between(plot["ccp_alpha"], plot["auc_cv_mean"] - plot["auc_cv_std"],
                    plot["auc_cv_mean"] + plot["auc_cv_std"], color="gray", alpha=0.25)
    if best_alpha > 0:
        ax.axvline(best_alpha, color="black", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save_figure(fig, filename)

    final = make_pipeline(DecisionTreeClassifier(
        random_state=SEED, class_weight="balanced", ccp_alpha=best_alpha))
    final.fit(X_train, y_train)
    log_tree_size(final, "Pohon cost-complexity terpilih")
    timed("post-pruning", start)
    return final, best_alpha


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def forest_baseline(X_train, y_train, X_test, y_test):
    header("6. RANDOM FOREST BASELINE DAN SKOR OOB")
    start = time.time()
    n_trees = 100 if QUICK else 300
    model = make_pipeline(RandomForestClassifier(
        n_estimators=n_trees, oob_score=True, class_weight="balanced",
        n_jobs=-1, random_state=SEED))
    model.fit(X_train, y_train)
    forest = model.named_steps["model"]
    oob_proba = forest.oob_decision_function_[:, 1]
    valid = ~np.isnan(oob_proba)
    log(f"n_estimators = {n_trees}, max_features = {forest.max_features}")
    log(f"Skor OOB (akurasi)          : {forest.oob_score_:.4f}")
    log(f"ROC-AUC OOB                 : {roc_auc_score(y_train[valid], oob_proba[valid]):.4f}")
    log(f"Baris tanpa pohon OOB       : {int((~valid).sum())}")
    test_metrics = evaluate(model, X_test, y_test)
    log(f"Akurasi uji                 : {test_metrics['accuracy']:.4f}")
    log(f"ROC-AUC uji                 : {test_metrics['roc_auc']:.4f}")
    log(f"Akurasi latih               : {accuracy_score(y_train, model.predict(X_train)):.4f}")
    timed("baseline RF", start)
    return model


def forest_tuning(X_train, y_train):
    header("7. TUNING RANDOM FOREST (RANDOMIZED SEARCH)")
    start = time.time()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    distributions = {
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__max_depth": [None, 10, 15, 20, 30],
    }
    n_iter = 4 if QUICK else 12
    base = make_pipeline(RandomForestClassifier(
        n_estimators=60 if QUICK else 150, class_weight="balanced", n_jobs=-1, random_state=SEED))
    search = RandomizedSearchCV(base, distributions, n_iter=n_iter, cv=cv, scoring="roc_auc",
                                n_jobs=1, random_state=SEED)
    search.fit(X_train, y_train)
    log(f"Parameter terbaik: {search.best_params_}")
    log(f"ROC-AUC validasi silang terbaik: {search.best_score_:.4f}")
    warn_if_on_grid_edge(search.best_params_, distributions)
    results = pd.DataFrame(search.cv_results_)[
        ["param_model__max_features", "param_model__min_samples_leaf",
         "param_model__max_depth", "mean_test_score", "std_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    log(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    final = make_pipeline(RandomForestClassifier(
        n_estimators=100 if QUICK else 500, oob_score=True, class_weight="balanced",
        n_jobs=-1, random_state=SEED, **params))
    final.fit(X_train, y_train)
    log(f"Skor OOB model akhir (akurasi): {final.named_steps['model'].oob_score_:.4f}")
    timed("tuning RF", start)
    return final, params


# ---------------------------------------------------------------------------
# Logistic Regression (pembanding)
# ---------------------------------------------------------------------------
def logistic_baseline(X_train, y_train, cv):
    header("8. LOGISTIC REGRESSION (PEMBANDING)")
    start = time.time()
    grid = {"model__C": [0.01, 0.1, 1, 10] if QUICK else [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    search = GridSearchCV(
        make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000),
                      scale_numeric=True),
        grid, cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log(f"Parameter terbaik: {search.best_params_}")
    log(f"ROC-AUC validasi silang terbaik: {search.best_score_:.4f}")
    scores = pd.DataFrame({"C": grid["model__C"],
                           "AUC CV": search.cv_results_["mean_test_score"],
                           "simpangan baku": search.cv_results_["std_test_score"]})
    log(scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    warn_if_on_grid_edge(search.best_params_, grid)
    timed("logistic", start)
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Perbandingan model
# ---------------------------------------------------------------------------
def compare_models(models, X_train, y_train, X_test, y_test, cv):
    header("9. EVALUASI PADA DATA UJI")
    start = time.time()
    rows, predictions = [], {}
    for name, model in models.items():
        test = evaluate(model, X_test, y_test)
        cv_scores = cross_val_score(clone(model), X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=1 if "Forest" in name else -1)
        rows.append({
            "Model": name,
            "Akurasi latih": accuracy_score(y_train, model.predict(X_train)),
            "Akurasi uji": test["accuracy"],
            "Precision (>50K)": test["precision_pos"],
            "Recall (>50K)": test["recall_pos"],
            "F1 (>50K)": test["f1_pos"],
            "F1 macro": test["f1_macro"],
            "ROC-AUC uji": test["roc_auc"],
            "Avg precision uji": test["avg_precision"],
            "ROC-AUC CV rata-rata": cv_scores.mean(),
            "ROC-AUC CV simpangan baku": cv_scores.std(ddof=1),
        })
        predictions[name] = model.predict(X_test)
    table = pd.DataFrame(rows).set_index("Model")
    log(table.round(4).T.to_string())
    save_table(table, "tabel_perbandingan_model.csv")
    timed("evaluasi", start)
    return table, predictions


def plot_confusion_matrices(predictions, y_test, filename="gambar_7_10_confusion_matrix.png"):
    fig, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.4))
    for ax, (name, pred) in zip(np.atleast_1d(axes), predictions.items()):
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.sum(axis=1).max() * 1.1)
        for i in range(2):
            for j in range(2):
                dark = cm[i, j] > cm.sum(axis=1).max() * 0.55
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if dark else "black", fontsize=11)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Prediksi")
        ax.set_title(name)
    np.atleast_1d(axes)[0].set_ylabel("Aktual")
    fig.tight_layout()
    save_figure(fig, filename)
    for name, pred in predictions.items():
        log(f"Confusion matrix {name} (baris aktual, kolom prediksi): "
            f"{confusion_matrix(y_test, pred).tolist()}")


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------
def aggregate_mdi(model):
    """Jumlahkan MDI kolom one-hot kembali ke fitur asal."""
    names = list(model.named_steps["prep"].get_feature_names_out())
    importances = model.named_steps["model"].feature_importances_
    totals = {}
    for name, value in zip(names, importances):
        origin = name.replace("num__", "", 1)
        if name.startswith("cat__"):
            origin = next(c for c in CATEGORICAL if name.startswith(f"cat__{c}_"))
        totals[origin] = totals.get(origin, 0.0) + value
    return pd.Series(totals).sort_values(ascending=False)


def feature_importance(model, X_test, y_test, filename="gambar_7_11_importance.png", top=10):
    header("10. FEATURE IMPORTANCE")
    start = time.time()
    mdi = aggregate_mdi(model)
    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5 if QUICK else 10, random_state=SEED, n_jobs=1)
    perm_mean = pd.Series(perm.importances_mean, index=X_test.columns)
    perm_std = pd.Series(perm.importances_std, index=X_test.columns)
    order = perm_mean.sort_values(ascending=False).index
    table = pd.DataFrame({
        "MDI": mdi.reindex(X_test.columns),
        "Permutation (penurunan AUC)": perm_mean,
        "Permutation simpangan baku": perm_std,
    }).loc[order]
    log(table.round(4).to_string())
    save_table(table, "tabel_importance.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    top_mdi = mdi.head(top)[::-1]
    axes[0].barh(top_mdi.index, top_mdi.values, color="dimgray")
    axes[0].set_xlabel("MDI (data latih)")
    axes[0].set_title("(a) Mean decrease in impurity")
    top_perm = perm_mean.sort_values(ascending=False).head(top)[::-1]
    axes[1].barh(top_perm.index, top_perm.values, xerr=perm_std[top_perm.index].values,
                 color="gray", ecolor="black")
    axes[1].set_xlabel("Penurunan ROC-AUC (data uji)")
    axes[1].set_title("(b) Permutation importance")
    fig.tight_layout()
    save_figure(fig, filename)
    timed("importance", start)


# ---------------------------------------------------------------------------
# Diagnostik tambahan
# ---------------------------------------------------------------------------
def oob_subset_check(model, n_train, X_test, y_test, label, n_subsets=30):
    """Uji hipotesis bahwa skor OOB sedikit pesimistis.

    Setiap baris latih diprediksi hanya oleh pohon yang tidak memuatnya, yaitu
    sekitar (1 - 1/n)^n, mendekati 0,368, dari seluruh pohon. Di sini akurasi uji
    dihitung ulang memakai subset acak pohon dengan proporsi yang sama. Jika
    akurasi dengan subset lebih rendah daripada dengan semua pohon, ukuran ensemble
    yang lebih kecil menjelaskan sebagian selisih antara skor OOB dan akurasi uji.
    """
    forest = model.named_steps["model"]
    Xt = model.named_steps["prep"].transform(X_test)
    proba = np.stack([tree.predict_proba(Xt) for tree in forest.estimators_])
    n_trees = proba.shape[0]
    y_arr = np.asarray(y_test)

    acc_all = float((proba.mean(axis=0).argmax(axis=1) == y_arr).mean())
    acc_ref = accuracy_score(y_test, model.predict(X_test))
    k = max(1, int(round(n_trees * (1 - 1 / n_train) ** n_train)))
    rng = np.random.default_rng(SEED)
    accs = []
    for _ in range(n_subsets):
        idx = rng.choice(n_trees, size=k, replace=False)
        accs.append((proba[idx].mean(axis=0).argmax(axis=1) == y_arr).mean())
    accs = np.array(accs)
    log(f"{label}")
    log(f"  Akurasi uji, semua {n_trees} pohon  : {acc_all:.4f} (model.predict: {acc_ref:.4f})")
    log(f"  Akurasi uji, {k} pohon acak ({n_subsets} subset): "
        f"{accs.mean():.4f} +/- {accs.std(ddof=1):.4f}")
    log(f"  Skor OOB (akurasi)               : {forest.oob_score_:.4f}")
    log(f"  Selisih uji (semua pohon) dan OOB: {acc_all - forest.oob_score_:+.4f}")
    log(f"  Selisih uji (semua pohon) dan uji (subset): {acc_all - accs.mean():+.4f}")


def cramers_v(a, b):
    """Cramer V antara dua variabel kategorikal (tanpa koreksi bias)."""
    table = pd.crosstab(a, b).to_numpy().astype(float)
    n = table.sum()
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    denom = n * (min(table.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def feature_association(X_train, filename="tabel_asosiasi_fitur.csv", top=8):
    """Asosiasi antar fitur kategorikal pada data latih. Nilai hilang dihitung sebagai kategori."""
    cats = X_train[CATEGORICAL].fillna("Missing")
    rows = [{"fitur_1": a, "fitur_2": b, "cramers_v": cramers_v(cats[a], cats[b])}
            for a, b in itertools.combinations(CATEGORICAL, 2)]
    table = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    log(f"Asosiasi antar fitur kategorikal (Cramer V, {top} pasangan teratas)")
    log(table.head(top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    save_table(table.set_index(["fitur_1", "fitur_2"]), filename)


def grouped_permutation(model, X_test, y_test, groups, n_repeats,
                        filename="tabel_permutation_gabungan.csv"):
    """Permutation importance untuk kelompok fitur.

    Semua kolom dalam satu kelompok diacak dengan urutan baris yang sama, sehingga
    hubungan antar kolom dalam kelompok tetap utuh dan hanya hubungannya dengan
    fitur lain dan dengan target yang dirusak.
    """
    base = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    log(f"ROC-AUC uji dasar = {base:.4f}")
    rng = np.random.default_rng(SEED)
    rows = []
    for cols in groups:
        drops = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X_test))
            permuted = X_test.copy()
            for col in cols:
                permuted[col] = X_test[col].to_numpy()[order]
            drops.append(base - roc_auc_score(y_test, model.predict_proba(permuted)[:, 1]))
        rows.append({
            "kelompok": " + ".join(cols),
            "penurunan AUC (rata-rata)": float(np.mean(drops)),
            "penurunan AUC (simpangan baku)": float(np.std(drops, ddof=1)),
        })
    table = pd.DataFrame(rows).set_index("kelompok")
    log(table.round(4).to_string())
    save_table(table, filename)

    singles = {r["kelompok"]: r["penurunan AUC (rata-rata)"] for r in rows if " + " not in r["kelompok"]}
    for r in rows:
        members = r["kelompok"].split(" + ")
        if len(members) > 1 and all(m in singles for m in members):
            total = sum(singles[m] for m in members)
            log(f"Kelompok {r['kelompok']}: penurunan gabungan = {r['penurunan AUC (rata-rata)']:.4f}; "
                f"jumlah penurunan tunggal = {total:.4f}")
    return table


def final_models(best_alpha, rf_params, logistic_c, n_trees, seed):
    """Tiga model akhir dengan hiperparameter tetap."""
    return {
        "Decision Tree (cost-complexity)": make_pipeline(DecisionTreeClassifier(
            random_state=seed, class_weight="balanced", ccp_alpha=best_alpha)),
        "Random Forest (tuning)": make_pipeline(RandomForestClassifier(
            n_estimators=n_trees, class_weight="balanced", n_jobs=-1,
            random_state=seed, **rf_params)),
        "Logistic Regression": make_pipeline(LogisticRegression(
            class_weight="balanced", max_iter=2000, C=logistic_c), scale_numeric=True),
    }


def repeated_splits(X, y, best_alpha, rf_params, logistic_c, n_splits):
    """Ulangi pembagian latih-uji dengan seed berbeda untuk tiga model akhir.

    Hiperparameter dipilih pada data latih split utama dan tidak dituning ulang.
    Data uji pada split lain dapat beririsan dengan data latih split utama, sehingga
    hasil ini mengukur sebaran akibat pembagian data, bukan prosedur tuning penuh.
    Antar-split tidak independen, jadi tidak ada nilai p yang dilaporkan.
    """
    start = time.time()
    rows = []
    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        for name, model in final_models(best_alpha, rf_params, logistic_c,
                                        100 if QUICK else 300, seed).items():
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_te, y_te)
            rows.append({"split_seed": seed, "Model": name, "roc_auc": metrics["roc_auc"],
                         "avg_precision": metrics["avg_precision"], "f1_pos": metrics["f1_pos"]})
        log(f"  split {seed + 1} dari {n_splits} selesai")
    results = pd.DataFrame(rows)
    save_table(results.set_index(["split_seed", "Model"]), "hasil_split_berulang.csv")

    summary = results.groupby("Model")[["roc_auc", "avg_precision", "f1_pos"]].agg(["mean", "std"])
    log(f"\nRingkasan {n_splits} pembagian data (rata-rata dan simpangan baku)")
    log(summary.round(4).to_string())
    save_table(summary, "ringkasan_split_berulang.csv")

    rf, lr, dt = ("Random Forest (tuning)", "Logistic Regression", "Decision Tree (cost-complexity)")
    log("\nSelisih berpasangan per split (Random Forest dikurangi pembanding)")
    for metric in ["roc_auc", "avg_precision", "f1_pos"]:
        wide = results.pivot(index="split_seed", columns="Model", values=metric)
        for other in (lr, dt):
            diff = wide[rf] - wide[other]
            log(f"  {metric:<14} RF - {other:<32} rata-rata {diff.mean():+.4f}, "
                f"simpangan baku {diff.std(ddof=1):.4f}, "
                f"min {diff.min():+.4f}, maks {diff.max():+.4f}, "
                f"RF lebih tinggi pada {int((diff > 0).sum())} dari {len(diff)} split")
    timed("split berulang", start)


def extra_diagnostics(forest_base, forest_best, tree_pre, X_train, X_test, y_test,
                      full_X, full_y, best_alpha, forest_params, logistic_c):
    header("11. DIAGNOSTIK TAMBAHAN")
    start = time.time()
    log("a. Ukuran pohon")
    log_forest_size(forest_base, "Random Forest baseline")
    log_forest_size(forest_best, "Random Forest tuning")
    log_tree_size(tree_pre, "Pohon pre-pruning")

    log("\nb. Skor OOB dan ukuran ensemble")
    for label, model in [("Random Forest baseline", forest_base),
                         ("Random Forest tuning", forest_best)]:
        oob_subset_check(model, len(X_train), X_test, y_test, label)

    log("\nc. Asosiasi antar fitur")
    feature_association(X_train)

    log("\nd. Permutation importance gabungan (data uji, Random Forest tuning)")
    grouped_permutation(forest_best, X_test, y_test, PERMUTATION_GROUPS,
                        n_repeats=5 if QUICK else 10)
    timed("diagnostik", start)

    if REPEAT_SPLITS > 0:
        header(f"12. PEMBAGIAN DATA BERULANG ({REPEAT_SPLITS} SEED)")
        repeated_splits(full_X, full_y, best_alpha, forest_params, logistic_c, REPEAT_SPLITS)


# ---------------------------------------------------------------------------
def main():
    log(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
        f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")
    log(f"Mode cepat: {QUICK} | seed: {SEED} | jumlah CPU: {os.cpu_count()}")
    total_start = time.time()
    set_style()

    X_train, X_test, y_train, y_test = prepare_data(load_data())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tree_full = tree_baseline(X_train, y_train, X_test, y_test)
    plot_shallow_tree(X_train, y_train, X_test, y_test)
    tree_pre, pre_params = tree_pre_pruning(X_train, y_train, cv)
    tree_post, best_alpha = tree_post_pruning(X_train, y_train, cv)
    forest_base = forest_baseline(X_train, y_train, X_test, y_test)
    forest_best, forest_params = forest_tuning(X_train, y_train)
    logistic = logistic_baseline(X_train, y_train, cv)

    models = {
        "Decision Tree (penuh)": tree_full,
        "Decision Tree (pre-pruning)": tree_pre,
        "Decision Tree (cost-complexity)": tree_post,
        "Random Forest (baseline)": forest_base,
        "Random Forest (tuning)": forest_best,
        "Logistic Regression": logistic,
    }
    _, predictions = compare_models(models, X_train, y_train, X_test, y_test, cv)
    plot_confusion_matrices({
        "Decision Tree (cost-complexity)": predictions["Decision Tree (cost-complexity)"],
        "Random Forest (tuning)": predictions["Random Forest (tuning)"],
        "Logistic Regression": predictions["Logistic Regression"],
    }, y_test)
    feature_importance(forest_best, X_test, y_test)

    logistic_c = logistic.named_steps["model"].C
    extra_diagnostics(forest_base, forest_best, tree_pre, X_train, X_test, y_test,
                      pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
                      best_alpha, forest_params, logistic_c)

    summary = {
        "pre_pruning_params": pre_params,
        "best_ccp_alpha": best_alpha,
        "rf_params": forest_params,
        "logistic_C": logistic_c,
    }
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(os.path.join(FIGURES_DIR, "parameter_terbaik.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    timed("total", total_start)


if __name__ == "__main__":
    main()
