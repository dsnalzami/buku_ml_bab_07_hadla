"""Verifikasi perhitungan tangan Bab 7 (Decision Tree dan Random Forest).

Skrip ini mencetak semua angka yang dipakai pada contoh step-by-step:
  1. Play Tennis. Entropy, information gain, gain ratio, Gini, pohon ID3.
  2. Pencarian threshold fitur numerik (data kecil, ilustratif).
  3. Pohon regresi mini (data kecil, ilustratif).
  4. Bootstrap dan out-of-bag (OOB).
  5. Random Forest mini dengan 3 pohon pada Play Tennis.

Keluaran gambar (matplotlib, folder FIGURES_DIR, default figures_bab7).
  gambar_7_4_pohon_play_tennis.png   Pohon ID3 penuh.
  gambar_pohon_sklearn_tennis.png    Pohon CART scikit-learn (split biner), pembanding.
  gambar_rf_mini_tiga_pohon.png      Tiga pohon Random Forest mini.

Cara menjalankan di Google Colab.
  Unggah file ini, lalu jalankan: !python 01_hitung_tangan.py
Skrip tidak membutuhkan file data eksternal.
"""

import math
import os
import platform
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # backend non-interaktif agar aman di server dan Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

SEED = 42
FIGURES_DIR = os.environ.get("FIGURES_DIR", "figures_bab7")
TARGET = "PlayTennis"
FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]

TENNIS_ROWS = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]

# Hari baru (tidak ada pada data latih) yang diprediksi pada contoh Random Forest mini.
# Hari A memberi suara bulat. Hari B memberi suara terbelah.
NEW_DAYS = {
    "A": {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"},
    "B": {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong"},
}


# ---------------------------------------------------------------------------
# Ukuran impurity
# ---------------------------------------------------------------------------
def entropy(labels):
    """Entropy Shannon (basis 2) dari sekumpulan label."""
    counts = Counter(labels)
    n = sum(counts.values())
    return sum((c / n) * math.log2(n / c) for c in counts.values())


def gini(labels):
    """Gini index dari sekumpulan label."""
    counts = Counter(labels)
    n = sum(counts.values())
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def weighted_impurity(df, attr, measure):
    """Impurity berbobot setelah data dipecah menurut nilai atribut."""
    n = len(df)
    return sum(len(g) / n * measure(g[TARGET]) for _, g in df.groupby(attr, sort=True))


def information_gain(df, attr):
    return entropy(df[TARGET]) - weighted_impurity(df, attr, entropy)


def gini_gain(df, attr):
    return gini(df[TARGET]) - weighted_impurity(df, attr, gini)


def split_info(df, attr):
    n = len(df)
    return sum((len(g) / n) * math.log2(n / len(g)) for _, g in df.groupby(attr, sort=True))


def gain_ratio(df, attr):
    info = split_info(df, attr)
    return information_gain(df, attr) / info if info > 0 else 0.0


def fmt_counts(labels):
    """Format hitungan kelas, misalnya 'No=5, Yes=9'."""
    counts = Counter(labels)
    return ", ".join(f"{k}={counts[k]}" for k in sorted(counts))


# ---------------------------------------------------------------------------
# ID3 sederhana (split multiway, atribut kategorikal)
# ---------------------------------------------------------------------------
def majority_label(labels):
    """Kelas mayoritas. Jika seri, dipilih kelas pertama menurut urutan abjad."""
    counts = Counter(labels)
    return max(sorted(counts), key=counts.get)


def build_tree(df, features, max_depth=None, depth=0, rng=None, m=None):
    """Bangun pohon ID3.

    Jika rng dan m diberikan, setiap node hanya mempertimbangkan m atribut acak
    (perilaku dasar Random Forest). Jika seri pada information gain, dipilih
    atribut pertama menurut urutan abjad.
    """
    labels = df[TARGET]
    counts = Counter(labels)
    node = {"n": len(df), "counts": dict(counts), "label": majority_label(labels), "leaf": True}
    if len(counts) == 1 or not features or (max_depth is not None and depth >= max_depth):
        return node

    candidates = sorted(features)
    if rng is not None and m is not None:
        candidates = sorted(rng.choice(candidates, size=min(m, len(candidates)), replace=False))
    gains = {a: information_gain(df, a) for a in candidates}
    best = max(candidates, key=lambda a: (round(gains[a], 12), -candidates.index(a)))

    node.update({"leaf": False, "attr": best, "gains": gains, "children": {}})
    remaining = [f for f in features if f != best]
    for value, subset in df.groupby(best, sort=True):
        node["children"][value] = build_tree(subset, remaining, max_depth, depth + 1, rng, m)
    return node


def predict_one(node, row):
    """Telusuri pohon. Jika nilai atribut tidak dikenal, pakai mayoritas node."""
    while not node["leaf"]:
        child = node["children"].get(row[node["attr"]])
        if child is None:
            return node["label"]
        node = child
    return node["label"]


def tree_lines(node, indent=0):
    pad = "    " * indent
    if node["leaf"]:
        return [f"{pad}-> {node['label']}  (n={node['n']}; {fmt_counts_dict(node['counts'])})"]
    lines = []
    for value, child in node["children"].items():
        lines.append(f"{pad}{node['attr']} = {value}")
        lines.extend(tree_lines(child, indent + 1))
    return lines


def fmt_counts_dict(counts):
    return ", ".join(f"{k}={counts[k]}" for k in sorted(counts))


def rule_lines(node, conditions=None):
    """Turunkan aturan IF-THEN, satu aturan per leaf."""
    conditions = conditions or []
    if node["leaf"]:
        cond = " AND ".join(conditions) if conditions else "TRUE"
        return [f"IF {cond} THEN {TARGET} = {node['label']}"]
    rules = []
    for value, child in node["children"].items():
        rules.extend(rule_lines(child, conditions + [f"{node['attr']} = {value}"]))
    return rules


# ---------------------------------------------------------------------------
# Gambar pohon (matplotlib)
# ---------------------------------------------------------------------------
def save_figure(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gambar disimpan: {path}")


def tree_positions(root):
    """Posisi (x, y) tiap node. Leaf diberi x berurutan, node internal di tengah anak-anaknya."""
    positions, counter = {}, [0]

    def place(node, depth):
        if node["leaf"]:
            x = float(counter[0])
            counter[0] += 1
        else:
            xs = [place(child, depth + 1) for child in node["children"].values()]
            x = (xs[0] + xs[-1]) / 2
        positions[id(node)] = (x, -float(depth))
        return x

    place(root, 0)
    return positions, counter[0]


def tree_depth(node):
    return 0 if node["leaf"] else 1 + max(tree_depth(c) for c in node["children"].values())


def node_label(node):
    """Teks di dalam node. Node internal memuat atribut dan gain, leaf memuat kelas."""
    stats = f"n = {node['n']}; {fmt_counts_dict(node['counts'])}"
    if node["leaf"]:
        return f"{node['label']}\n{stats}"
    return f"{node['attr']}\nGain = {node['gains'][node['attr']]:.3f}\n{stats}"


def draw_id3_tree(ax, root, fontsize=9, width_leaves=None):
    """Gambar pohon ID3 (split multiway) pada sumbu matplotlib.

    width_leaves menyamakan skala horizontal antar panel (jumlah leaf pohon terlebar).
    """
    positions, n_leaves = tree_positions(root)
    depth = tree_depth(root)

    def draw_edges(node):
        if node["leaf"]:
            return
        x, y = positions[id(node)]
        for value, child in node["children"].items():
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y, cy], color="black", linewidth=0.9, zorder=1)
            ax.text(x + 0.45 * (cx - x), y + 0.45 * (cy - y), str(value), ha="center",
                    va="center", fontsize=fontsize - 1, style="italic", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
            draw_edges(child)

    def draw_nodes(node):
        x, y = positions[id(node)]
        leaf = node["leaf"]
        ax.text(x, y, node_label(node), ha="center", va="center", fontsize=fontsize, zorder=3,
                bbox=dict(boxstyle="square,pad=0.35" if leaf else "round,pad=0.35",
                          facecolor="#e6e6e6" if leaf else "white",
                          edgecolor="black", linewidth=0.9))
        if not leaf:
            for child in node["children"].values():
                draw_nodes(child)

    draw_edges(root)
    draw_nodes(root)
    center = (n_leaves - 1) / 2
    half = ((width_leaves or n_leaves) - 1) / 2 + 0.7
    ax.set_xlim(center - half, center + half)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")


def plot_id3_tree(root, filename):
    """Satu pohon ID3 pada satu gambar."""
    _, n_leaves = tree_positions(root)
    depth = tree_depth(root)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.7 * n_leaves + 1.0), 1.7 * (depth + 1)))
    draw_id3_tree(ax, root, fontsize=9)
    save_figure(fig, filename)


def plot_mini_forest(trees, boots, filename="gambar_rf_mini_tiga_pohon.png"):
    """Tiga pohon Random Forest mini, ditumpuk vertikal dengan skala horizontal sama."""
    leaves = [tree_positions(t)[1] for t in trees]
    widest = max(leaves)
    depth = max(tree_depth(t) for t in trees)
    fig, axes = plt.subplots(len(trees), 1, figsize=(max(6.0, 1.6 * widest + 1.0),
                                                     (1.15 * (depth + 1) + 0.6) * len(trees)))
    for number, (ax, tree, boot) in enumerate(zip(axes, trees, boots), start=1):
        candidates = ", ".join(sorted(tree["gains"])) if not tree["leaf"] else "-"
        ax.set_title(f"({chr(96 + number)}) Pohon {number}. Bootstrap: {fmt_counts(boot[TARGET])}. "
                     f"Kandidat root: {candidates}", fontsize=9, loc="left")
        draw_id3_tree(ax, tree, fontsize=9, width_leaves=widest)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_sklearn_tree(clf, feature_names, filename):
    """Pohon CART scikit-learn (split biner pada fitur one-hot) sebagai pembanding."""
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_tree(clf, feature_names=feature_names, class_names=list(clf.classes_), rounded=True,
              filled=False, impurity=True, fontsize=8, ax=ax)
    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Bagian 1. Play Tennis
# ---------------------------------------------------------------------------
def section_play_tennis(df):
    print_header("1. PLAY TENNIS. ENTROPY, GAIN, GINI, ID3")
    y = df[TARGET]
    print(f"Distribusi kelas: {fmt_counts(y)}")
    print(f"Entropy awal H(S)  = {entropy(y):.4f}")
    print(f"Gini awal Gini(S)  = {gini(y):.4f}")

    summary = []
    for attr in FEATURES:
        print(f"\nAtribut {attr}")
        for value, g in df.groupby(attr, sort=True):
            print(f"  {value:<9} n={len(g):>2}  {fmt_counts(g[TARGET]):<12} "
                  f"entropy={entropy(g[TARGET]):.4f}  gini={gini(g[TARGET]):.4f}")
        summary.append({
            "Atribut": attr,
            "Entropy berbobot": weighted_impurity(df, attr, entropy),
            "Info gain": information_gain(df, attr),
            "Gini gain": gini_gain(df, attr),
            "SplitInfo": split_info(df, attr),
            "Gain ratio": gain_ratio(df, attr),
        })
    table = pd.DataFrame(summary).set_index("Atribut")
    print("\nRingkasan root node (4 desimal)")
    print(table.round(4).to_string())
    print(f"\nAtribut terbaik menurut information gain: {table['Info gain'].idxmax()}")
    print(f"Atribut terbaik menurut Gini gain       : {table['Gini gain'].idxmax()}")
    print(f"Atribut terbaik menurut gain ratio      : {table['Gain ratio'].idxmax()}")
    print(f"Info gain Outlook dengan 3 desimal      : {table.loc['Outlook', 'Info gain']:.3f}")

    tree = build_tree(df, FEATURES)
    print("\nPohon ID3 penuh (split multiway)")
    print("\n".join(tree_lines(tree)))
    print("\nAturan IF-THEN")
    print("\n".join(rule_lines(tree)))
    plot_id3_tree(tree, "gambar_7_4_pohon_play_tennis.png")

    # Pohon level kedua. Gain pada cabang Sunny dan Rain.
    for value in ["Sunny", "Rain"]:
        sub = df[df["Outlook"] == value]
        remaining = [f for f in FEATURES if f != "Outlook"]
        print(f"\nCabang Outlook = {value} (n={len(sub)}; {fmt_counts(sub[TARGET])}), "
              f"H = {entropy(sub[TARGET]):.4f}")
        for attr in remaining:
            print(f"  Gain({attr}) = {information_gain(sub, attr):.4f}")

    # Prediksi hari baru oleh pohon ID3 penuh.
    for name, day in NEW_DAYS.items():
        print(f"\nPrediksi ID3 penuh untuk hari {name} {day}: {predict_one(tree, day)}")

    # Pembanding scikit-learn. Memakai one-hot sehingga split bersifat biner.
    X = pd.get_dummies(df[FEATURES]).astype(int)
    clf = DecisionTreeClassifier(criterion="entropy", random_state=SEED).fit(X, y)
    print("\nPembanding scikit-learn (CART, entropy, one-hot, split biner)")
    print(export_text(clf, feature_names=list(X.columns)))
    plot_sklearn_tree(clf, list(X.columns), "gambar_pohon_sklearn_tennis.png")
    return tree


# ---------------------------------------------------------------------------
# Bagian 2. Threshold numerik
# ---------------------------------------------------------------------------
def section_numeric_threshold():
    print_header("2. THRESHOLD FITUR NUMERIK (DATA ILUSTRATIF)")
    hours = np.array([20, 25, 30, 35, 40, 45, 50, 55])
    label = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    n = len(hours)
    print(f"Jam kerja per minggu: {hours.tolist()}")
    print(f"Label (1 = pendapatan tinggi): {label.tolist()}")
    parent = gini(label.tolist())
    print(f"Gini induk = {parent:.4f}\n")

    rows = []
    for i in range(n - 1):
        thr = (hours[i] + hours[i + 1]) / 2
        left, right = label[: i + 1].tolist(), label[i + 1:].tolist()
        w = len(left) / n * gini(left) + len(right) / n * gini(right)
        rows.append({
            "Threshold": thr,
            "Kiri (0:1)": f"{left.count(0)}:{left.count(1)}",
            "Kanan (0:1)": f"{right.count(0)}:{right.count(1)}",
            "Gini kiri": gini(left),
            "Gini kanan": gini(right),
            "Gini berbobot": w,
            "Gini gain": parent - w,
            "Kelas berubah": label[i] != label[i + 1],
        })
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    best = table.loc[table["Gini berbobot"].idxmin()]
    print(f"\nThreshold terbaik: {best['Threshold']:.1f} dengan Gini berbobot {best['Gini berbobot']:.4f}")
    print(f"Threshold terbaik berada pada batas perubahan kelas: {bool(best['Kelas berubah'])}")


# ---------------------------------------------------------------------------
# Bagian 3. Pohon regresi mini
# ---------------------------------------------------------------------------
def sse(values):
    values = np.asarray(values, dtype=float)
    return float(((values - values.mean()) ** 2).sum()) if len(values) else 0.0


def section_regression_tree():
    print_header("3. POHON REGRESI MINI (DATA ILUSTRATIF)")
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([3.0, 3.5, 4.0, 8.0, 9.0, 8.5])
    parent = sse(y)
    print(f"x = {x.tolist()}")
    print(f"y = {y.tolist()}")
    print(f"Rata-rata y = {y.mean():.4f}; SSE induk = {parent:.4f}\n")

    rows = []
    for i in range(len(x) - 1):
        thr = (x[i] + x[i + 1]) / 2
        left, right = y[: i + 1], y[i + 1:]
        total = sse(left) + sse(right)
        rows.append({
            "Threshold": thr,
            "Mean kiri": left.mean(),
            "Mean kanan": right.mean(),
            "SSE kiri": sse(left),
            "SSE kanan": sse(right),
            "SSE total": total,
            "Penurunan SSE": parent - total,
        })
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    best = table.loc[table["SSE total"].idxmin()]
    print(f"\nThreshold terbaik: {best['Threshold']:.1f}; leaf kiri = {best['Mean kiri']:.4f}; "
          f"leaf kanan = {best['Mean kanan']:.4f}")


# ---------------------------------------------------------------------------
# Bagian 4. Bootstrap dan OOB
# ---------------------------------------------------------------------------
def section_bootstrap(n_trees=3):
    print_header("4. BOOTSTRAP DAN OUT-OF-BAG")
    n = len(TENNIS_ROWS)
    print("Probabilitas satu data tidak terpilih pada bootstrap, (1 - 1/n)^n")
    for size in [14, 100, 1000, 10000]:
        print(f"  n = {size:>5}: {(1 - 1 / size) ** size:.4f}")
    print(f"  limit n -> tak hingga, 1/e = {math.exp(-1):.4f}\n")

    boot_seq, _ = np.random.SeedSequence(SEED).spawn(2)
    rng = np.random.default_rng(boot_seq)
    samples = []
    for t in range(1, n_trees + 1):
        idx = rng.integers(0, n, size=n)
        counts = Counter(idx.tolist())
        oob = sorted(set(range(n)) - set(idx.tolist()))
        samples.append(idx)
        print(f"Pohon {t}")
        print(f"  Indeks bootstrap (hari ke-, 1 sampai 14): {sorted((idx + 1).tolist())}")
        print(f"  Jumlah hari unik terpilih: {len(counts)}")
        print(f"  Data OOB (hari ke-): {[i + 1 for i in oob]}  ({len(oob)} dari {n} = {len(oob) / n:.3f})")
    return samples


# ---------------------------------------------------------------------------
# Bagian 5. Random Forest mini
# ---------------------------------------------------------------------------
def section_mini_forest(df, samples, m=2, max_depth=2):
    print_header("5. RANDOM FOREST MINI (3 POHON, m = 2, KEDALAMAN MAKS 2)")
    n = len(df)
    _, feat_seq = np.random.SeedSequence(SEED).spawn(2)
    rng = np.random.default_rng(feat_seq)

    trees, boots = [], []
    for t, idx in enumerate(samples, start=1):
        boot = df.iloc[idx].reset_index(drop=True)
        tree = build_tree(boot, FEATURES, max_depth=max_depth, rng=rng, m=m)
        trees.append(tree)
        boots.append(boot)
        print(f"\nPohon {t}. Sampel bootstrap: {fmt_counts(boot[TARGET])}, H = {entropy(boot[TARGET]):.4f}")
        if not tree["leaf"]:
            gains = ", ".join(f"Gain({a}) = {g:.4f}" for a, g in sorted(tree["gains"].items()))
            print(f"  Kandidat atribut root: {gains}")
            print(f"  Atribut root terpilih: {tree['attr']}")
        print("\n".join("  " + line for line in tree_lines(tree)))

    plot_mini_forest(trees, boots)

    for name, day in NEW_DAYS.items():
        print(f"\nPrediksi hari baru {name} {day}")
        votes = []
        for t, tree in enumerate(trees, start=1):
            pred = predict_one(tree, day)
            votes.append(pred)
            print(f"  Pohon {t}: {pred}")
        print(f"  Voting: {fmt_counts(votes)} -> {majority_label(votes)}")

    # OOB. Setiap hari diprediksi hanya oleh pohon yang tidak memakainya.
    records = df[FEATURES].to_dict("records")
    actual = df[TARGET].tolist()
    oob_sets = [set(range(n)) - set(idx.tolist()) for idx in samples]
    print("\nPrediksi OOB per hari")
    n_scored = n_wrong = 0
    for i in range(n):
        preds = [predict_one(trees[t], records[i]) for t in range(len(trees)) if i in oob_sets[t]]
        if not preds:
            print(f"  Hari {i + 1:>2}: tanpa pohon OOB")
            continue
        pred = majority_label(preds)
        n_scored += 1
        n_wrong += int(pred != actual[i])
        print(f"  Hari {i + 1:>2}: aktual={actual[i]:<3} OOB={preds} -> {pred}"
              f"{'' if pred == actual[i] else '  SALAH'}")
    print(f"\nHari dengan minimal satu pohon OOB: {n_scored} dari {n}")
    if n_scored:
        print(f"Kesalahan OOB: {n_wrong} dari {n_scored} = {n_wrong / n_scored:.4f}")


# ---------------------------------------------------------------------------
def print_header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main():
    print(f"Python {platform.python_version()} | numpy {np.__version__} | "
          f"pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
    df = pd.DataFrame(TENNIS_ROWS, columns=FEATURES + [TARGET])
    df.insert(0, "Day", range(1, len(df) + 1))
    section_play_tennis(df)
    section_numeric_threshold()
    section_regression_tree()
    samples = section_bootstrap()
    section_mini_forest(df, samples)


if __name__ == "__main__":
    main()
