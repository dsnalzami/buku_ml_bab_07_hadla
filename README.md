# Kode Bab 7. Decision Tree dan Random Forest

Kode pendamping untuk Bab 7 buku **Dasar-Dasar Machine Learning dan Pengenalan Pola**. Seluruh angka, tabel, dan gambar berbasis data pada bab tersebut dihasilkan oleh tiga skrip Python di repositori ini.

Penulis bab. Farrikh Alzami, Fakultas Ilmu Komputer, Universitas Dian Nuswantoro.

## Isi repositori

| Berkas | Fungsi | Data |
| --- | --- | --- |
| `01_hitung_tangan.py` | Verifikasi perhitungan tangan. Entropy, information gain, gain ratio, Gini, pohon ID3, ambang batas numerik, pohon regresi mini, bootstrap, OOB, dan Random Forest mini | Play Tennis (14 baris, tertanam di skrip) |
| `02_ensemble_2d.py` | Kurva impurity, batas keputusan pohon tunggal, bagging, dan Random Forest, serta eksperimen variansi | Sintetis (`make_moons`) |
| `03_adult_pipeline.py` | Pipeline Decision Tree dan Random Forest tanpa kebocoran data, pruning, tuning, evaluasi, importance, dan pembagian data berulang | `adult.data` dari UCI |
| `requirements.txt` | Pustaka yang dibutuhkan | |

## Persiapan

Skrip diuji pada Python 3.13.9, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, dan matplotlib 3.10.6. Parameter `sparse_output` pada `OneHotEncoder` mensyaratkan scikit-learn 1.2 atau lebih baru.

```bash
git clone https://github.com/dsnalzami/buku_ml_bab_07_hadla.git
cd buku_ml_bab_07_hadla
pip install -r requirements.txt
```

## Menjalankan

Skrip 01 dan 02 tidak membutuhkan berkas data.

```bash
python 01_hitung_tangan.py
python 02_ensemble_2d.py
```

Skrip 03 mencari `adult.data` berurutan di variabel lingkungan `DATA_PATH`, lalu di `./adult.data`, lalu mengunduhnya dari UCI Machine Learning Repository bila keduanya tidak ditemukan.

```bash
# macOS dan Linux
DATA_PATH=/lokasi/adult.data python 03_adult_pipeline.py

# Windows PowerShell
$env:DATA_PATH="C:\lokasi\adult.data"; python 03_adult_pipeline.py
```

Dataset dapat diunduh dari https://doi.org/10.24432/C5XW20 (berkas `adult.data`).

### Google Colab

Unggah ketiga skrip ke sesi Colab, lalu jalankan sel berikut.

```python
!python 01_hitung_tangan.py
!python 02_ensemble_2d.py
%env DATA_PATH=/content/adult.data
!python 03_adult_pipeline.py
```

Untuk mengunduh semua keluaran, jalankan `!zip -r figures_bab7.zip figures_bab7` lalu unduh berkas zip tersebut dari panel berkas.

### Variabel lingkungan

| Variabel | Fungsi | Bawaan |
| --- | --- | --- |
| `DATA_PATH` | Lokasi `adult.data` (skrip 03) | tidak diatur |
| `FIGURES_DIR` | Folder keluaran gambar dan tabel | `figures_bab7` |
| `QUICK` | Isi `1` untuk grid dan jumlah pohon yang lebih kecil, hanya untuk uji coba | `0` |
| `REPEAT_SPLITS` | Jumlah pembagian data berulang pada skrip 03. Isi `0` untuk melewati | `10` (`3` pada mode cepat) |

Pada Mac dengan 12 CPU, skrip 03 selesai dalam sekitar 80 detik. Pada satu CPU, waktunya beberapa menit.

## Keluaran dan pemetaan ke bab

Semua keluaran ditulis ke folder `figures_bab7`.

| Keluaran | Skrip | Tempat di bab |
| --- | --- | --- |
| `gambar_7_3_kurva_impurity.png` | 02 | Gambar 7.3 |
| `gambar_7_4_pohon_play_tennis.png` | 01 | Gambar 7.4 |
| `gambar_7_5_jalur_pruning.png` | 03 | Gambar 7.5 |
| `gambar_rf_mini_tiga_pohon.png` | 01 | Gambar 7.7 (di bab bernama `gambar_7_7_rf_mini_tiga_pohon.png`) |
| `gambar_7_8_batas_keputusan.png` | 02 | Gambar 7.8 |
| `gambar_7_9_pohon_dangkal.png` | 03 | Gambar 7.9 |
| `gambar_7_10_confusion_matrix.png` | 03 | Gambar 7.10 |
| `gambar_7_11_importance.png` | 03 | Gambar 7.11 |
| `gambar_pohon_sklearn_tennis.png` | 01 | Pembanding CART scikit-learn pada Play Tennis (tanpa nomor gambar) |
| `hasil_eksperimen_variansi.csv` | 02 | Tabel 7.9 |
| `tabel_ringkasan_atribut.csv` | 03 | Tabel 7.10 |
| `tabel_perbandingan_model.csv` | 03 | Tabel 7.11 |
| `ringkasan_split_berulang.csv`, `hasil_split_berulang.csv` | 03 | Tabel 7.12 |
| `tabel_permutation_gabungan.csv` | 03 | Tabel 7.13 |
| `tabel_importance.csv`, `tabel_asosiasi_fitur.csv`, `hasil_ccp_alpha.csv`, `parameter_terbaik.json` | 03 | Angka pendukung subbab 7.7 dan 7.12 |

Angka perhitungan tangan pada Tabel 7.3, 7.4, 7.5, 7.7, dan 7.8 dicetak ke layar oleh skrip 01 dan tidak disimpan sebagai berkas.

## Ringkasan hasil pada dataset adult

Kelas positif adalah `>50K`. Rata-rata dan simpangan baku dari 10 pembagian data (seed 0 sampai 9), dengan hiperparameter tetap dari split utama.

| Model | ROC-AUC | Average precision | F1 (>50K) |
| --- | --- | --- | --- |
| Decision Tree (cost-complexity pruning) | 0,9061 ± 0,0027 | 0,7591 ± 0,0074 | 0,6842 ± 0,0085 |
| Logistic Regression | 0,9068 ± 0,0032 | 0,7679 ± 0,0064 | 0,6835 ± 0,0051 |
| Random Forest (tuning) | 0,9178 ± 0,0020 | 0,8063 ± 0,0062 | 0,6967 ± 0,0031 |

Pembacaan angka ini dibatasi oleh beberapa hal. Pembagian data berulang tidak independen sehingga tidak ada nilai p, hiperparameter dipilih pada data latih split utama (seed 42), dan split seed 42 sendiri lebih mudah daripada rata-rata 10 split. Pembahasan lengkap ada pada subbab 7.12 bab.

## Reproduksibilitas

- Semua sumber keacakan memakai seed tetap (`42` untuk split utama dan model).
- Model yang dibatasi (pohon hasil pruning, Random Forest hasil tuning, Logistic Regression) menghasilkan angka yang identik sampai empat desimal pada dua versi scikit-learn yang diuji, yaitu 1.7.2 dan 1.8.0.
- Pohon tanpa batas dan Random Forest baseline dapat bergeser sekitar 0,004 antar versi scikit-learn. Selisih ini kemungkinan berasal dari pemilihan pemisahan yang seri, tetapi penyebabnya belum diuji.
- Angka di bab berasal dari lingkungan yang disebut pada bagian Persiapan. Hasil di Google Colab dapat berbeda tipis bila versi pustakanya berbeda.
- Skrip mencetak versi pustaka di awal keluaran untuk dicatat.

## Catatan penggunaan data

Dataset adult berasal dari basis data sensus penduduk Amerika Serikat tahun 1994 dan memuat atribut sensitif (`sex`, `race`, `native_country`). Dataset dan kode di sini ditujukan untuk pembelajaran. Model yang dihasilkan tidak dimaksudkan untuk dipakai dalam keputusan nyata.

## Sitasi dan lisensi

Dataset. Becker, Barry, dan Ronny Kohavi. 1996. *Adult*. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20. Dataset berlisensi Creative Commons Attribution 4.0 International (CC BY 4.0).

Bab. [INSERT: sitasi buku dan penerbit setelah terbit]

Lisensi kode. [INSERT: lisensi kode, misalnya MIT]

