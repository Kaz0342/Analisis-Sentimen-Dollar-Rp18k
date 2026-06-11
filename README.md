#  Analisis Kesenjangan Sentimen: Media vs Publik
### Studi Kasus: Dollar AS Tembus Rp18.000

> Proyek ini membandingkan **narasi media berita** dengan **opini publik di Instagram** menggunakan model NLP IndoBERT — untuk melihat apakah framing media selaras dengan apa yang dirasakan masyarakat.

---

##  Cara Kerja

```
Artikel Berita (Kompas, Detik, dll)
    └──► Scrape teks artikel
    └──► Ekstrak kata kunci (Top 10)
    └──► Analisis sentimen per kalimat → IndoBERT

Postingan Instagram
    └──► Login otomatis → Scroll komentar
    └──► Ambil caption postingan
    └──► Preprocessing komentar (lowercase, hapus emoji, simbol, URL)
    └──► Analisis sentimen tiap komentar → IndoBERT

Output
    └──► 5 file CSV + 1 PNG pie chart
```

---

##  Struktur Output

| File | Isi |
|---|---|
| `kata_kunci_media.csv` | Top 10 kata paling sering muncul di artikel berita |
| `kata_kunci_publik.csv` | Top 10 kata paling sering muncul di komentar IG |
| `hasil_sentimen_publik.csv` | Semua komentar + label sentimen + confidence score |
| `kesenjangan_narasi.csv` | Perbandingan frekuensi kata: Artikel vs Caption IG vs Komentar Publik |
| `ringkasan_penting_artikel.txt` | Kalimat-kalimat artikel yang mengandung kata kunci utama |
| `grafik_perbandingan_sentimen.png` | Pie chart sentimen Media vs Publik |

---

## ⚙️ Instalasi

### 1. Clone / download project ini

### 2. Install dependencies
```bash
pip install pandas transformers torch requests beautifulsoup4 selenium undetected-chromedriver matplotlib python-dotenv emoji nltk
```

### 3. Buat file `.env` di folder yang sama
```env
HF_TOKEN=your_huggingface_token_here
IG_USER=your_instagram_username
IG_PASS=your_instagram_password
```

> **Cara dapet HF_TOKEN:** Login ke [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New Token

### 4. Pastikan Google Chrome sudah terinstall
Script ini pakai `undetected-chromedriver` yang butuh Chrome. Versi Chrome harus sesuai dengan `version_main` di kode (default: **148**).

Cek versi Chrome kamu:
```
chrome://settings/help
```
Kalau beda, ubah di kode:
```python
driver = uc.Chrome(options=options, version_main=148)  # Ganti 148 sesuai versi Chrome kamu
```

---

##  Cara Pakai

### 1. Tambah URL artikel berita yang relevan
```python
NEWS_URLS = [
    'https://money.kompas.com/...',
    'https://finance.detik.com/...',
    # tambah URL lain di sini
]
```

>  Saat ini scraper berita hanya support artikel **Kompas** (selector: `detail__body itp_bodycontent_wrapper`). Artikel dari domain lain perlu selector berbeda.

### 2. Tambah URL postingan Instagram
```python
IG_POST_URLS = [
    'https://www.instagram.com/p/xxx/',
    'https://www.instagram.com/p/yyy/',
    # makin banyak makin bagus biar target komentar terpenuhi
]
```

### 3. Set target jumlah komentar
```python
TARGET_COMMENTS = 5500  # Ubah sesuai kebutuhan
```

### 4. Jalankan
```bash
python main.py
```

---

##  Resume Otomatis

Kalau proses tiba-tiba berhenti di tengah jalan (mati lampu, laptop panas, dll), **tidak perlu mulai dari awal**. Script akan otomatis lanjut dari komentar terakhir yang sudah diproses berdasarkan `hasil_sentimen_publik.csv` yang ada.

>  Jangan hapus `hasil_sentimen_publik.csv` kalau mau resume!

---

## Model NLP

| Properti | Detail |
|---|---|
| Model | `crypter70/IndoBERT-Sentiment-Analysis` |
| Sumber | HuggingFace |
| Bahasa | Indonesia |
| Label | `LABEL_0` = Negatif, `LABEL_1` = Positif |
| Threshold Netral | Confidence < 0.60 → diklasifikasikan sebagai Netral |
| Device | CPU (ubah `device = -1` ke `device = 0` kalau punya GPU) |

---

##  Contoh Output `kesenjangan_narasi.csv`

| Kata_Kunci | Freq_Artikel | Freq_Caption_IG | Freq_Komen_Publik |
|---|---|---|---|
| rupiah | 47 | 12 | 3 |
| nilai | 31 | 0 | 8 |
| prabowo | 0 | 5 | 263 |
| ekonomi | 28 | 7 | 41 |

> Dari tabel ini bisa keliatan **gap narasi** — media fokus ke "rupiah" dan "nilai tukar", tapi publik justru lebih banyak nyebut nama tokoh politik.

---

## Catatan Penting

- **Instagram bisa blokir akun** kalau scraping terlalu agresif. Gunakan akun cadangan, bukan akun utama.
- Script ini butuh **koneksi internet stabil** selama proses scraping.
- Estimasi waktu untuk 5.500 komentar di CPU: **2–4 jam** tergantung spesifikasi laptop.
- Kalau kena CAPTCHA atau checkpoint IG saat login, selesaikan secara manual di browser yang muncul, lalu biarkan script lanjut otomatis.

---

##  Dependency Lengkap

```
pandas
transformers
torch
requests
beautifulsoup4
selenium
undetected-chromedriver
matplotlib
python-dotenv
emoji>=2.0.0
nltk
```

---

##  Author

Proyek analisis sentimen untuk keperluan akademik.
Model: IndoBERT | Scraper: Selenium + undetected-chromedriver | NLP Pipeline: HuggingFace Transformers
