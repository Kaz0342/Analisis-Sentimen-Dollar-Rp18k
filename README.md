# Analisis Sentimen Opini Publik vs Berita (Dollar Rp18.000)

## Deskripsi Proyek
Proyek ini dibuat sebagai tugas akhir mata kuliah Kapita Selekta. Script ini berfungsi untuk melakukan komparasi narasi antara media massa dan opini publik terkait isu nilai tukar Dollar mencapai Rp18.000. 

Pipeline otomatis ini mencakup:
1. **News Scraping:** Mengekstrak artikel berita dan menganalisis kata kunci utama.
2. **Instagram Scraping:** Mengambil komentar publik dari postingan portal berita menggunakan `undetected_chromedriver`.
3. **Sentiment Analysis:** Menganalisis sentimen komentar menggunakan model NLP `IndoBERT` (Berjalan di CPU dengan sistem checkpoint).

## ⚙️ Prasyarat (Requirements)
Pastikan Anda telah menginstal library berikut sebelum menjalankan script:
- `pandas`
- `undetected-chromedriver`
- `selenium`
- `beautifulsoup4`
- `transformers`
- `torch`
- `python-dotenv`

##  Cara Menjalankan Script
1. Clone repository ini ke direktori lokal Anda.
2. Buat file bernama `.env` di folder yang sama dengan script utama. Isi kredensial berikut:
   ```env
   HF_TOKEN=token_huggingface_anda
   IG_USER=username_instagram_anda
   IG_PASS=password_instagram_anda
