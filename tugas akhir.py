import os
import time
import pandas as pd
import re
import emoji
import nltk
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # noqa
from bs4 import BeautifulSoup
from transformers import pipeline
import requests
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab', quiet=True)

# ==========================================
# 0. KONFIGURASI AWAL
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
IG_USER = os.getenv("IG_USER")
IG_PASS = os.getenv("IG_PASS")

# Taruh link artikel berita soal Dollar Rp18.000 di sini
NEWS_URLS = [
    'https://money.kompas.com/read/2026/06/04/113502926/dollar-hari-ini-tembus-rp-18000-ini-penyebab-rupiah-terpuruk-ke-level-terendah?page=all',
    'https://finance.detik.com/moneter/d-8516100/dolar-as-nyaris-rp-18-000-bi-akhirnya-buka-suara',
    'https://www.malangtimes.com/baca/3331344763/20260603/115900/rupiah-nyaris-sentuh-rp18-000-per-dolar-as-jadi-mata-uang-terlemah-se-asia',
]

# Taruh link postingan IG, WAJIB BANYAK BIAR TEMBUS!
IG_POST_URLS = [
    'https://www.instagram.com/p/DZJWu-1zlTI/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==',
    'https://www.instagram.com/p/DZJcFuDympN/?igsh=MXc1dDczeXQ4NGVnZw==',
]

TARGET_COMMENTS = 5500
CHECKPOINT_EVERY = 50

# Load Model NLP (CPU)
print("[0] Memuat otak IndoBERT...")
device = -1
id_model = pipeline("text-classification", model="crypter70/IndoBERT-Sentiment-Analysis", token=HF_TOKEN, device=device)
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

stopwords = {
    # Kata umum formal
    "dan", "yang", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
    "adalah", "sebagai", "bahwa", "dalam", "oleh", "akan", "bisa", "ada", "tidak",
    "juga", "sudah", "tersebut", "karena", "masih",
    # Kata gaul/informal IG
    "yg", "aja", "kalo", "udah", "nya", "kita", "pake", "jadi", "orang", "sama",
    "mana", "buat", "tapi", "kalau", "kayak", "emang", "banget", "bgt", "gak",
    "nggak", "ngga", "gw", "gue", "lu", "lo", "dia", "mereka", "saja", "lagi",
    "mau", "harus", "punya", "sini", "situ", "kok", "dong", "deh", "lah", "sih",
    "kan", "nih", "tuh", "doang", "terus"
}


# ==========================================
# 1. ANALISIS MEDIA (KATA KUNCI & SENTIMEN)
# ==========================================
def analyze_news(urls):
    print("\n[1] MENGGALI NARASI & SENTIMEN MEDIA...")
    all_text = ""
    for url in urls:
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            body = soup.find('div', class_='detail__body itp_bodycontent_wrapper')
            if body:
                all_text += body.get_text().lower() + " "
        except Exception as e:
            print(f"Gagal mengambil berita: {e}")

    clean_text = re.sub(r'http\S+|www\.\S+', '', all_text).lower()

    text_for_keyword = re.sub(r'[^\w\s]', '', clean_text)
    words = [w for w in text_for_keyword.split() if w not in stopwords and len(w) > 3]

    top_words = Counter(words).most_common(10)

    # Save Kata Kunci Media ke CSV
    df_media_kw = pd.DataFrame(top_words, columns=['Kata_Kunci', 'Frekuensi'])
    df_media_kw.to_csv("kata_kunci_media.csv", index=False)
    print("  [+] Kata kunci media disave ke 'kata_kunci_media.csv'")

    kalimat = sent_tokenize(clean_text)

    if top_words:
        top_3_keywords = [w[0] for w in top_words[:3]]
        with open("ringkasan_penting_artikel.txt", "w", encoding="utf-8") as f:
            f.write(f"=== KALIMAT ARTIKEL TERKAIT KATA KUNCI UTAMA ({', '.join(top_3_keywords)}) ===\n\n")
            count = 1
            for s in kalimat:
                if any(k in s for k in top_3_keywords) and len(s.split()) > 5:
                    f.write(f"{count}. {s.strip().capitalize()}\n")
                    count += 1
        print("  [+] Kalimat penting artikel disave ke 'ringkasan_penting_artikel.txt'")
    # -----------------------------------------------

    chunks = []
    temp = ""
    for s in kalimat:
        s_clean = re.sub(r'[^\w\s]', '', s).strip()
        if len(temp) + len(s_clean) <= 400:
            temp += " " + s_clean
        else:
            chunks.append(temp.strip())
            temp = s_clean
    if temp:
        chunks.append(temp.strip())

    media_sentimen = {'positive': 0, 'negative': 0, 'neutral': 0}

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            res = id_model(chunk)[0]
            conf = res["score"]
            if conf < 0.60:
                sent = "neutral"
            else:
                sent = label_map.get(res["label"], "neutral")
            media_sentimen[sent] += 1
        except Exception as e:
            print(f"  [!] Skip, error: {e}")
            pass

    return top_words, media_sentimen


# ==========================================
# 2. MENGAMBIL KOMENTAR IG (UPDATED HUMANIZED VERSION)
# ==========================================
def scrape_ig_comments(urls, target_amount):
    print("\n[2] MEMULAI OPERASI PENGAMBILAN KOMENTAR & CAPTION IG...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    driver = uc.Chrome(options=options, version_main=148)

    driver.get('https://www.instagram.com/accounts/login/')
    wait = WebDriverWait(driver, 20)
    try:
        inputs = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'input')))
        inputs[0].send_keys(IG_USER)
        inputs[1].send_keys(IG_PASS + Keys.RETURN)
        time.sleep(8)
        print("[+] Login berhasil")
    except Exception as e:
        print(f"[-] Login Gagal: {e}")
        driver.quit()
        return [], []

    all_comments = set()
    all_captions = []
    ui_texts = [
        'balas', 'suka', 'lihat terjemahan', 'hari yang lalu', 'view all',
        'replies', 'likes', 'jam yang lalu', 'menit yang lalu', 'disematkan'
    ]

    for url in urls:
        if len(all_comments) >= target_amount:
            print(f"[+] Target {target_amount} komen terpenuhi. Stop mengambil data.")
            break

        print(f"  -> Mengambil dari: {url}")
        driver.get(url)
        time.sleep(6)

        # --- FITUR TAMBAHAN: Ambil Caption IG Sebelum Scroll ---
        soup_awal = BeautifulSoup(driver.page_source, 'html.parser')
        h1_tag = soup_awal.find('h1')
        if h1_tag:
            caption_text = h1_tag.get_text().strip()
            if len(caption_text.split()) > 5:
                all_captions.append(caption_text)
                print("  [+] Caption postingan berhasil diamankan!")
        else:
            spans = soup_awal.find_all('span', dir='auto')
            for sp in spans[:5]:
                txt = sp.get_text().strip()
                if len(txt.split()) > 10:
                    all_captions.append(txt)
                    print("  [+] Caption postingan (fallback) berhasil didapatkan")
                    break
        # -------------------------------------------------------

        # Mencari DIV scroll komentar secara dinamis dengan multi-selector cadangan
        scroll_div = None
        selectors = [
            (By.CLASS_NAME, 'x5yr21d.xw2csxc.x1odjw0f.x1n2onr6'),
            (By.XPATH, "//div[contains(@style, 'overflow-y: auto')]"),
            (By.XPATH, "//div[contains(@class, 'x168nmei')]")
        ]

        for selector_type, selector_val in selectors:
            try:
                scroll_div = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((selector_type, selector_val))
                )
                if scroll_div:
                    break
            except:
                continue

        if not scroll_div:
            print("  [!] div scroll tidak ditemukan di post ini. Coba fallback body scroll...")
            last_height = driver.execute_script("return document.body.scrollHeight")
            for step in range(400):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(3)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        else:
            print("  [+] Mulai scroll")
            last_height = driver.execute_script("return arguments[0].scrollHeight", scroll_div)
            no_change_count = 0

            for step in range(400):
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div)
                time.sleep(2.5)

                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollTop - 300", scroll_div)
                time.sleep(0.5)

                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div)
                time.sleep(1.5)

                new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_div)

                if new_height == last_height:
                    no_change_count += 1
                    if no_change_count >= 4:
                        print(f"  [!] Scroll berhenti di step ke-{step}. IG mulai susah memberi data.")
                        break
                else:
                    no_change_count = 0
                    last_height = new_height

                soup_temp = BeautifulSoup(driver.page_source, 'html.parser')
                spans_temp = soup_temp.find_all('span', dir='auto')
                current_link_data = 0
                for span in spans_temp:
                    text = span.get_text().strip()
                    if text and len(text.split()) > 2 and not any(ui in text.lower() for ui in ui_texts):
                        current_link_data += 1

                if current_link_data >= 600:
                    print("  [+] Udah dapet ~600 komen dari link ini. Pindah ke link lain agar tidak dicurigai!")
                    break

        # Sesi panen komentar dari link ini
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        spans = soup.find_all('span', dir='auto')

        for span in spans:
            c = span.get_text().strip()
            if not c or len(c.split()) <= 2:
                continue
            if not any(ui in c.lower() for ui in ui_texts):
                all_comments.add(c)

        print(f"  Total akumulasi seluruh komen bersih saat ini: {len(all_comments)}")

    driver.quit()
    return list(all_comments)[:target_amount], all_captions


# ==========================================
# 3. SENTIMEN PUBLIK & REPORT GENERATOR
# ==========================================

# FIX: Guard pie chart kalau data kosong
def safe_pie(ax, values, labels, colors, title):
    if sum(values) == 0:
        ax.text(0.5, 0.5, 'Data Kosong', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(title, fontsize=14, fontweight='bold')


def run_sentiment_and_generate_reports(comments, captions, media_kw, media_sent_counts):
    print(f"\n[3] ANALISIS SENTIMEN {len(comments)} KOMENTAR & BIKIN REPORT...")
    csv_filename = "hasil_sentimen_publik.csv"
    results = []
    semua_kata_publik = []
    publik_sent_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    if not os.path.exists(csv_filename):
        pd.DataFrame(columns=["raw_comment", "comment", "sentiment", "confidence"]).to_csv(csv_filename, index=False)
        start_idx = 0
    else:
        existing_df = pd.read_csv(csv_filename)
        start_idx = len(existing_df)
        print(f"  [!] Melanjutkan dari index ke-{start_idx}...")
        for s in existing_df['sentiment']:
            if s in publik_sent_counts:
                publik_sent_counts[s] += 1

    # Logging waktu eksekusi
    start_time = time.time()

    for i in range(start_idx, len(comments)):
        c_raw = comments[i]
        # PREPROCESSING LENGKAP
        c_clean = c_raw.lower()
        c_clean = emoji.replace_emoji(c_clean, replace='')   # Hapus emoji
        c_clean = re.sub(r'http\S+|www\.\S+', '', c_clean)   # Hapus URL
        c_clean = re.sub(r'[^\w\s]', '', c_clean)            # Hapus simbol
        c_clean = re.sub(r'\s+', ' ', c_clean).strip()       # Normalisasi spasi

        if not c_clean:
            continue

        # Kumpulin kata buat keyword publik
        words = [w for w in c_clean.split() if w not in stopwords and len(w) > 3]
        semua_kata_publik.extend(words)

        try:
            res = id_model(c_clean[:512])[0]
            conf = round(res["score"], 4)
            if conf < 0.60:
                sentiment = "neutral"
            else:
                sentiment = label_map.get(res["label"], res["label"].lower())

            publik_sent_counts[sentiment] += 1
            results.append({
                "raw_comment": c_raw,
                "comment": c_clean,
                "sentiment": sentiment,
                "confidence": conf
            })
        except Exception as e:
            pass

        # Checkpoint + estimasi waktu
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(comments):
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(csv_filename, mode='a', header=False, index=False)
            results = []

            elapsed = time.time() - start_time
            processed = (i + 1) - start_idx
            sisa = len(comments) - (i + 1)
            estimasi = (elapsed / processed) * sisa / 60 if processed > 0 else 0
            print(f"  [{i+1}/{len(comments)}] Checkpoint saved | Estimasi sisa: {estimasi:.1f} menit")

    # --- BIKIN CSV KATA KUNCI PUBLIK (RESTORED!) ---
    top_publik_kw = Counter(semua_kata_publik).most_common(10)
    df_publik_kw = pd.DataFrame(top_publik_kw, columns=['Kata_Kunci', 'Frekuensi'])
    df_publik_kw.to_csv("kata_kunci_publik.csv", index=False)
    print("  [+] Kata kunci publik disave ke 'kata_kunci_publik.csv'")

    # --- EKSTRAK KATA KUNCI DARI CAPTION IG ---
    caption_words = []
    for cap in captions:
        cap_clean = re.sub(r'http\S+|www\.\S+', '', cap.lower())
        cap_clean = re.sub(r'[^\w\s]', '', cap_clean)
        words = [w for w in cap_clean.split() if w not in stopwords and len(w) > 3]
        caption_words.extend(words)
    caption_dict = dict(Counter(caption_words).most_common(20))

    # --- BIKIN CSV KESENJANGAN NARASI ---
    media_dict = dict(media_kw)
    publik_dict = dict(top_publik_kw)
    all_words = set(media_dict) | set(publik_dict) | set(caption_dict)

    df_gap = pd.DataFrame([{
        'Kata_Kunci': w,
        'Freq_Artikel': media_dict.get(w, 0),
        'Freq_Caption_IG': caption_dict.get(w, 0),
        'Freq_Komen_Publik': publik_dict.get(w, 0)
    } for w in all_words]).sort_values(by=['Freq_Artikel', 'Freq_Komen_Publik'], ascending=[False, False])

    df_gap.to_csv("kesenjangan_narasi.csv", index=False)
    print("  [+] File CSV analisis lengkap 3 Arah berhasil dibuat!")

    # --- BIKIN VISUALISASI PIE CHART ---
    labels = ['Positif', 'Negatif', 'Netral']
    colors = ['#66b3ff', '#ff9999', '#99ff99']

    val_media = [
        media_sent_counts.get('positive', 0),
        media_sent_counts.get('negative', 0),
        media_sent_counts.get('neutral', 0)
    ]
    val_publik = [
        publik_sent_counts.get('positive', 0),
        publik_sent_counts.get('negative', 0),
        publik_sent_counts.get('neutral', 0)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # FIX: Pakai safe_pie biar nggak error kalau data kosong
    safe_pie(ax1, val_media, labels, colors, 'Sentimen Media Berita (Dollar Rp18rb)')
    safe_pie(ax2, val_publik, labels, colors, 'Sentimen Opini Publik IG (Dollar Rp18rb)')

    plt.suptitle("Analisis Kesenjangan Sentimen: Media vs Publik", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grafik_perbandingan_sentimen.png')
    plt.show()
    print("  [+] Grafik disimpan ke 'grafik_perbandingan_sentimen.png'")


# ==========================================
# EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    media_keywords, media_sentiments = analyze_news(NEWS_URLS)
    komen_mentah, caption_mentah = scrape_ig_comments(IG_POST_URLS, TARGET_COMMENTS)
    if komen_mentah:
        run_sentiment_and_generate_reports(komen_mentah, caption_mentah, media_keywords, media_sentiments)