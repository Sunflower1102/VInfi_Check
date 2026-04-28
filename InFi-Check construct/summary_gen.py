# 📝 InFi-Check — Summary Generation Pipeline (per-folder)
# Bước 3: Dataset NLP/<category>/*.txt -> new_summary/<category>/<file>_summary.txt

import os
import re
import time
import math
import random
from openai import OpenAI, BadRequestError
from requests.exceptions import RequestException

# ── Cấu hình đường dẫn ────────────────────────────────────────────────
BASE_DOC_FOLDER   = '/content/drive/MyDrive/Dataset NLP'
PROJECT_ROOT      = '/content/drive/MyDrive/Phosphor-Bai-InFi-Check/InFi-Check construct/selected_dataset'
BASE_SUMMARY_ROOT = os.path.join(PROJECT_ROOT, 'new_summary')

os.makedirs(BASE_SUMMARY_ROOT, exist_ok=True)

categories = sorted([
    d for d in os.listdir(BASE_DOC_FOLDER)
    if os.path.isdir(os.path.join(BASE_DOC_FOLDER, d))
])
print(f'📂 Doc root    : {BASE_DOC_FOLDER}')
print(f'📂 Summary root: {BASE_SUMMARY_ROOT}')
print(f'\n📋 Danh sách {len(categories)} folder:')
for cat in categories:
    n = len([f for f in os.listdir(os.path.join(BASE_DOC_FOLDER, cat)) if f.endswith('.txt')])
    tag = '  ⚡ se loc' if n > 250 else ''
    print(f'   {cat:<35} {n:>5} bai{tag}')

# ── Cấu hình API key & model ──────────────────────────────────────────
DEEPSEEK_API_KEY = "your api key"

MODEL_NAME = 'deepseek-chat'
API_BASE   = 'https://api.deepseek.com'

print(f'Model  : {MODEL_NAME}')
if DEEPSEEK_API_KEY:
    print(f'API key: {DEEPSEEK_API_KEY[:8]}...{DEEPSEEK_API_KEY[-4:]}')
else:
    print('❌ Chưa có DEEPSEEK_API_KEY!')

# ── Hằng số ───────────────────────────────────────────────────────────
WORD_MIN = 150
WORD_MAX = 700


# ── Hàm lọc chất lượng cho folder lớn ────────────────────────────────
def _ttr(text):
    """Type-Token Ratio: tỉ lệ từ duy nhất / tổng từ"""
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0


def _entity_density(text):
    """Mật độ thực thể đơn giản: đếm token bắt đầu bằng chữ HOA / tổng từ"""
    words = text.split()
    if not words:
        return 0
    named = sum(1 for w in words if w and w[0].isupper() and not w.isupper())
    return named / len(words)


def _bigrams(text):
    """Tập bigram từ của văn bản"""
    words = text.lower().split()
    return set(zip(words, words[1:]))


def _jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _quality_filter(file_paths, limit, word_min=150, word_max=1200, jaccard_thresh=0.5):
    """Lọc và chọn `limit` bài tốt nhất từ danh sách file."""
    scored = []
    for path in file_paths:
        try:
            text = open(path, encoding='utf-8').read()
        except Exception:
            continue
        words = text.split()
        n = len(words)
        if not (word_min <= n <= word_max):
            continue
        score = _ttr(text) * 0.5 + _entity_density(text) * 0.5
        scored.append((score, path, text))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    selected_bigrams = []
    for score, path, text in scored:
        bg = _bigrams(text)
        if any(_jaccard(bg, sb) >= jaccard_thresh for sb in selected_bigrams):
            continue
        selected.append(path)
        selected_bigrams.append(bg)
        if len(selected) >= limit:
            break

    return selected


class SummaryGenerator:
    def __init__(self, api_key, model_name, base_url):
        self.client     = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _build_messages(self, document, doc_words):
        words = document.split()
        if len(words) > 1200:
            document  = ' '.join(words[:1200])
            doc_words = 1200
        max_words = int(min(150, max(60, doc_words / 8)))
        return [
            {'role': 'system', 'content': 'Ban la mot tro ly huu ich, thanh thao tieng Viet.'},
            {'role': 'user', 'content': (
                f'Toi se cung cap cho ban mot bai bao tieng Viet. '
                f'Nhiem vu cua ban la viet tom tat ngan cho bai bao nay theo cac yeu cau sau:\n'
                f'1. Do dai tom tat trong khoang 60 den {max_words} tu.\n'
                f'2. Moi cau trong tom tat phai duoc ho tro truc tiep boi noi dung tai lieu.\n'
                f'3. Doi voi moi su kien, hay giu lai cac thuc the quan trong nhu nguoi, dia diem '
                f'va thoi gian, dac biet la cac thuc the xuat hien song song.\n'
                f'4. Khi don gian hoa, dam bao moi su kien hoac y tuong phuc tap van trung thuc '
                f'voi y nghia goc. Tranh don gian hoa qua muc dan den mau thuan voi tai lieu goc.\n'
                f'5. Viet tom tat bang tieng Viet.\n'
                f'6. Giu nguyen ten rieng, so lieu, ngay thang.\n'
                f'7. Khong dung dai tu thay the (ong, ba, ho, anh...) neu chua de cap doi tuong trong cung cau do.\n\n'
                f'Tai lieu:\n{document}\n\n'
                f'Hay xuat truc tiep tom tat ma khong co bat ky tu thua nao.'
            )}
        ]

    def _call_api(self, messages, doc_words):
        min_words = 30
        for attempt in range(1, 6):
            try:
                resp    = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.3
                )
                answer  = resp.choices[0].message.content
                if 'deepseek-r1' in self.model_name.lower():
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                answer  = answer.strip()
                n_words = len(answer.split())
                print(f'  [attempt {attempt}] {resp.usage.prompt_tokens}p/{resp.usage.completion_tokens}c tok '
                      f'| summary: {n_words} tu (min {min_words})')
                if n_words >= min_words:
                    return answer
                print('  ⚠️  Qua ngan, thu lai...')
            except RequestException as e:
                wait = 2 ** attempt
                print(f'  ❌ Loi mang attempt {attempt}: {e} - cho {wait}s')
                time.sleep(wait)
            except BadRequestError as e:
                print(f'  ❌ BadRequest: {e}')
                return None
            except Exception as e:
                print(f'  ❌ Loi khac attempt {attempt}: {e}')
        return None

    def process_category(self, category, input_folder, output_folder, limit):
        cat_out = os.path.join(output_folder, category)
        os.makedirs(cat_out, exist_ok=True)

        all_files = sorted([
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder) if f.endswith('.txt')
        ])
        total_raw = len(all_files)

        print(f'\n{"="*60}')
        if total_raw > limit:
            print(f'📂 [{category}]  {total_raw} bai > {limit}  ->  loc chat luong, giu {limit} bai tot nhat')
            candidates = _quality_filter(all_files, limit=limit, word_min=150, word_max=1200)
            print(f'   -> {len(candidates)} bai sau loc chat luong + dedup Jaccard bigram')
        else:
            print(f'📂 [{category}]  {total_raw} bai <= {limit}  ->  lay het')
            candidates = [
                p for p in all_files
                if WORD_MIN <= len(open(p, encoding='utf-8').read().split()) <= WORD_MAX
            ]
            print(f'   -> {len(candidates)} bai vao hang doi')

        stats = {'done': 0, 'skipped_exists': 0, 'failed': 0}

        for doc_path in candidates:
            fname    = os.path.basename(doc_path)
            out_path = os.path.join(cat_out, f'{fname[:-4]}_summary.txt')

            if os.path.exists(out_path):
                stats['skipped_exists'] += 1
                continue

            with open(doc_path, 'r', encoding='utf-8') as f:
                document = f.read()
            doc_words = len(document.split())
            print(f'\n  --- {fname} ({doc_words} tu) ---')

            summary = self._call_api(self._build_messages(document, doc_words), doc_words)

            if summary is None:
                print('  ❌ Bo qua')
                stats['failed'] += 1
                continue

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f'  ✅ Luu: {out_path}')
            stats['done'] += 1

        print(f'\n  📊 [{category}] ✅:{stats["done"]} | ⏭️:{stats["skipped_exists"]} | ❌:{stats["failed"]}')
        return stats


# ── Chạy từng folder ──────────────────────────────────────────────────
CATEGORIES_TO_RUN = [
    'Van_hoa',
    'Thi_truong',
    'Moi_truong',
    'So_hoa',
    'Tam_su',
    'Y_kien',
    'Cong_doan',
    'Chong_Dien_Bien_Hoa_Binh', 'Khoa_hoc', 'Du_lich'
]
LIMIT_PER_FOLDER = 200

generator = SummaryGenerator(DEEPSEEK_API_KEY, MODEL_NAME, API_BASE)

run_list = (
    sorted([d for d in os.listdir(BASE_DOC_FOLDER)
            if os.path.isdir(os.path.join(BASE_DOC_FOLDER, d))])
    if CATEGORIES_TO_RUN is None else CATEGORIES_TO_RUN
)
print(f'Se xu ly {len(run_list)} folder | LIMIT_PER_FOLDER = {LIMIT_PER_FOLDER}')

all_stats = {}
for category in run_list:
    cat_input = os.path.join(BASE_DOC_FOLDER, category)
    if not os.path.isdir(cat_input):
        print(f'⚠️  Khong tim thay: {cat_input}')
        continue
    all_stats[category] = generator.process_category(
        category     = category,
        input_folder = cat_input,
        output_folder= BASE_SUMMARY_ROOT,
        limit        = LIMIT_PER_FOLDER,
    )

# ── Tổng kết ──────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'{"Category":<30} {"Moi":>6} {"San":>6} {"Loi":>6}')
print('-'*48)
td = te = tf = 0
for cat, s in all_stats.items():
    print(f'{cat:<30} {s["done"]:>6} {s["skipped_exists"]:>6} {s["failed"]:>6}')
    td += s['done']; te += s['skipped_exists']; tf += s['failed']
print('-'*48)
print(f'{"TONG":<30} {td:>6} {te:>6} {tf:>6}')

# ── Xem trước kết quả ─────────────────────────────────────────────────
print(f'{"Category":<30} {"Summaries":>10}')
print('-' * 45)
grand_total = 0
for cat in sorted(os.listdir(BASE_SUMMARY_ROOT)):
    cat_path = os.path.join(BASE_SUMMARY_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    n = len([f for f in os.listdir(cat_path) if f.endswith('.txt')])
    grand_total += n
    print(f'{cat:<30} {n:>10}')
print('-' * 45)
print(f'{"TONG":<30} {grand_total:>10}')

print('\n-- Mau ngau nhien --')
all_done = [
    os.path.join(BASE_SUMMARY_ROOT, cat, f)
    for cat in os.listdir(BASE_SUMMARY_ROOT)
    if os.path.isdir(os.path.join(BASE_SUMMARY_ROOT, cat))
    for f in os.listdir(os.path.join(BASE_SUMMARY_ROOT, cat))
    if f.endswith('.txt')
]
for fpath in random.sample(all_done, min(3, len(all_done))):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    rel = os.path.relpath(fpath, BASE_SUMMARY_ROOT)
    print(f'\n📄 {rel}')
    print(f'   ({len(content.split())} tu) {content[:250]}...')
