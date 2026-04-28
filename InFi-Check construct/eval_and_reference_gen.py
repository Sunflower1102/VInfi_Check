# 🔍 InFi-Check — Eval & Reference Generation Pipeline
# Bước 4: new_summary/*_summary.txt -> new_supported_summary/ + new_reference/

import os
import re
import time
import json
from openai import OpenAI, BadRequestError
from difflib import SequenceMatcher
from requests.exceptions import RequestException
import random

# ── Cấu hình đường dẫn ────────────────────────────────────────────────
PROJECT_ROOT     = '/content/drive/MyDrive/Phosphor-Bai-InFi-Check/InFi-Check construct/selected_dataset'
BASE_DOC_FOLDER  = '/content/drive/MyDrive/Dataset NLP'

SUMMARY_ROOT             = os.path.join(PROJECT_ROOT, 'new_summary')
SUPPORTED_SUMMARY_FOLDER = os.path.join(PROJECT_ROOT, 'new_supported_summary')
REFERENCE_FOLDER         = os.path.join(PROJECT_ROOT, 'new_reference')
PROMPT_FOLDER            = os.path.join(PROJECT_ROOT, '..', 'summary_eval_prompt')

os.makedirs(SUPPORTED_SUMMARY_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER,         exist_ok=True)

all_summary_pairs = []
for cat in sorted(os.listdir(SUMMARY_ROOT)):
    cat_path = os.path.join(SUMMARY_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    for f in sorted(os.listdir(cat_path)):
        if f.endswith('.txt'):
            all_summary_pairs.append((cat, f, os.path.join(cat_path, f)))

done_files = [f for f in os.listdir(SUPPORTED_SUMMARY_FOLDER) if f.endswith('.txt')]

print(f'📂 Summary root       : {SUMMARY_ROOT}')
print(f'📂 Doc root           : {BASE_DOC_FOLDER}')
print(f'📂 Supported summary  : {SUPPORTED_SUMMARY_FOLDER}')
print(f'📂 Reference          : {REFERENCE_FOLDER}')
print(f'📄 Tổng summary       : {len(all_summary_pairs)} files')
print(f'✅ Đã xử lý xong      : {len(done_files)} files')
print(f'⏳ Còn lại            : {len(all_summary_pairs) - len(done_files)} files')

# ── Cấu hình API keys ─────────────────────────────────────────────────
OPENAI_API_KEY = "your api key"

OPENROUTER_KEYS = ["your api key"]  # Thêm key vào list nếu có nhiều key
# Ví dụ: OPENROUTER_KEYS = ["key1", "key2", "key3"]

GROQ_KEYS = ["your api key"]  # Thêm key vào list nếu có nhiều key
# Ví dụ: GROQ_KEYS = ["key1", "key2", ..., "key24"]

def _mask(k):
    return f'{k[:8]}...{k[-4:]}' if k and len(k) > 12 else '❌ CHƯA CÓ'

print(f'OpenAI key      : {_mask(OPENAI_API_KEY)}')
print(f'OpenRouter keys : {len(OPENROUTER_KEYS)}')
for i, ok in enumerate(OPENROUTER_KEYS):
    print(f'  - Key {i+1}: {_mask(ok)}')
print(f'Groq keys found : {len(GROQ_KEYS)}')
for i, gk in enumerate(GROQ_KEYS):
    print(f'  - Key {i+1}: {_mask(gk)}')

# ── Đọc các file prompt ───────────────────────────────────────────────
def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'❌ Không tìm thấy prompt file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

FIND_SUPPORT_PROMPT          = load_prompt('find_support.txt')
FIND_SUPPORT_FORMAT_PROMPT   = load_prompt('find_support_format.txt')
CRITICS_PROMPT               = load_prompt('critics.txt')
CRITICS_FORMAT_PROMPT        = load_prompt('critics_format.txt')
CRITICS_REVISE_PROMPT        = load_prompt('critics_with_revise.txt')
CRITICS_REVISE_FORMAT_PROMPT = load_prompt('critics_with_revise_format.txt')

print('✅ Đã load xong 6 file prompt:')
for name in [
    'find_support.txt', 'find_support_format.txt',
    'critics.txt', 'critics_format.txt',
    'critics_with_revise.txt', 'critics_with_revise_format.txt'
]:
    print(f'   📄 {name}')

# ── Khởi tạo clients ──────────────────────────────────────────────────
openai_client = OpenAI(
    api_key  = OPENAI_API_KEY,
    base_url = 'https://api.openai.com/v1'
)

initial_or_key = OPENROUTER_KEYS[0] if OPENROUTER_KEYS else 'MISSING_KEY'
qwen_client = OpenAI(
    api_key  = initial_or_key,
    base_url = 'https://openrouter.ai/api/v1'
)

initial_groq_key = GROQ_KEYS[0] if GROQ_KEYS else 'MISSING_KEY'
groq_client = OpenAI(
    api_key  = initial_groq_key,
    base_url = 'https://api.groq.com/openai/v1'
)

FIND_SUPPORT_CLIENT = openai_client
FIND_SUPPORT_MODEL  = 'gpt-4o-mini'

MODEL_LIST = [
    {'name': 'gpt-4o-mini',                'client': openai_client, 'revise': True},
    {'name': 'qwen/qwen-2.5-72b-instruct', 'client': qwen_client,   'revise': False},
    {'name': 'llama-3.3-70b-versatile',    'client': groq_client,   'revise': False},
]


# ── Hàm tiện ích ──────────────────────────────────────────────────────
def get_source_description(category: str) -> str:
    return 'Vietnamese news article'


def sent_tokenize_vi(text: str) -> list:
    """Tách câu cho cả tiếng Việt lẫn tiếng Anh."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()


def clean_llm_output(raw: str) -> str:
    """Loại bỏ markdown code fence trước khi eval()."""
    raw = raw.strip()
    for fence in ('```python', '```json', '```'):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith('```'):
        raw = raw[:-3]
    return raw.strip()


print('✅ Clients & hàm tiện ích đã sẵn sàng')


# ── Hàm eval_summary ──────────────────────────────────────────────────
def eval_summary(document: str, summary: str, source_description: str):
    find_support_result = []; eval_result = []; errors = []

    # Bước 1: find_support
    for attempt in range(5):
        try:
            completion = FIND_SUPPORT_CLIENT.chat.completions.create(
                model=FIND_SUPPORT_MODEL,
                messages=[
                    {'role': 'system', 'content': 'Support finder.'},
                    {'role': 'user', 'content': f'{FIND_SUPPORT_PROMPT}\n\nDoc:\n{document}\nSumm:\n{summary}\n\n{FIND_SUPPORT_FORMAT_PROMPT}'}
                ]
            )
            summary_sentences = sent_tokenize_vi(summary)
            find_support_result = eval(clean_llm_output(completion.choices[0].message.content))
            find_support_result = [
                {
                    'document': ' '.join(r['sentences from the document']),
                    'summary sentence': r['summary sentence'],
                    'reference': r['sentences from the document']
                }
                for r in find_support_result
            ]
            break
        except Exception as e:
            print(f'  [find_support attempt {attempt+1}] error: {e}')

    if not find_support_result:
        return 'CANNOT PARSE RESULT', []

    # Bước 2: critics (3 model vote)
    for r in find_support_result:
        revision = ''; supported_num = 0; response_num = 0; revise_count = 0
        while revise_count <= 3:
            for model_cfg in MODEL_LIST:
                retries = 0
                while retries < 5:
                    try:
                        current_client = model_cfg['client']
                        if 'groq' in model_cfg['name'].lower() and GROQ_KEYS:
                            current_client.api_key = GROQ_KEYS[retries % len(GROQ_KEYS)]
                        elif 'qwen' in model_cfg['name'].lower() and OPENROUTER_KEYS:
                            current_client.api_key = OPENROUTER_KEYS[retries % len(OPENROUTER_KEYS)]

                        prompt_text = CRITICS_REVISE_PROMPT if model_cfg['revise'] else CRITICS_PROMPT
                        fmt_text = CRITICS_REVISE_FORMAT_PROMPT if model_cfg['revise'] else CRITICS_FORMAT_PROMPT

                        comp = current_client.chat.completions.create(
                            model=model_cfg['name'],
                            messages=[
                                {'role': 'system', 'content': 'Strict checker.'},
                                {'role': 'user', 'content': f'{prompt_text}\n\nDoc:\n{r["document"]}\nSumm:\n{r["summary sentence"]}\n{fmt_text}'}
                            ]
                        )
                        result = eval(clean_llm_output(comp.choices[0].message.content))
                        response_num += 1
                        if result['support or not'] == 'YES':
                            supported_num += 1
                        elif model_cfg['revise']:
                            revision = result.get('summary sentence that is supported', '')
                        break
                    except Exception as e:
                        err_msg = str(e).lower()
                        if '401' in err_msg or 'api_key' in err_msg:
                            print(f"  ⚠️ Model {model_cfg['name']} rotation attempt {retries+1}")
                        time.sleep(1)
                        retries += 1

            if supported_num > 0.5 * response_num:
                eval_result.append({
                    'summary sentence': r['summary sentence'],
                    'reference': r['reference'],
                    'votes': f'{supported_num}/{response_num}'
                })
                break
            elif revision:
                r['summary sentence'] = revision
                revision = ''; supported_num = 0; response_num = 0
                revise_count += 1
            else:
                revise_count += 1

    return eval_result, errors


# ── Chạy toàn bộ dataset ──────────────────────────────────────────────
LIMIT = None

stats     = {'done': 0, 'skipped': 0, 'error': 0}
processed = 0

print(f'Tổng {len(all_summary_pairs)} file cần xử lý.\n')

for category, summary_name, summary_path in all_summary_pairs:
    if LIMIT is not None and processed >= LIMIT:
        print(f'\n🛑 Đã đạt giới hạn {LIMIT} file.')
        break

    doc_name         = summary_name.replace('_summary.txt', '')
    out_summary_path = os.path.join(
        SUPPORTED_SUMMARY_FOLDER,
        summary_name.replace('_summary', '_supported_summary')
    )
    if os.path.exists(out_summary_path):
        stats['skipped'] += 1
        continue

    doc_file = os.path.join(BASE_DOC_FOLDER, category, f'{doc_name}.txt')
    if not os.path.exists(doc_file):
        print(f'[WARN] Không tìm thấy document: {doc_file}')
        stats['error'] += 1
        continue

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = f.read().replace('"', "'").replace('\u201c', "'").replace('\u201d', "'")

    with open(doc_file, 'r', encoding='utf-8') as f:
        document = f.read().replace('"', "'").replace('\u201c', "'").replace('\u201d', "'")

    _words = document.split()
    if len(_words) > 800:
        document = ' '.join(_words[:800])

    src = get_source_description(category)
    print(f'\n━━━ [{processed+1}] {doc_name}  ({src}) ━━━')

    eval_result, errors = eval_summary(document, summary, src)

    if not isinstance(eval_result, list):
        print(f'  ❌ Lỗi: {eval_result}')
        stats['error'] += 1
        continue

    supported_summary = ' '.join([r['summary sentence'] for r in eval_result])
    reference_result  = {'find_support_result': eval_result, 'errors': errors}

    with open(out_summary_path, 'w', encoding='utf-8') as f:
        f.write(supported_summary)

    ref_path = os.path.join(REFERENCE_FOLDER, f'{doc_name}_ref.json')
    with open(ref_path, 'w', encoding='utf-8') as f:
        json.dump(reference_result, f, indent=4, ensure_ascii=False)

    print(f'  ✅ {len(eval_result)} câu  |  {os.path.basename(out_summary_path)}')
    stats['done'] += 1
    processed += 1

print(f'\n{"="*55}')
print(f'📊 Kết quả:')
print(f'   ✅ Xử lý thành công : {stats["done"]}')
print(f'   ⏭️  Đã có sẵn (skip) : {stats["skipped"]}')
print(f'   ❌ Lỗi              : {stats["error"]}')
remaining = len(all_summary_pairs) - stats['done'] - stats['skipped'] - stats['error']
print(f'   ⏳ Còn lại          : {remaining}')

# ── Xem trước kết quả ─────────────────────────────────────────────────
done = [f for f in os.listdir(SUPPORTED_SUMMARY_FOLDER) if f.endswith('.txt')]
refs = [f for f in os.listdir(REFERENCE_FOLDER)         if f.endswith('.json')]

print(f'Supported summary: {len(done)} files')
print(f'Reference JSON   : {len(refs)} files\n')

for fname in random.sample(done, min(3, len(done))):
    with open(os.path.join(SUPPORTED_SUMMARY_FOLDER, fname), 'r', encoding='utf-8') as f:
        text = f.read()

    doc_name = fname.replace('_supported_summary.txt', '')
    ref_file = os.path.join(REFERENCE_FOLDER, f'{doc_name}_ref.json')
    n_sents  = 'N/A'
    if os.path.exists(ref_file):
        with open(ref_file, 'r', encoding='utf-8') as f:
            ref = json.load(f)
        n_sents = len(ref.get('find_support_result', []))

    print(f'📄 {fname}')
    print(f'   Số câu được xác nhận: {n_sents}')
    print(f'   Preview: {text[:200]}...')
    print()
