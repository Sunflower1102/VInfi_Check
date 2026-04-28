# ⚠️ InFi-Check — Structured Error Dataset Generation Pipeline
# Bước 5: new_reference/*_ref.json -> short_error_dataset/<doc>/<error_type>/<method>.txt

import os
import re
import time
import json
import asyncio
import aiohttp
import random as _r
from openai import OpenAI

# ── Cấu hình đường dẫn ────────────────────────────────────────────────
PROJECT_ROOT     = '/content/drive/MyDrive/Phosphor-Bai-InFi-Check/InFi-Check construct/selected_dataset'
PROMPT_ROOT      = '/content/drive/MyDrive/Phosphor-Bai-InFi-Check/InFi-Check construct/summary_gen_prompt'
BASE_DOC_FOLDER  = '/content/drive/MyDrive/Dataset NLP'

REFERENCE_FOLDER = os.path.join(PROJECT_ROOT, 'new_reference')
ERROR_FOLDER     = os.path.join(PROJECT_ROOT, 'short_error_dataset')
SUMMARY_ROOT     = os.path.join(PROJECT_ROOT, 'new_summary')

os.makedirs(ERROR_FOLDER, exist_ok=True)

# Build doc_name -> category từ new_summary subfolder
DOC_CATEGORY_MAP = {}
for cat in os.listdir(SUMMARY_ROOT):
    cat_path = os.path.join(SUMMARY_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    for f in os.listdir(cat_path):
        if f.endswith('_summary.txt'):
            DOC_CATEGORY_MAP[f.replace('_summary.txt', '')] = cat

ref_files = [f for f in os.listdir(REFERENCE_FOLDER) if f.endswith('.json')]
done_docs = [d for d in os.listdir(ERROR_FOLDER) if os.path.isdir(os.path.join(ERROR_FOLDER, d))]

print(f'📂 Base doc folder   : {BASE_DOC_FOLDER}')
print(f'📂 Reference folder  : {REFERENCE_FOLDER}')
print(f'📂 Error dataset     : {ERROR_FOLDER}')
print(f'📂 Prompt folder     : {PROMPT_ROOT}')
print(f'🗂️  DOC_CATEGORY_MAP  : {len(DOC_CATEGORY_MAP)} entries')
print()
print(f'📄 Reference JSON    : {len(ref_files)}')
print(f'✅ Doc đã có error   : {len(done_docs)}')
print(f'⏳ Doc chưa xử lý   : {len(ref_files) - len(done_docs)}')

# ── Cấu hình API key ──────────────────────────────────────────────────
DEEPSEEK_API_KEY = "your api key"

def _mask(k):
    return f'{k[:8]}...{k[-4:]}' if k and len(k) > 12 else '❌ CHƯA CÓ'

print(f'🔑 DeepSeek API Key: {_mask(DEEPSEEK_API_KEY)}')

if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your api key":
    raise ValueError('❌ Chưa có DEEPSEEK_API_KEY! Hãy điền API key vào biến DEEPSEEK_API_KEY.')
else:
    print('✅ API key đã sẵn sàng.')


# ── Parse instruction templates ───────────────────────────────────────
def parse_txt_to_dict(txt_content: str) -> dict | None:
    """Parse một file instruction .txt thành dict."""
    result = {
        'Error Type': '', 'Method': '',
        'Instruction': '', 'Format': '', 'Few Shot': False
    }

    for field, pattern in [
        ('Error Type', r'Error Type: (.*)'),
        ('Method',     r'Method: (.*)'),
    ]:
        m = re.search(pattern, txt_content)
        if m:
            result[field] = m.group(1).strip()
        else:
            print(f'  [WARN] Không parse được {field}')
            return None

    m = re.search(r'Few Shot: (.*)', txt_content)
    if m:
        fs = m.group(1).strip()
        assert fs in ['Yes', 'No'], f'Few Shot phải là Yes/No, nhận được: {fs}'
        result['Few Shot'] = (fs == 'Yes')
    else:
        print('  [WARN] Không parse được Few Shot')
        return None

    m = re.search(r'Instruction:([\s\S]*?)Format:', txt_content)
    if m:
        result['Instruction'] = m.group(1).strip()
    else:
        print('  [WARN] Không parse được Instruction')
        return None

    if result['Few Shot']:
        for field, pat_start in [
            ('Format',          r'Format:([\s\S]*?)Example Document:'),
            ('Example Document',r'Example Document:([\s\S]*?)Example Summary:'),
            ('Example Summary', r'Example Summary:([\s\S]*?)Example Output:'),
        ]:
            m = re.search(pat_start, txt_content)
            if m:
                result[field] = m.group(1).strip()
            else:
                print(f'  [WARN] Không parse được {field}')
                return None
        m = re.search(r'Example Output:([\s\S]*)', txt_content)
        if m:
            result['Example Output'] = str(eval(m.group(1).strip()))
        else:
            print('  [WARN] Không parse được Example Output')
            return None
    else:
        m = re.search(r'Format:([\s\S]*)', txt_content)
        if m:
            result['Format'] = m.group(1).strip()
        else:
            print('  [WARN] Không parse được Format')
            return None

    return result


def load_instructions(prompt_root: str) -> dict:
    """Đọc toàn bộ file .txt trong prompt_root và parse thành dict."""
    instruction_dict = {}
    for root, dirs, files in os.walk(prompt_root):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            full_path     = os.path.join(root, fname)
            relative_path = os.path.relpath(full_path, prompt_root)

            with open(full_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            parsed = parse_txt_to_dict(txt)
            if not parsed:
                print(f'  [SKIP] Lỗi parse: {relative_path}')
                continue
            parsed['Relative Path'] = relative_path
            key = f"{parsed['Error Type']}|{parsed['Method']}"
            instruction_dict[key] = parsed
    return instruction_dict


INSTRUCTION_DICT = load_instructions(PROMPT_ROOT)

print(f'✅ Đã load {len(INSTRUCTION_DICT)} instruction type:')
for k in sorted(INSTRUCTION_DICT.keys()):
    fs = '(few-shot)' if INSTRUCTION_DICT[k]['Few Shot'] else ''
    print(f'   • {k} {fs}')

# ── DeepSeek client & API functions ───────────────────────────────────
GEN_MODEL    = 'deepseek-chat'
API_BASE_URL = 'https://api.deepseek.com/chat/completions'

client = OpenAI(
    api_key  = DEEPSEEK_API_KEY,
    base_url = 'https://api.deepseek.com'
)


def build_messages(document: str, summary: str, instruction: dict) -> list:
    inst_text = instruction['Instruction']
    fmt_text  = instruction['Format']
    vi_note   = (
        '\nNote: The document and summary are in Vietnamese. '
        'All modifications must be applied to the Vietnamese text and the result must remain in Vietnamese.'
    )
    sys_msg = (
        'You are a helpful assistant. '
        'The document and summary may be written in Vietnamese. '
        'When modifying the summary, keep all Vietnamese text in Vietnamese - do not translate.'
    )
    if instruction['Few Shot']:
        user_intro = (
            f'Here is a document with a summary (the summary is given as a Python list, '
            f'each element is a sentence string). '
            f'Please create a fake summary based on the original summary by the following steps:\n'
            f'{inst_text}{vi_note}\n'
            f'Make sure the new summary is NOT fully supported by the document, '
            f'and do not change any other part of the summary besides those associated with the modification.\n\n'
            f"You should only respond in the format described below. "
            f"Do not return anything else. START YOUR RESPONSE WITH '{{'.\n\n"
            f'Return the result as a Python dictionary with the following keys:\n{fmt_text}\n'
            f"Replace any line breaks in the values with '\\n' so that the dictionary can be parsed using eval()."
        )
        return [
            {'role': 'system',    'content': sys_msg},
            {'role': 'user',      'content': user_intro},
            {'role': 'assistant', 'content': 'Sure! Please give me the document and the summary.'},
            {'role': 'user',      'content': f"Document:\n{instruction['Example Document']}\n\nSummary:\n{instruction['Example Summary']}"},
            {'role': 'assistant', 'content': instruction['Example Output']},
            {'role': 'user',      'content': f'Document:\n{document}\n\nSummary:\n{summary}'},
        ]
    else:
        user_prompt = (
            f'Here is a document with a summary. '
            f'Please create a fake summary based on the original summary by the following steps:\n'
            f'{inst_text}{vi_note}\n'
            f'Make sure the new summary is NOT fully supported by the document, '
            f'and do not change any other part of the summary besides those associated with the modification.\n\n'
            f'Document:\n{document}\n\nSummary:\n{summary}\n\n'
            f"You should only respond in the format described below. "
            f"Do not return anything else. START YOUR RESPONSE WITH '{{'.\n\n"
            f'Return the result as a Python dictionary with the following keys:\n{fmt_text}\n\n'
            f"Make sure the dictionary can be parsed using eval():\n"
            f"Replace any line breaks in the values with '\\n'.\n"
            f'Wrap each string with double quotes ("), replace any double quotes (") inside a string with single quotes (\').'
        )
        return [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user',   'content': user_prompt},
        ]


def parse_answer(answer: str):
    answer = answer.replace('```python', '').replace('```json', '').replace('```', '')
    start = answer.find('{')
    end   = answer.rfind('}') + 1
    assert start >= 0 and end > start
    return eval(answer[start:end])


async def call_api_async(session, messages, semaphore, max_retries=5):
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type':  'application/json',
    }
    payload = {
        'model':       GEN_MODEL,
        'messages':    messages,
        'temperature': 0.4,
        'max_tokens':  2048,
    }
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(API_BASE_URL, headers=headers,
                                        json=payload,
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        print(f'  429 Rate limit - cho {wait}s')
                        await asyncio.sleep(wait)
                        continue
                    data   = await resp.json(content_type=None)
                    answer = data['choices'][0]['message']['content'] or ''
                    return parse_answer(answer)
            except (AssertionError, KeyError, SyntaxError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
    return 'FAIL TO GENERATE DATA'


def call_api(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp   = client.chat.completions.create(
                model=GEN_MODEL, messages=messages,
                temperature=0.4, max_tokens=2048)
            answer = resp.choices[0].message.content or ''
            return parse_answer(answer)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return 'FAIL TO GENERATE DATA'


print(f'✅ DeepSeek async client khởi tạo xong')
print(f'   Model   : {GEN_MODEL}')
print(f'   Endpoint: {API_BASE_URL}')

# ── Domain distribution & cap ─────────────────────────────────────────
DOMAIN_MAP = {
    'Cong_nghe': 'tech',          'So_hoa': 'tech',
    'Kinh_doanh': 'business',     'Kinh_te': 'business',
    'Bat_dong_san': 'business',   'Thi_truong': 'business',
    'Phap_luat': 'legal',         'Chinh_tri': 'politics',
    'Cong_doan': 'politics',
    'Doi_song': 'lifestyle',      'Du_lich': 'lifestyle',
    'Tam_su': 'lifestyle',        'Y_kien': 'lifestyle',
    'Suc_khoe': 'health',         'Y_te': 'health',
    'Giai_tri': 'entertainment',  'Truyen_hinh': 'entertainment',
    'Van_hoa': 'culture',
    'The_thao': 'sports',         'Oto_xe_may': 'sports',
    'The_gioi': 'world',          'Xa_Hoi': 'society',
    'Thoi_su': 'society',         'Chong_Dien_Bien_Hoa_Binh': 'society',
    'Khoa_hoc': 'science',        'Moi_truong': 'science',
    'Giao_duc': 'education',
}


def get_domain(doc_name: str) -> str:
    cat = DOC_CATEGORY_MAP.get(doc_name, '')
    return DOMAIN_MAP.get(cat, 'other')


MAX_DOMAIN_RATIO = 0.30

all_refs    = sorted([f for f in os.listdir(REFERENCE_FOLDER) if f.endswith('.json')])
total       = len(all_refs)
max_per_dom = int(total * MAX_DOMAIN_RATIO)
domain_files = {}
for f in all_refs:
    d = get_domain(f.replace('_ref.json', ''))
    domain_files.setdefault(d, []).append(f)

print(f'Tong: {total} files | Cap: {max_per_dom}/domain ({MAX_DOMAIN_RATIO*100:.0f}%)')
ALLOWED_FILES = set()
for domain, files in sorted(domain_files.items(), key=lambda x: -len(x[1])):
    count = len(files)
    flag  = '  [cap]' if count > max_per_dom else ''
    print(f'  {domain:<15} {count:>4}  ({count/total*100:.1f}%){flag}')
    _r.seed(42); _r.shuffle(files)
    for f in files[:max_per_dom]:
        ALLOWED_FILES.add(f)

print(f'\nSe xu ly: {len(ALLOWED_FILES)}/{total} files')

# ── Chạy toàn bộ dataset (Async) ──────────────────────────────────────
# Thông số cần chỉnh
CONCURRENT  = 7
START_IDX   = 768
END_IDX     = None
LIMIT_DOCS  = None
EXIST_ERROR = ['Paul Warnke']

ref_files_sorted = sorted([
    f for f in os.listdir(REFERENCE_FOLDER)
    if f.endswith('.json') and f in ALLOWED_FILES
])

_start = START_IDX if START_IDX is not None else 0
_end   = END_IDX   if END_IDX   is not None else len(ref_files_sorted)
ref_files_sorted = ref_files_sorted[_start:_end]
if LIMIT_DOCS:
    ref_files_sorted = ref_files_sorted[:LIMIT_DOCS]

print(f"Se xu ly : {len(ref_files_sorted)} documents  (index {_start} -> {_end})")
print(f"Instruction types : {len(INSTRUCTION_DICT)}")
print(f"Concurrent        : {CONCURRENT} luong song song")
print(f"Model             : {GEN_MODEL}")
print(f"Tong API calls uoc tinh : {len(ref_files_sorted) * len(INSTRUCTION_DICT)}\n")

processed_count = 0
skipped_count   = 0
failed_count    = 0


async def process_one_instruction(session, semaphore, doc_name, document,
                                   summary, inst_key, instruction):
    out_dir  = os.path.join(
        ERROR_FOLDER, doc_name,
        instruction['Relative Path'].replace('.txt', '')
    )
    out_file = os.path.join(out_dir, instruction['Method'] + '.txt')

    if os.path.exists(out_file):
        return 'skip'

    messages = build_messages(document, summary, instruction)
    result   = await call_api_async(session, messages, semaphore)

    if not isinstance(result, dict):
        return 'fail'

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(str(result))
    return 'ok'


async def process_one_doc(session, semaphore, ref_fname):
    global processed_count, skipped_count, failed_count

    doc_name = ref_fname.replace('_ref.json', '')
    if doc_name in EXIST_ERROR:
        skipped_count += 1
        return

    category = DOC_CATEGORY_MAP.get(doc_name, '')
    doc_path = os.path.join(BASE_DOC_FOLDER, category, f'{doc_name}.txt') if category else ''
    if not doc_path or not os.path.exists(doc_path):
        skipped_count += 1
        return

    with open(doc_path, 'r', encoding='utf-8') as f:
        document = f.read().replace('"', "'")

    with open(os.path.join(REFERENCE_FOLDER, ref_fname), 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    if ref_data.get('errors'):
        skipped_count += 1
        return

    sents   = ref_data.get('find_support_result', [])
    summary = '[' + ', '.join(
        '"' + s['summary sentence'].replace('"', "'") + '"'
        for s in sents
    ) + ']'

    tasks   = [
        process_one_instruction(session, semaphore, doc_name, document,
                                summary, k, v)
        for k, v in INSTRUCTION_DICT.items()
    ]
    results = await asyncio.gather(*tasks)

    n_ok   = results.count('ok')
    n_fail = results.count('fail')
    n_skip = results.count('skip')

    if n_ok > 0:
        processed_count += 1
        print(f"[{processed_count:>4}] OK {doc_name}  (+{n_ok} new | skip {n_skip} | fail {n_fail})")
    elif n_fail > 0:
        failed_count += n_fail
        print(f"       FAIL {doc_name}  ({n_fail} fail)")
    else:
        skipped_count += 1


async def run_all():
    global processed_count, skipped_count, failed_count
    processed_count = skipped_count = failed_count = 0
    semaphore = asyncio.Semaphore(CONCURRENT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT + 5)
    t0 = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[
            process_one_doc(session, semaphore, ref_fname)
            for ref_fname in ref_files_sorted
        ])
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Xu ly moi : {processed_count}")
    print(f"Bo qua    : {skipped_count}")
    print(f"That bai  : {failed_count}")
    print(f"Thoi gian : {elapsed/60:.1f} phut")


asyncio.run(run_all())

# ── Kiểm tra cấu trúc output ──────────────────────────────────────────
total_error_files = 0
doc_dirs = [d for d in os.listdir(ERROR_FOLDER) if os.path.isdir(os.path.join(ERROR_FOLDER, d))]

for doc_dir in doc_dirs:
    for root, dirs, files in os.walk(os.path.join(ERROR_FOLDER, doc_dir)):
        total_error_files += len([f for f in files if f.endswith('.txt')])

print(f'📁 Số document có error data : {len(doc_dirs)}')
print(f'📄 Tổng số error file sinh ra: {total_error_files}\n')

if doc_dirs:
    sample_doc = _r.choice(doc_dirs)
    sample_files = []
    for root, dirs, files in os.walk(os.path.join(ERROR_FOLDER, sample_doc)):
        for f in files:
            if f.endswith('.txt'):
                sample_files.append(os.path.join(root, f))

    if sample_files:
        sample_file = _r.choice(sample_files)
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = eval(f.read())

        rel = os.path.relpath(sample_file, ERROR_FOLDER)
        print(f'📄 Mẫu ngẫu nhiên: {rel}')
        print(f'   Error Type     : {content.get("error type", content.get("Error Type", "N/A"))}')
        print(f'   Wrong info     : {str(content.get("wrong information", ""))[:120]}')
        print(f'   Modified text  : {str(content.get("modified text", ""))[:120]}')

# ── Kiểm tra document nào còn thiếu error file ────────────────────────
n_inst = len(INSTRUCTION_DICT)
incomplete_docs = []

for ref_fname in sorted(os.listdir(REFERENCE_FOLDER)):
    if not ref_fname.endswith('.json'):
        continue
    doc_name = ref_fname.replace('_ref.json', '')

    done_count = 0
    for inst_key, instruction in INSTRUCTION_DICT.items():
        out_dir  = os.path.join(ERROR_FOLDER, doc_name,
                                instruction['Relative Path'].replace('.txt', ''))
        out_file = os.path.join(out_dir, instruction['Method'] + '.txt')
        if os.path.exists(out_file):
            done_count += 1

    if done_count < n_inst:
        incomplete_docs.append((doc_name, done_count, n_inst))

if not incomplete_docs:
    print(f'✅ Tất cả {len(os.listdir(REFERENCE_FOLDER))} document đã đủ {n_inst} error file!')
else:
    print(f'⚠️  {len(incomplete_docs)} document chưa đủ error file (cần {n_inst} mỗi doc):')
    for doc_name, done, total in incomplete_docs[:20]:
        print(f'   • {doc_name}: {done}/{total}')
    if len(incomplete_docs) > 20:
        print(f'   ... và {len(incomplete_docs)-20} document khác')
    print('\n👉 Chạy lại script để tiếp tục sinh error data còn thiếu.')
