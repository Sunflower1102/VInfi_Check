# ============================================================
# InFi-Check — Prepare SFT Dataset (v2)
# Pipeline: pos/neg data → ghép → chia train/valid/test → xuất JSONL
# ============================================================

import os
import re
import json
import ast
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines

random.seed(312)

# ============================================================
# 1. Cấu hình đường dẫn  ✏️ sửa nếu cần
# ============================================================
ROOT_FOLDER  = '/content/drive/MyDrive/Phosphor-Bai-InFi-Check'
DATASET_ROOT = os.path.join(ROOT_FOLDER, 'InFi-Check construct', 'selected_dataset')
PROMPT_ROOT  = os.path.join(ROOT_FOLDER, 'training_dataset_construct')
OUTPUT_ROOT  = os.path.join(ROOT_FOLDER, 'training_dataset_construct')

DOCUMENT_PATH          = os.path.join(DATASET_ROOT, 'Document')
SUMMARY_PATH           = os.path.join(DATASET_ROOT, 'new_summary')
SUPPORTED_SUMMARY_PATH = os.path.join(DATASET_ROOT, 'new_supported_summary')
REFERENCE_PATH         = os.path.join(DATASET_ROOT, 'new_reference')
ERROR_PATH             = os.path.join(DATASET_ROOT, 'short_error_dataset')

SFT_PATH    = os.path.join(OUTPUT_ROOT, 'sft_dataset', 'jsonl')
PROMPT_FILE = os.path.join(PROMPT_ROOT, 'prompt', 'sft_prompt.txt')

os.makedirs(SFT_PATH, exist_ok=True)

# ── Đếm file ──────────────────────────────────────────────
def count_walk(folder, ext='.txt'):
    if not os.path.exists(folder): return 0
    return sum(1 for _, _, files in os.walk(folder) for f in files if f.endswith(ext))

def count_flat(folder, ext='.txt'):
    if not os.path.exists(folder): return 0
    return sum(1 for f in os.listdir(folder) if f.endswith(ext))

def count_error_docs_fast(folder):
    if not os.path.exists(folder): return 0
    return sum(1 for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))

with ThreadPoolExecutor() as ex:
    fd  = ex.submit(count_walk, DOCUMENT_PATH, '.txt')
    fs  = ex.submit(count_walk, SUMMARY_PATH, '.txt')
    fss = ex.submit(count_flat, SUPPORTED_SUMMARY_PATH, '.txt')
    fr  = ex.submit(count_flat, REFERENCE_PATH, '.json')
    fe  = ex.submit(count_error_docs_fast, ERROR_PATH)

print(f'Document/               : {fd.result()} files')
print(f'new_summary/            : {fs.result()} files')
print(f'new_supported_summary/  : {fss.result()} files')
print(f'new_reference/          : {fr.result()} files')
print(f'short_error_dataset/    : {fe.result()} docs')
print(f'Output sft_dataset/     : {SFT_PATH}')
print(f'Prompt file             : {PROMPT_FILE}')

if not os.path.exists(PROMPT_FILE):
    print('Không tìm thấy prompt file → dùng prompt mặc định')
else:
    print('Tất cả thư mục sẵn sàng')

# ============================================================
# 2. Cấu hình
# ============================================================
ERROR_TYPE_DICT = {
    'predicate':             'Predicate Error',
    'entity':                'Entity Error',
    'circumstance':          'Circumstance Error',
    'co-reference':          'Co-reference Error',
    'discourse link':        'Discourse Link Error',
    'extrinsic':             'Extrinsic Error',
    'extrinsic error':       'Extrinsic Error',
    'extrinsic information': 'Extrinsic Error',
}

ERROR_SAMPLE_NUM = {
    'Predicate Error':      2,
    'Entity Error':         2,
    'Circumstance Error':   1,
    'Co-reference Error':   2,
    'Discourse Link Error': 1,
    'Extrinsic Error':      1,
}

NEG_PER_POS    = 1      # pos:neg = 1:1
MIN_CONFIDENCE = 0.67   # parse từ votes string "2/3" → 0.67

DOMAIN_MAP = {
    'Cong_nghe': 'tech',        'So_hoa': 'tech',
    'Kinh_doanh': 'business',   'Kinh_te': 'business',
    'Bat_dong_san': 'business', 'Thi_truong': 'business',
    'Phap_luat': 'legal',       'Chinh_tri': 'politics',
    'Cong_doan': 'politics',
    'Doi_song': 'lifestyle',    'Suc_khoe': 'health',
    'Du_lich': 'lifestyle',     'Am_thuc': 'lifestyle',
    'Tam_su': 'lifestyle',      'Y_kien': 'lifestyle',
    'Giai_tri': 'entertainment','Van_hoa': 'culture',
    'Truyen_hinh': 'entertainment',
    'The_thao': 'sports',       'Oto_xe_may': 'sports',
    'The_gioi': 'world',        'Xa_Hoi': 'society',
    'Thoi_su': 'society',       'Chong_Dien_Bien_Hoa_Binh': 'society',
    'Khoa_hoc': 'science',      'Giao_duc': 'education',
    'Y_te': 'health',           'Moi_truong': 'science',
}
MAX_DOMAIN_RATIO = 1.0

# ── Đọc prompt (fallback về mặc định nếu chưa có file) ────
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        SFT_INPUT_FORMAT = f.read()
    print(f'Đã load prompt SFT ({len(SFT_INPUT_FORMAT)} ký tự)')
else:
    SFT_INPUT_FORMAT = (
        'You are a fact-checking assistant.\n'
        'Given a document and its summary, determine whether each sentence '
        'in the summary is fully supported by the document.\n\n'
    )
    print('Dùng prompt mặc định (chưa có sft_prompt.txt)')

print(f'NEG_PER_POS    : {NEG_PER_POS}  (train ratio 1:{NEG_PER_POS})')
print(f'MIN_CONFIDENCE : {MIN_CONFIDENCE}')

# ============================================================
# 3. Hàm xử lý dữ liệu
# ============================================================
pos_counter = pos_num = 0
neg_counter = neg_num = 0
low_conf_skipped = 0

# sort dài trước để tránh match nhầm prefix ngắn hơn
_ALL_PREFIXES = sorted([k + '_' for k in DOMAIN_MAP.keys()], key=len, reverse=True)

def strip_category_prefix(title: str) -> str:
    for prefix in _ALL_PREFIXES:
        if title.startswith(prefix):
            return title[len(prefix):]
    return title

def get_domain(doc_name: str) -> str:
    for prefix, domain in DOMAIN_MAP.items():
        if doc_name.startswith(prefix + '_'):
            return domain
    return 'other'

def make_sample(input_text: str, output_text: str) -> dict:
    # Qwen format — tránh lỗi tokenizer với tiếng Việt
    return {'text': f'<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>'}

def parse_confidence(item: dict) -> float:
    if 'confidence' in item:
        return float(item['confidence'])
    votes_str = item.get('votes', '')
    if '/' in votes_str:
        try:
            num, den = votes_str.split('/')
            return int(num.strip()) / int(den.strip())
        except Exception:
            pass
    return 1.0

# ── Đọc file an toàn: JSON → ast.literal_eval → nhiều encoding ────────
def safe_read_text(fpath: str) -> str | None:
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    print(f'  [WARN] Không đọc được (encoding lạ): {fpath}')
    return None

def safe_load_error_file(fpath: str) -> dict | None:
    raw = safe_read_text(fpath)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        print(f'  [WARN] Không parse được {fpath}: {e}')
        return None

def safe_load_json(fpath: str) -> dict | None:
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            print(f'  [SKIP] JSON lỗi {fpath}: {e}')
            return None
    print(f'  [SKIP] Không đọc được JSON: {fpath}')
    return None

# ── Sinh output negative ───────────────────────────────────
def generate_negative_output(error_dict: dict) -> str:
    add_mark         = True
    modified_element = error_dict['modified element']
    explanation      = error_dict.get('explanation', '')
    wrong_info       = error_dict.get('wrong information', '')

    if (
        'The meaning has not been altered' in explanation
        or 'No wrong information' in wrong_info
        or 'no wrong information' in wrong_info
    ):
        return ''

    error_type = error_dict['error type']

    if error_type == 'Co-reference Error':
        if 'The subject of the new sentence is' in modified_element:
            modified_element = modified_element.replace('The subject of the new sentence is', '').strip()
        elif 'The new pronoun' in modified_element:
            modified_element = modified_element.replace('The new pronoun', '').strip()

    elif error_type == 'Circumstance Error':
        for pattern in [
            r"The new (\w+?) (?:(['\"].*?['\"])|([^\s]+(?:\s+[^\s]+)*)) used to replace the original \1 .+",
            r"The new circumstance ([^']+|'[^']*') used to replace",
            r"The new (.+?) used to replace",
        ]:
            match = re.match(pattern, modified_element) or re.search(pattern, modified_element)
            if match:
                groups = [g for g in match.groups() if g]
                if groups:
                    modified_element = groups[-1]
                    if 'used to replace' in pattern and 'circumstance' not in pattern:
                        modified_element = modified_element[0].lower() + modified_element[1:]
                        if not modified_element.startswith('the'):
                            modified_element = 'the ' + modified_element
                        add_mark = False
                break

    specific_begin = ' Specifically, '
    if not modified_element:
        if 'remove' in error_dict.get('modification explanation', '').lower():
            specific_begin = ''
        else:
            print(f'  [WARN] Empty modified element: {error_dict.get("modified text", "")[:60]}')
    elif len(modified_element) >= 0.9 * len(error_dict.get('modified text', modified_element)):
        specific_begin   = ''
        modified_element = ''
    else:
        if modified_element.endswith('.'):
            modified_element = modified_element[:-1]
        if add_mark and not (modified_element.startswith("'") and modified_element.endswith("'")):
            modified_element = f"'{modified_element}'"
        modified_element += '.'

    original_text = error_dict.get('original text in summary', '')
    if isinstance(original_text, list):
        original_text = ' '.join(str(x).strip().strip("'\"") for x in original_text if x)
    elif not isinstance(original_text, str):
        original_text = str(original_text)

    method = error_dict.get('method', '')

    if method == 'merging sentences':
        if 'Sentence 1:' in original_text and 'Sentence 2:' in original_text:
            original_text = original_text.replace('Sentence 1:', '').strip()
            original_text = ' '.join(s.strip() for s in original_text.split('Sentence 2:'))
    elif method == 'swapping numbers':
        for marker in ('The sentence', 'Sentence in'):
            if marker in original_text:
                parts = original_text.split(marker)
                if len(parts) == 2:
                    original_text = parts[0]
                break
        if ':' in original_text:
            prefix = original_text[:original_text.find(':')]
            if 'The original' in prefix or 'Original' in prefix:
                original_text = original_text[original_text.find(':'):].strip()

    # Dịch các cụm template tiếng Anh sang tiếng Việt
    def _clean_vi(text: str) -> str:
        replacements = [
            (r'The summary now incorrectly states that', 'Bản tóm tắt nêu sai rằng'),
            (r'The summary incorrectly states that',     'Bản tóm tắt nêu sai rằng'),
            (r'The summary now incorrectly',             'Bản tóm tắt sai khi'),
            (r'The summary incorrectly',                 'Bản tóm tắt sai khi'),
            (r'The summary now states that',             'Bản tóm tắt nêu rằng'),
            (r'The summary states that',                 'Bản tóm tắt nêu rằng'),
            (r'The summary now claims that',             'Bản tóm tắt cho rằng'),
            (r'The summary claims that',                 'Bản tóm tắt cho rằng'),
            (r'The summary now mentions',                'Bản tóm tắt đề cập'),
            (r'The summary mentions',                    'Bản tóm tắt đề cập'),
            (r'The summary now introduces',              'Bản tóm tắt đưa vào'),
            (r'The summary introduces',                  'Bản tóm tắt đưa vào'),
            (r'The summary now',                         'Bản tóm tắt'),
            (r'The summary',                             'Bản tóm tắt'),
            (r'The document clearly states that',        'Tài liệu gốc nêu rõ rằng'),
            (r'The document clearly states',             'Tài liệu gốc nêu rõ'),
            (r'The document only mentions',              'Tài liệu gốc chỉ đề cập'),
            (r'The document only states',                'Tài liệu gốc chỉ nêu'),
            (r'The document only',                       'Tài liệu gốc chỉ'),
            (r'The document mentions',                   'Tài liệu gốc đề cập'),
            (r'The document states',                     'Tài liệu gốc nêu'),
            (r'The document',                            'Tài liệu gốc'),
            (r'The modified summary',                    'Bản tóm tắt'),
            (r'This is incorrect because',               'Điều này sai vì'),
            (r'incorrectly states that',                 'nêu sai rằng'),
            (r'is not supported by the origin document', 'không được hỗ trợ bởi tài liệu gốc'),
            (r'is not supported by the document',        'không được hỗ trợ bởi tài liệu gốc'),
            (r'whereas the document',                    'trong khi tài liệu gốc'),
            (r'while the document',                      'trong khi tài liệu gốc'),
            (r'omitting the fact that',                  'bỏ qua chi tiết rằng'),
            (r'according to the document',               'theo tài liệu gốc'),
            (r'According to the document',               'Theo tài liệu gốc'),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    wrong_info = _clean_vi(wrong_info)

    modified_text = error_dict.get('modified text') or error_dict.get('full text of modified summary', '')
    if isinstance(modified_text, list):
        modified_text = ' '.join(str(x).strip().strip("'\"") for x in modified_text if x)

    return (
        f"The following part is not supported by the origin document:\n"
        f"- Location: Lỗi xuất hiện trong câu: {modified_text}"
        f"{specific_begin}{modified_element}\n"
        f"- Explanation: {wrong_info}\n"
        f"- Correction: {original_text}\n"
        f"- Error Type: {error_type}\n"
        f"Therefore, the answer is YES."
    )

# ── Build index ────────────────────────────────────────────
def _make_key(fname: str) -> str:
    return (fname
        .replace('_supported_summary.txt', '')
        .replace('_summary.txt', '')
        .replace('_ref.json', '')
        .replace('.txt', '')
        .replace('.json', ''))

def build_file_index_walk(folder_path: str, ext: str) -> dict:
    """Dùng cho folder có subfolder domain (Document/, new_summary/)."""
    index = {}
    if not os.path.exists(folder_path): return index
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if not f.endswith(ext): continue
            key = strip_category_prefix(_make_key(f))
            index[key] = os.path.join(root, f)
    return index

def build_file_index_flat(folder_path: str, ext: str) -> dict:
    """Dùng cho folder phẳng (new_supported_summary/, new_reference/)."""
    index = {}
    if not os.path.exists(folder_path): return index
    for f in os.listdir(folder_path):
        if not f.endswith(ext): continue
        key = strip_category_prefix(_make_key(f))
        index[key] = os.path.join(folder_path, f)
    return index

def build_folder_index(folder_path: str) -> dict:
    """os.listdir 1 cấp cho short_error_dataset/."""
    index = {}
    if not os.path.exists(folder_path): return index
    for d in os.listdir(folder_path):
        full = os.path.join(folder_path, d)
        if not os.path.isdir(full): continue
        key_clean = strip_category_prefix(d.lstrip('_'))
        index[key_clean] = full
        index[d]         = full
    return index

# ── Load negative data cho 1 doc ──────────────────────────
def prepare_negative_data(document: str, title: str, error_folder: str) -> list:
    global neg_counter, neg_num
    if not error_folder or not os.path.exists(error_folder):
        return []

    negative_set = []
    for root, dirs, files in os.walk(error_folder):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            fpath         = os.path.join(root, fname)
            relative_path = os.path.relpath(fpath, error_folder).replace('\\', '/')
            path_parts    = relative_path.split('/')

            error_dict = safe_load_error_file(fpath)
            if error_dict is None:
                continue

            raw_key = path_parts[-3].lower() if len(path_parts) >= 3 else ''
            matched = None
            for k, v in ERROR_TYPE_DICT.items():
                if k in raw_key or raw_key in k:
                    matched = v
                    break
            if matched is None:
                print(f'  [WARN] Không nhận ra error type: "{raw_key}" ({relative_path})')
                continue

            error_dict['error type'] = matched
            error_dict['method']     = path_parts[-2] if len(path_parts) >= 2 else 'unknown'

            full_summary = error_dict['full text of modified summary']
            if isinstance(full_summary, list):
                full_summary = ' '.join(str(x).strip().strip("'\"") for x in full_summary if x)
            elif not isinstance(full_summary, str):
                full_summary = str(full_summary)

            input_text  = f"{SFT_INPUT_FORMAT}\nDocument:\n{document}\nSummary:\n{full_summary}"
            output_text = generate_negative_output(error_dict)

            if output_text:
                neg_counter += len(output_text.split())
                neg_num     += 1
                negative_set.append([make_sample(input_text, output_text), matched])

    random.shuffle(negative_set)
    return negative_set

# ── Load positive data cho 1 doc ──────────────────────────
def prepare_positive_data(document: str, summary: str, reference: dict) -> dict | None:
    global pos_counter, pos_num, low_conf_skipped

    input_text   = f"{SFT_INPUT_FORMAT}\nDocument:\n{document}\nSummary:\n{summary}"
    output_lines = []

    for i, item in enumerate(reference.get('find_support_result', [])):
        sent  = item.get('summary sentence', '')
        evids = item.get('reference', item.get('sentences from the document', []))
        conf  = parse_confidence(item)

        if conf < MIN_CONFIDENCE:
            low_conf_skipped += 1
            continue

        if len(evids) == 1:
            output_lines.append(
                f'Sentence {i+1}: "{sent}"\n'
                f'→ Supported by: "{evids[0]}"'
            )
        else:
            joined = '\n  '.join(f'"{e}"' for e in evids)
            output_lines.append(
                f'Sentence {i+1}: "{sent}"\n'
                f'→ Supported by multiple sentences:\n  {joined}'
            )

    if not output_lines:
        return None

    output_text  = '\n\n'.join(output_lines)
    output_text += (
        '\n\nAll sentences in the summary are directly supported '
        'by the original document.\n'
        'Therefore, the answer is NO.'
    )
    pos_counter += len(output_text.split())
    pos_num     += 1
    return make_sample(input_text, output_text)

# ── Xử lý 1 doc (chạy trong ThreadPool) ───────────────────
def process_one_doc(args):
    title, summary_filepath, doc_file, ref_file, error_folder = args
    try:
        summary  = safe_read_text(summary_filepath) or ''
        document = safe_read_text(doc_file)         or ''

        neg_data   = prepare_negative_data(document, title, error_folder)
        pos_sample = None

        if ref_file:
            reference = safe_load_json(ref_file)
            if reference is not None:
                pos_sample = prepare_positive_data(document, summary, reference)

        if pos_sample is not None:
            return title, {'positive': pos_sample, 'negative': neg_data}
        elif neg_data:
            return title, {'negative': neg_data}
        return title, None
    except Exception as e:
        print(f'  [ERR] {title}: {e}')
        return title, None

# ── Main load ──────────────────────────────────────────────
def prepare_sft_data() -> dict:
    full_data    = {}
    skip_count   = 0
    domain_count = {}

    print('Build index song song...')
    with ThreadPoolExecutor() as ex:
        # Document/ và new_summary/ có subfolder → walk
        f_doc = ex.submit(build_file_index_walk, DOCUMENT_PATH, '.txt')
        f_sum = ex.submit(build_file_index_walk, SUMMARY_PATH, '.txt')
        # new_supported_summary/ và new_reference/ phẳng → flat
        f_sup = ex.submit(build_file_index_flat, SUPPORTED_SUMMARY_PATH, '.txt')
        f_ref = ex.submit(build_file_index_flat, REFERENCE_PATH, '.json')
        f_err = ex.submit(build_folder_index, ERROR_PATH)

    doc_index = f_doc.result()
    ref_index = f_ref.result()
    sum_index = f_sum.result()
    sum_index.update(f_sup.result())  # supported ghi đè nếu trùng key
    err_index = f_err.result()

    total_docs     = len(sum_index)
    max_per_domain = int(total_docs * MAX_DOMAIN_RATIO)
    print(f'{total_docs} summary | {len(doc_index)} document | {len(ref_index)} reference')
    print(f'Domain cap: tối đa {max_per_domain} doc/domain')

    task_args = []
    for title, summary_filepath in sum_index.items():
        domain = get_domain(title)
        if domain_count.get(domain, 0) >= max_per_domain:
            skip_count += 1
            continue

        stripped     = strip_category_prefix(title)
        doc_file     = doc_index.get(title) or doc_index.get(stripped)
        ref_file     = ref_index.get(title) or ref_index.get(stripped)
        error_folder = err_index.get(title) or err_index.get(stripped)

        if not doc_file:
            skip_count += 1
            continue

        domain_count[domain] = domain_count.get(domain, 0) + 1
        task_args.append((title, summary_filepath, doc_file, ref_file, error_folder))

    print(f'Xử lý {len(task_args)} doc (bỏ qua {skip_count})')

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_one_doc, args): args[0] for args in task_args}
        for future in as_completed(futures):
            title, result = future.result()
            if result is not None:
                full_data[title] = result
            done += 1
            if done % 200 == 0:
                print(f'   ... {done}/{len(task_args)}')

    print(f'\nLoad xong {len(full_data)} doc | bỏ qua {skip_count}')
    print(f'Low-confidence sentences skipped: {low_conf_skipped}')
    print('Domain distribution:')
    for d, cnt in sorted(domain_count.items(), key=lambda x: -x[1]):
        print(f'  {d:<20} {cnt:>5}  ({cnt/max(len(full_data),1)*100:.1f}%)')
    return full_data

# ── Split ──────────────────────────────────────────────────
def calc_split(total_size: int):
    if total_size >= 300:
        valid_size = test_size = 100
    elif total_size >= 30:
        valid_size = test_size = max(1, total_size // 10)
    else:
        valid_size = test_size = max(1, total_size // 7)
    train_size = max(0, total_size - valid_size - test_size)
    print(f'  split → train={train_size}  valid={valid_size}  test={test_size}')
    return train_size, valid_size, test_size

def shuffle_and_select_data(full_data: dict, round_num: int = 1):
    sft_train = {i: [] for i in range(round_num)}
    sft_valid, sft_test = [], []

    total_size = len(full_data)
    print(f'total size: {total_size}')
    train_size, valid_size, test_size = calc_split(total_size)

    keys = list(full_data.keys())
    random.shuffle(keys)
    train_data = {k: full_data[k] for k in keys[:train_size]}
    valid_data = {k: full_data[k] for k in keys[train_size:train_size + valid_size]}
    test_data  = {k: full_data[k] for k in keys[train_size + valid_size:]}

    for rnd in range(round_num):
        pos_list = []
        neg_pool = []
        for title in train_data:
            if 'positive' in train_data[title]:
                pos_list.append(train_data[title]['positive'])
            for neg_data, error_type in train_data[title].get('negative', []):
                neg_pool.append((neg_data, error_type))

        n_pos        = len(pos_list)
        n_neg_target = n_pos * NEG_PER_POS
        random.shuffle(neg_pool)
        n_types  = len(ERROR_SAMPLE_NUM)
        quota    = max(1, n_neg_target // n_types)
        error_count  = {k: 0 for k in ERROR_SAMPLE_NUM}
        selected_neg = []

        for neg_data, error_type in neg_pool:
            if error_count.get(error_type, 0) < quota:
                selected_neg.append(neg_data)
                error_count[error_type] = error_count.get(error_type, 0) + 1
            if len(selected_neg) >= n_neg_target:
                break

        sft_train[rnd] = pos_list + selected_neg
        random.shuffle(sft_train[rnd])
        print(f'  Round {rnd}: {n_pos} pos + {len(selected_neg)} neg '
              f'(ratio 1:{len(selected_neg)//max(n_pos,1)}) | error dist: {error_count}')

    for split_data, split_list in [(valid_data, sft_valid), (test_data, sft_test)]:
        pos_samples, neg_samples = [], []
        for title in split_data:
            if 'positive' in split_data[title]:
                pos_samples.append(split_data[title]['positive'])
            for neg_data, _ in split_data[title].get('negative', []):
                neg_samples.append(neg_data)
        n = min(len(pos_samples), len(neg_samples))
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        split_list.extend(pos_samples[:n])
        split_list.extend(neg_samples[:n])
        random.shuffle(split_list)

    return sft_train, sft_valid, sft_test

# ============================================================
# 4. Kiểm tra nhanh 1 error file
# ============================================================
sample_error_file = None
for root, dirs, files in os.walk(ERROR_PATH):
    for f in files:
        if f.endswith('.txt'):
            sample_error_file = os.path.join(root, f)
            break
    if sample_error_file:
        break

if sample_error_file is None:
    print('Chưa có error file')
else:
    rel = os.path.relpath(sample_error_file, ERROR_PATH)
    print(f'Sample: {rel}\n')
    sample = safe_load_error_file(sample_error_file)
    if sample:
        for k, v in sample.items():
            print(f'  {k}: {str(v)[:100]}')

# ============================================================
# 5. Load & xuất JSONL
# ============================================================
pos_counter = pos_num = 0
neg_counter = neg_num = 0
low_conf_skipped = 0

print('=== Bước 1: Load data ===')
full_sft_data = prepare_sft_data()

print('\n=== Bước 2: Chia train/valid/test ===')
sft_train, sft_valid, sft_test = shuffle_and_select_data(full_sft_data, round_num=1)

print('\n=== Bước 3: Xuất JSONL ===')
train_data = sft_train[0]

files_to_write = [
    (f'summary_sft_train_pos1neg{NEG_PER_POS}_with_ref.jsonl', train_data),
    ('summary_sft_valid_with_ref.jsonl',                        sft_valid),
    ('summary_sft_test_with_ref.jsonl',                         sft_test),
]

for fname, data in files_to_write:
    out_path = os.path.join(SFT_PATH, fname)
    with jsonlines.open(out_path, mode='w') as writer:
        for item in data:
            writer.write(item)
    print(f'  {fname}: {len(data)} samples')

example_path = os.path.join(OUTPUT_ROOT, 'example_summary_sft_data_with_ref.jsonl')
with jsonlines.open(example_path, mode='w') as writer:
    for item in train_data[:10]:
        writer.write(item)
print(f'  example (10 mẫu) → {example_path}')

print(f'\n{"="*50}')
print(f'Thống kê:')
print(f'   Train : {len(train_data):>6} samples')
print(f'   Valid : {len(sft_valid):>6} samples')
print(f'   Test  : {len(sft_test):>6} samples')
print(f'   Tổng  : {len(train_data)+len(sft_valid)+len(sft_test):>6} samples')
print(f'   Low-conf bị filter: {low_conf_skipped} câu')
if pos_num > 0:
    print(f'   Avg positive output : {pos_counter/pos_num:.1f} từ')
if neg_num > 0:
    print(f'   Avg negative output : {neg_counter/neg_num:.1f} từ')

# ============================================================
# 6. Xem trước & phân tích phân phối
# ============================================================
print('=' * 60)
print('POSITIVE SAMPLE:')
print('=' * 60)
pos_samples = [s for s in train_data if 'answer is NO.' in s['text']]
if pos_samples:
    parts = random.choice(pos_samples)['text'].split('<|im_start|>assistant\n')
    if len(parts) == 2:
        print('[OUTPUT]:')
        print(parts[1].replace('<|im_end|>', '')[:600])

print()
print('=' * 60)
print('NEGATIVE SAMPLE:')
print('=' * 60)
neg_samples = [s for s in train_data if 'answer is YES.' in s['text']]
if neg_samples:
    parts = random.choice(neg_samples)['text'].split('<|im_start|>assistant\n')
    if len(parts) == 2:
        print('[OUTPUT]:')
        print(parts[1].replace('<|im_end|>', '')[:600])

# ── Phân phối error type ───────────────────────────────────
def count_error_types(data: list) -> Counter:
    c = Counter()
    for sample in data:
        output_part = sample['text'].split('<|end_header_id|>:')[-1]
        if 'answer is YES.' in output_part:
            m = re.search(r'- Error Type: (.+)', output_part)
            if m:
                c[m.group(1).strip()] += 1
        elif 'answer is NO.' in output_part:
            c['[Positive — No Error]'] += 1
    return c

for split_name, split_data in [('Train', train_data), ('Valid', sft_valid), ('Test', sft_test)]:
    counts = count_error_types(split_data)
    total  = sum(counts.values())
    print(f'\n{split_name} ({total} samples):')
    for error_type, count in sorted(counts.items()):
        bar = '█' * (count * 30 // max(counts.values()))
        print(f'  {error_type:<30} {count:>4}  {bar}')
