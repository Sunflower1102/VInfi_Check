# ============================================================
# InFi-Check — Fine-tune Qwen2.5-7B QLoRA (v4)
# ============================================================

import os
import sys
import json
import time
import subprocess

# ── Env vars (đặt trước khi import torch) ──────────────────
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_ALLOC_CONF']     = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    GenerationConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import HfApi, login

# ============================================================
# 1. Kiểm tra GPU
# ============================================================
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)

if not torch.cuda.is_available():
    raise RuntimeError('Không tìm thấy GPU!')

print(f'GPU   : {torch.cuda.get_device_name(0)}')
print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
USE_BF16 = torch.cuda.is_bf16_supported()
print(f'BF16  : {USE_BF16}')

# ============================================================
# 2. Cấu hình đường dẫn  ✏️ sửa nếu cần
# ============================================================
SFT_ROOT   = '/workspace/inficheck-dataset'
SAVE_ROOT  = '/workspace/infi-check-output'
OUTPUT_DIR = '/workspace/infi-check-qwen25-7b-v4'

TRAIN_FILE = os.path.join(SFT_ROOT, 'summary_sft_train_pos1neg1_with_ref.jsonl')
VALID_FILE = os.path.join(SFT_ROOT, 'summary_sft_valid_with_ref.jsonl')
TEST_FILE  = os.path.join(SFT_ROOT, 'summary_sft_test_with_ref.jsonl')

os.makedirs(SAVE_ROOT,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def count_labels(data):
    yes = 0
    for d in data:
        text = d['text']
        if '<|im_start|>assistant\n' in text:
            gt = text.split('<|im_start|>assistant\n', 1)[-1].replace('<|im_end|>', '')
        elif '<|end_header_id|>:' in text:
            gt = text.split('<|end_header_id|>:', 1)[-1]
        else:
            gt = text
        if 'the answer is yes' in gt.lower():
            yes += 1
    return yes, len(data) - yes

print('\n📂 Kiểm tra dataset:')
for name, path in [('Train', TRAIN_FILE), ('Valid', VALID_FILE), ('Test', TEST_FILE)]:
    try:
        data = load_jsonl(path)
        yes, no = count_labels(data)
        print(f'  {name}: {len(data)} mẫu | YES: {yes} | NO: {no} | '
              f'Tỉ lệ: {yes/len(data)*100:.0f}/{no/len(data)*100:.0f}')
    except Exception as e:
        print(f'  {name}: {e}')

# ============================================================
# 3. HuggingFace Login  ✏️ điền token
# ============================================================
HF_TOKEN = 'your_token_here'
login(token=HF_TOKEN)
print('HuggingFace login thành công')

# ============================================================
# 4. Load Model (4-bit QLoRA)
# ============================================================
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f'\n📥 Đang load {MODEL_ID}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

IM_END_ID      = tokenizer.convert_tokens_to_ids('<|im_end|>')
EOT_ID         = tokenizer.convert_tokens_to_ids('<|endoftext|>')
STOP_TOKEN_IDS = list(set([IM_END_ID, EOT_ID]))
EOS_ID         = EOT_ID

tokenizer.pad_token    = tokenizer.convert_ids_to_tokens(EOT_ID)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={'': 0},
    trust_remote_code=True,
    attn_implementation='sdpa',
)

for i in range(torch.cuda.device_count()):
    used  = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'GPU {i}: {used:.2f} GB / {total:.1f} GB')

# ============================================================
# 5. LoRA Config
# ============================================================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'\nLoRA r=32, alpha=64')
print(f'Trainable : {trainable:,} ({trainable/total:.2%})')
print(f'Total     : {total:,}')

# ============================================================
# 6. Dataset
# ============================================================
def formatting_func(example):
    texts = example['text']
    return texts  # hoạt động cả batch lẫn single mode

def convert_llama_to_qwen(item):
    """Tương thích ngược — convert Llama format sang Qwen nếu cần."""
    text = item['text']
    if '<|im_start|>' in text:
        return item
    parts = text.split('<|end_header_id|>:')
    input_text  = parts[0].replace('<|start_header_id|>:', '').strip()
    output_text = parts[1].strip() if len(parts) > 1 else ''
    return {'text': (
        f'<|im_start|>user\n{input_text}<|im_end|>\n'
        f'<|im_start|>assistant\n{output_text}<|im_end|>'
    )}

train_data = load_jsonl(TRAIN_FILE)
valid_data = load_jsonl(VALID_FILE)
valid_data = [convert_llama_to_qwen(d) for d in valid_data]

MAX_SEQ_LEN = 1536  # giảm từ 2048 để tiết kiệm VRAM

response_template_ids = tokenizer.encode(
    '<|im_start|>assistant\n',
    add_special_tokens=False,
)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

dataset = {
    'train':      Dataset.from_list(train_data),
    'validation': Dataset.from_list(valid_data),
}

print(f'\nDataset: Train={len(dataset["train"])} | Valid={len(dataset["validation"])}')

# ── Sanity check: collator tìm đúng response boundary ──────
sample_text = formatting_func(train_data[0])
tokenized   = tokenizer(sample_text, return_tensors='pt')
batch       = collator([{
    'input_ids':      tokenized['input_ids'][0],
    'attention_mask': tokenized['attention_mask'][0],
}])
labels    = batch['labels'][0]
n_learned = (labels != -100).sum().item()
print(f'Sanity check — tokens cần học: {n_learned} / {len(labels)}')
assert n_learned > 0, 'STOP — collator không tìm thấy response boundary!'
print('Sanity check passed!')

# ============================================================
# 7. Training Config
# ============================================================
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
EFFECTIVE_BS = BATCH_SIZE * GRAD_ACCUM

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    optim='adamw_torch',
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    max_grad_norm=0.3,

    label_smoothing_factor=0.0,

    eval_strategy='steps',
    eval_steps=100,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,

    logging_steps=50,
    report_to='none',

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,

    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)

# ── Callback log tiến trình từng epoch ────────────────────
class EpochProgressCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start = None
        self.train_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        epoch   = int(state.epoch) + 1
        total   = int(args.num_train_epochs)
        elapsed = time.time() - self.train_start
        print(f'\nEpoch {epoch}/{total} bắt đầu... (đã chạy: {elapsed/60:.1f} phút)')

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch      = int(state.epoch)
        total      = int(args.num_train_epochs)
        epoch_time = time.time() - self.epoch_start
        elapsed    = time.time() - self.train_start
        remaining  = (elapsed / epoch) * (total - epoch) if epoch > 0 else 0
        eval_loss  = next(
            (log['eval_loss'] for log in reversed(state.log_history) if 'eval_loss' in log),
            None
        )
        loss_str = f' | eval_loss: {eval_loss:.4f}' if eval_loss else ''
        print(f'Epoch {epoch}/{total} xong{loss_str} | '
              f'{epoch_time/60:.1f} phút | còn lại ~{remaining/60:.1f} phút')

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=collator,
    max_seq_length=MAX_SEQ_LEN,
    formatting_func=formatting_func,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

steps_per_epoch = len(dataset['train']) // EFFECTIVE_BS
print(f'\nTrainer sẵn sàng')
print(f'  Train          : {len(dataset["train"])} mẫu')
print(f'  Valid          : {len(dataset["validation"])} mẫu')
print(f'  Effective batch: {EFFECTIVE_BS} ({BATCH_SIZE} × {GRAD_ACCUM})')
print(f'  Steps/epoch    : ~{steps_per_epoch}')
print(f'  Max seq len    : {MAX_SEQ_LEN}')

# ============================================================
# 8. Train
# ============================================================
def remove_neftune_hooks(model):
    emb = model.get_input_embeddings()
    n   = len(emb._forward_hooks)
    if n > 0:
        emb._forward_hooks.clear()
        print(f'Đã xóa {n} NEFTune hook(s)')
    return model

if os.path.exists(f'{SAVE_ROOT}/adapter_model.safetensors'):
    print('Model đã train xong, skip!')
else:
    print('🚀 Bắt đầu fine-tuning...')
    trainer.add_callback(EpochProgressCallback())
    trainer.train()
    print('Training hoàn tất!')

model = remove_neftune_hooks(trainer.model)

# ============================================================
# 9. Lưu model
# ============================================================
trainer.model.save_pretrained(SAVE_ROOT)
tokenizer.save_pretrained(SAVE_ROOT)

with open(f'{SAVE_ROOT}/training_log.json', 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f'Model lưu tại: {SAVE_ROOT}')
print(subprocess.run(['du', '-sh', SAVE_ROOT], capture_output=True, text=True).stdout)

# ============================================================
# 10. Learning curve
# ============================================================
log_history = trainer.state.log_history
train_steps, train_loss = [], []
eval_steps,  eval_loss  = [], []

for log in log_history:
    if 'loss' in log and 'eval_loss' not in log:
        train_steps.append(log['step'])
        train_loss.append(log['loss'])
    if 'eval_loss' in log:
        eval_steps.append(log['step'])
        eval_loss.append(log['eval_loss'])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_steps, train_loss, label='Train Loss', alpha=0.7)
ax.plot(eval_steps,  eval_loss,  label='Eval Loss',  marker='o')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('InFi-Check v4 — Training Curve')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_ROOT}/training_curve_v4.png', dpi=150)
plt.show()

if eval_loss:
    best_step = eval_steps[eval_loss.index(min(eval_loss))]
    print(f'Best eval_loss: {min(eval_loss):.4f} tại step {best_step}')

# ============================================================
# 11. Inference helper
# ============================================================
model.eval()
model.generation_config = GenerationConfig(
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=1.1,
    eos_token_id=STOP_TOKEN_IDS,
    pad_token_id=EOS_ID,
)

def build_prompt(document: str, summary: str) -> str:
    instruction = (
        'Your task is to evaluate a summary by comparing it to the original document '
        'and identifying any errors present in the summary. These errors may involve '
        'incorrect information, over-simplifications, misrepresentations, or other discrepancies. '
        'Below are the possible types of errors you should consider:\n'
        '- Semantic Frame Errors: Predicate Error, Entity Error, Circumstance Error\n'
        '- Discourse Errors: Co-reference Error, Discourse Link Error\n'
        '- Extrinsic Errors: Extrinsic Error (information introduced into the summary '
        'that is not present in or supported by the original document)\n\n'
        'You are provided with the full text of the original document, and a summary '
        'of the document that might contain errors.\n\n'
        'You should output:\n'
        '1. Analyze the content of the summary compared to the original document. '
        'For each identified error, provide:\n'
        '- Location: Where the error occurs in the summary.\n'
        '- Explanation: Why the original meaning is altered or why the information '
        'is not supported by the document.\n'
        '- Correction: A revised version of the erroneous part of the summary.\n'
        '- Error Type: Specify the exact error type based on the categories listed above.\n'
        '2. Answer whether the summary contains errors that make it not fully '
        'supported by the document.\n\n'
        'CRITICAL - Do NOT flag as errors:\n'
        '- Sentences copied word-for-word from the document\n'
        '- Sentences that omit details but do not add incorrect information\n'
        '- Listing a subset of items from the document without adding wrong ones\n'
        '- Paraphrasing that preserves the original meaning\n\n'
        'Important: Write your analysis and explanations in Vietnamese. '
        'Only keep the field labels (Location, Explanation, Correction, Error Type) '
        'and the final answer format in English.\n\n'
        f'Document:\n{document}\n\nSummary:\n{summary}'
    )
    return f'<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'

def run_inference(document: str, summary: str, max_new_tokens: int = 512) -> str:
    prompt = build_prompt(document, summary)
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=STOP_TOKEN_IDS,
            pad_token_id=EOS_ID,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).replace('<|im_end|>', '').strip()

def parse_sample(raw: dict):
    """Parse 1 dòng JSONL — hỗ trợ cả Llama format lẫn Qwen format."""
    text = raw['text']
    if '<|im_start|>user' in text:
        parts = text.split('<|im_start|>assistant\n')
        instr = parts[0].replace('<|im_start|>user\n', '').replace('<|im_end|>', '').strip()
        gt    = parts[1].replace('<|im_end|>', '').strip() if len(parts) > 1 else ''
    else:
        parts = text.split('<|end_header_id|>:')
        instr = parts[0].replace('<|start_header_id|>:', '').strip()
        gt    = parts[1].strip() if len(parts) > 1 else ''

    doc_start = instr.find('Document:')
    sum_start = instr.rfind('Summary:')
    document  = instr[doc_start:sum_start].replace('Document:', '', 1).strip()
    summary   = instr[sum_start:].replace('Summary:', '', 1).strip()
    return document, summary, gt

# Quick test trên 1 mẫu test
with open(TEST_FILE) as f:
    test_sample = json.loads(f.readline())

document, summary, gt = parse_sample(test_sample)
print('\n📄 DOCUMENT (300 ký tự):')
print(document[:300] + '...')
print('\n📝 SUMMARY:')
print(summary)
print('\n' + '='*60)
print('🎯 GROUND TRUTH:')
print(gt)
print('\n' + '='*60)
print('🤖 MODEL OUTPUT:')
print(run_inference(document, summary))

# ============================================================
# 12. Push lên HuggingFace  ✏️ điền token write
# ============================================================
HF_TOKEN_WRITE = 'your_write_token_here'
login(token=HF_TOKEN_WRITE)

HF_USERNAME = 'sunflowerbiii'
REPO_NAME   = 'infi-check-qwen25-7b-qlora-c'

api = HfApi()
api.create_repo(repo_id=f'{HF_USERNAME}/{REPO_NAME}', token=HF_TOKEN_WRITE, exist_ok=True)
api.upload_folder(
    folder_path=SAVE_ROOT,
    repo_id=f'{HF_USERNAME}/{REPO_NAME}',
    token=HF_TOKEN_WRITE,
)
print(f'https://huggingface.co/{HF_USERNAME}/{REPO_NAME}')
