# VInFi-Check: Interpretable Fact-Checking for Vietnamese News Summarization

<p align="center">
  <a href="https://arxiv.org/abs/2601.06666">
    <img src="https://img.shields.io/badge/Inspired%20by-InFi--Check%20(arXiv%3A2601.06666)-b31b1b?logo=arxiv"/>
  </a>
  <img src="https://img.shields.io/badge/Language-Vietnamese-yellow"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab"/>
</p>

**VInFi-Check** is a fact-checking pipeline for Vietnamese news summarization. Given a Vietnamese news article and an LLM-generated summary, VInFi-Check verifies each summary sentence against the source document through multi-LLM voting, producing a *supported summary* with sentence-level evidence traces.

Most existing fact-checking methods treat hallucination detection as a binary problem. VInFi-Check instead performs **sentence-level, evidence-backed verification** — each sentence is either confirmed with its grounding sentences from the source, revised, or discarded. The resulting dataset can then be used to fine-tune a lightweight Vietnamese fact-checker.

---

## ✨ Contributions

- **A quality-filtered Vietnamese news corpus** spanning 20 topic categories, selected via Type-Token Ratio, entity density scoring, and Jaccard bigram deduplication.
- **A Vietnamese summary generation pipeline** using DeepSeek with constraints designed to preserve named entities, dates, and factual precision.
- **A multi-LLM voting verification pipeline** (GPT-4o-mini · Qwen-2.5-72B · LLaMA-3.3-70B) that produces sentence-level supported summaries with evidence references — adapted from the [InFi-Check](https://arxiv.org/abs/2601.06666) framework for Vietnamese.
- **A structured training dataset and fine-tuned model**, with an interactive Gradio demo for real-time fact-checking.

---

## 🗂️ Repository Structure

```
VInFi_Check/
├── InFi-Check construct/
│   └── selected_dataset/
│       ├── new_summary/                  # LLM-generated summaries (per category)
│       ├── new_supported_summary/        # Verified summaries (passed majority vote)
│       └── new_reference/               # Sentence-level evidence + vote counts (.json)
├── training_dataset_construct/
│   ├── prepare_dataset_pipeline_c_.ipynb
│   └── structured_dataset_gen.py
├── finetune/
│   └── finetune.ipynb
├── VInFiCheck_Gradio.ipynb
└── README.md
```

---

## 🚀 Pipeline

### Step 1 — Dataset Preparation

Articles from `Dataset NLP/` are filtered per category using three quality signals:

- **Type-Token Ratio (TTR)** — lexical richness
- **Entity density** — ratio of capitalized tokens (named entities)
- **Jaccard bigram deduplication** — removes near-duplicate articles (threshold 0.5)

Up to **200 articles per category** are selected, within 150–700 words.

---

### Step 2 — Summary Generation · `summary_gen.py`

**Model:** `deepseek-chat` via DeepSeek API

Each article is summarized in **60–150 words** in Vietnamese. The prompt explicitly instructs the model to:

- Retain named entities, locations, dates, and numbers exactly
- Keep co-occurring entities in parallel (e.g., two people mentioned together)
- Avoid over-simplification that changes factual meaning
- Never use pronouns (ông, bà, họ, ...) without a referent in the same sentence

**Output:** `new_summary/<category>/<file>_summary.txt`

---

### Step 3 — Eval & Reference Generation · `eval_and_reference_gen.py`

This is the core verification step. For each summary sentence:

**① Find support** — GPT-4o-mini locates grounding sentences from the source document.

**② Multi-LLM majority vote** — Three models independently judge whether the sentence is supported:

| Model | Provider | Role |
|---|---|---|
| `gpt-4o-mini` | OpenAI | Judge + can suggest revision |
| `qwen/qwen-2.5-72b-instruct` | OpenRouter | Judge |
| `llama-3.3-70b-versatile` | Groq | Judge |

A sentence passes if **> 50% of responses** vote YES.

**③ Iterative revision** — If the sentence fails but GPT-4o-mini proposes a corrected version, the revised sentence is re-voted. This repeats up to **3 rounds**.

**Output:**
- `new_supported_summary/<file>_supported_summary.txt` — verified summary text
- `new_reference/<file>_ref.json` — per-sentence evidence and vote counts

```json
{
  "find_support_result": [
    {
      "summary sentence": "...",
      "reference": ["grounding sentence 1", "grounding sentence 2"],
      "votes": "2/3"
    }
  ]
}
```

---

### Step 4 — Training Data Construction

`structured_dataset_gen.py` and `prepare_dataset_pipeline_c_.ipynb` convert the verified summaries and reference JSONs into structured JSONL samples for instruction fine-tuning.

---

### Step 5 — Fine-tuning · `finetune.ipynb`

Supervised fine-tuning (SFT) of a language model on the Vietnamese fact-checking dataset constructed in Step 4.

---

### Step 6 — Gradio Demo · `VInFiCheck_Gradio.ipynb`

An interactive web interface where users paste a Vietnamese article and a summary to receive sentence-level fact-checking results in real time.

---

## 📋 Dataset — 20 Categories

| Folder | Topic | 
|---|---|
| `Van_hoa` | Culture & arts |
| `Du_lich` | Travel | 
| `Khoa_hoc` | Science | 
| `Thi_truong` | Market | 
| `Moi_truong` | Environment | 
| `So_hoa` | Digitalization | 
| `Tam_su` | Personal stories | 
| `Y_kien` | Opinion | 
| `Cong_doan` | Trade unions |
| `Chong_Dien_Bien_Hoa_Binh` | Political content | 
| `Kinh_te` | Economics |
| `Y_te` | Healthcare | 
| `The_thao` | Sports | 
| `Kinh_doanh` | Business |
| `The_gioi` | World news | 
| `Cong_nghe` | Technology | 
| `Doi_song` | Lifestyle | 
| `Giao_duc` | Education | 
| `Giai_tri` | Entertainment |
| `Phap_luat` | Law & justice | 

> Articles for categories marked — are included in the pipeline with the same quality-filtering logic (TTR + entity density + Jaccard bigram dedup, up to 200 per category).

---
## 🤗 Model
 
Built on Qwen2.5-7B with QLoRA fine-tuning on the VInFi-Check Vietnamese fact-checking dataset.
 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model_id = "sunflowerbiii/infi-check-qwen25-7b-qlor"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```
 
---
## ⚙️ Setup

```bash
pip install openai requests gradio
```

### API Keys

| Variable | Service |
|---|---|
| `DEEPSEEK_API_KEY` | [DeepSeek Platform](https://platform.deepseek.com/) |
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) |
| `OPENROUTER_KEYS` | [OpenRouter](https://openrouter.ai/) — Qwen-2.5-72B |
| `GROQ_KEYS` | [Groq](https://console.groq.com/) — LLaMA-3.3-70B |

Keys for OpenRouter and Groq accept **lists** to enable automatic rotation on rate-limit errors.

### Required Prompt Files

The pipeline loads 6 prompt files at runtime from `summary_eval_prompt/`:

```
find_support.txt
find_support_format.txt
critics.txt
critics_format.txt
critics_with_revise.txt
critics_with_revise_format.txt
```

### Google Drive Layout (Colab)

```
/content/drive/MyDrive/
├── Dataset NLP/
│   ├── Van_hoa/
│   ├── Du_lich/
│   ├── Khoa_hoc/
│   └── ...  (20 categories)
└── Phosphor-Bai-InFi-Check/
    └── InFi-Check construct/
        └── selected_dataset/
            ├── new_summary/
            ├── new_supported_summary/
            ├── new_reference/
            └── ../summary_eval_prompt/
```

---

## 🔗 Reference

This work adapts the verification pipeline design from:

> **InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs**  
> Yuzhuo Bai\*, Shuzheng Si\*, Kangyang Luo, Qingyi Wang, Wenhao Li, Gang Chen, Fanchao Qi, Maosong Sun  
> Tsinghua University · DeepLang AI · Fudan University  
> arXiv:2601.06666, January 2026 · https://arxiv.org/abs/2601.06666

```bibtex
@article{bai2026inficheck,
  title   = {InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs},
  author  = {Bai, Yuzhuo and Si, Shuzheng and Luo, Kangyang and Wang, Qingyi and
             Li, Wenhao and Chen, Gang and Qi, Fanchao and Sun, Maosong},
  journal = {arXiv preprint arXiv:2601.06666},
  year    = {2026}
}
```
