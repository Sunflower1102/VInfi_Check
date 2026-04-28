# 🔍 VInfi-Check: Interpretable and Fine-grained Fact-Checking for Vietnamese Summaries

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Qwen](https://img.shields.io/badge/Model-Qwen2.5--7B-green.svg)
![Task](https://img.shields.io/badge/Task-Fact--Checking%20%7C%20Hallucination%20Detection-lightgrey.svg)

**VInfi-Check** là một 파peline hoàn chỉnh để tự động xây dựng bộ dữ liệu (Dataset Construction) và Fine-tune mô hình Ngôn ngữ lớn (LLM) nhằm mục đích kiểm chứng thông tin (Fact-checking) cho các văn bản tóm tắt tiếng Việt. 

Dự án này tập trung vào việc phát hiện và giải thích các lỗi "ảo giác" (hallucinations) chi tiết ở cấp độ câu, bao gồm các loại lỗi: *Predicate Error, Entity Error, Circumstance Error, Co-reference Error, Discourse Link Error, và Extrinsic Error*.

## 📑 Cấu trúc dự án (Pipeline)

Dự án được chia thành 5 bước chính, tương ứng với các file mã nguồn:

1. **`summary_gen.py` (Bước 3 - Summary Generation):** - Sử dụng DeepSeek API để tạo các bản tóm tắt ngắn (60-150 từ) từ các bài báo gốc tiếng Việt.
   - Tích hợp bộ lọc chất lượng (TTR, Entity Density, Jaccard Jaccard bigram) để loại bỏ các văn bản kém chất lượng.
2. **`eval_and_reference_gen.py` (Bước 4 - Eval & Reference):**
   - Đánh giá các bản tóm tắt vừa tạo để tìm ra các câu được hỗ trợ (supported) bởi văn bản gốc.
   - Sử dụng cơ chế Multi-Agent Voting với sự tham gia của `gpt-4o-mini`, `qwen-2.5-72b`, và `llama-3.3-70b` để đảm bảo độ chính xác của nhãn.
3. **`structured_dataset_gen.py` (Bước 5 - Negative Sample Generation):**
   - Tạo dữ liệu lỗi (Negative data) một cách có cấu trúc dựa trên các mẫu tóm tắt đúng.
   - Bơm các lỗi ảo giác (hallucination) có chủ đích thông qua DeepSeek API, kết hợp Few-shot prompting.
4. **`prepare_dataset_pipeline(c).ipynb` (Bước 6 - SFT Dataset Prep):**
   - Ghép nối dữ liệu Positive (đúng) và Negative (lỗi).
   - Phân chia tập Train/Valid/Test và xuất ra định dạng `.jsonl` chuẩn bị cho Supervised Fine-Tuning.
5. **`finetune.py` (Bước 7 - Fine-Tuning):**
   - Huấn luyện mô hình `Qwen/Qwen2.5-7B-Instruct` bằng kỹ thuật **QLoRA** (4-bit) kết hợp **NEFTune**.
   - Tích hợp tự động push mô hình lên Hugging Face Hub.

## 🛠 Yêu cầu hệ thống (Prerequisites)

- **OS:** Linux (Ubuntu) / Google Colab Pro / Workspace có GPU.
- **Hardware:** Ít nhất 1 GPU có VRAM ≥ 16GB (Khuyến nghị: RTX 3090, 4090 hoặc A100) để chạy QLoRA 7B.
- **API Keys:** Cần có API Keys của OpenAI, OpenRouter, Groq, và DeepSeek.

## 📦 Cài đặt

1. Clone repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<ten-github-cua-ban>/VInfi_Check.git
   cd VInfi_Check
