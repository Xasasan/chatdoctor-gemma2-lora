# ChatDoctor: Gemma QLoRA fine-tuning for medical Q&A

Fine-tunes Google Gemma on patient-question → doctor-answer pairs from
[`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
using 4-bit QLoRA, then merges the adapter and publishes the model to the Hugging Face Hub.

| Notebook | Base model | Data | LoRA | Training | Model on HF |
|---|---|---|---|---|---|
| `chatdoctor_gemma2_2b_lora.ipynb` | `google/gemma-2-2b-it` | 2,000-sample subset (90/10 split) | r=16, α=32, all linear layers | 3 epochs, lr 2e-4, eff. batch 16, seq 512 | [Xasan01/Gemma-2-2b-it-ChatDoctor](https://huggingface.co/Xasan01/Gemma-2-2b-it-ChatDoctor) |
| `gemma7b_doctor_ai.ipynb` | `google/gemma-7b` | 6,000-sample subset (90/10 split) | r=32, α=64, all linear layers | 3 epochs, lr 2e-4, eff. batch 16, seq 1024 | [Xasan01/gemma-Doctor_ai](https://huggingface.co/Xasan01/gemma-Doctor_ai) |

## What the notebooks do
1. Load the base model in **4-bit NF4** (bitsandbytes), so a 2B/7B model fits on one Kaggle GPU (~75% less memory than fp16).
2. Find every 4-bit linear layer and attach **LoRA** adapters to them (`find_all_linear_names`).
3. Format each record with a ChatML chat template (`<|im_start|>system/user/assistant`).
4. Train with TRL `SFTTrainer` (early stopping on eval loss in the 7B run).
5. Merge LoRA into the base weights and `push_to_hub`; run a sample medical question through the merged model.

## Limitations (honest notes)
- Trained on small subsets (2K / 6K of ~112K records) because of free-GPU time limits.
- Evaluated by loss and manual spot checks only. No held-out QA metric or clinician review.
- **Not medical advice.** This is a learning project, not a clinical tool.

## Next steps I'd take
Train on the full dataset, add an automatic eval (e.g. LLM-as-judge on a fixed held-out set + ROUGE),
and compare against the base model on the same questions.

Stack: PyTorch · Transformers · PEFT · TRL · bitsandbytes · Kaggle GPUs
