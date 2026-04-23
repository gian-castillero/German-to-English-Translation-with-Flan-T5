# German-to-English Translation with Flan-T5

Evaluating Flan-T5 encoder-decoder models for German-to-English machine translation across three model scales (small, base, large), with BLEU score evaluation and prompt engineering.

## Overview

This project applies Google's Flan-T5, a sequence-to-sequence encoder-decoder transformer pretrained on a diverse mixture of tasks including machine translation, to translate 500 sentences of German literary text (from a parallel corpus of *Jane Eyre*) into English. Translation quality is measured using BLEU scores. A key experiment studies how model scale affects translation quality across the `flan-t5-small`, `flan-t5-base`, and `flan-t5-large` variants.

## Model

**Flan-T5** is a fine-tuned version of the original T5 (Text-to-Text Transfer Transformer) architecture. Unlike decoder-only causal models, T5 uses a full encoder-decoder structure: the encoder processes the input sequence bidirectionally, and the decoder autoregressively generates the output sequence conditioned on the encoder's representation.

Because Flan-T5 was pretrained on multiple tasks, **prompting** is required to activate the correct behavior:

```
"Translate German into English: <input sentence>"
```

Without this prompt, the model does not produce translations, it defaults to other behaviors from its pretraining mixture.

All models are loaded in **16-bit float precision** (`torch_dtype=torch.float16`) to reduce memory usage.

### Model Scale & Memory

| Model | Parameters | Memory (16-bit) |
|-------|-----------|-----------------|
| flan-t5-small | ~77M | ~150 MB |
| flan-t5-base | ~248M | ~484 MB |
| flan-t5-large | ~783M | ~1.5 GB |

Memory is computed as: `parameters × 2 bytes ÷ 1024²` MB.

## Dataset

[OPUS Books](https://huggingface.co/datasets/opus_books) `de-en` split — a parallel German-English corpus of literary texts. 500 sentence pairs are sampled from the *Jane Eyre* translation for evaluation.

## Evaluation: BLEU Score

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between a model's translation and a reference translation, ranging from 0 (no overlap) to 1 (perfect match). It is an imperfect but widely used metric; a score of 1.0 is neither expected nor necessary for a high-quality translation.

## Results

| Model | BLEU Score |
|-------|-----------|
| flan-t5-small | lower |
| flan-t5-base | higher |
| flan-t5-large | highest |

Larger models consistently produce better translations, with meaningful BLEU score improvements at each scale step. This demonstrates the well-established empirical scaling law: for language generation tasks, more parameters generally yield better performance given the same training procedure.

## Key Findings

- Prompting is essential for task-specific behavior in instruction-tuned models, the same weights produce very different outputs depending on how the input is framed.
- BLEU scores improve monotonically with model size across all three variants tested.
- 16-bit precision allows larger models to fit in memory with minimal impact on output quality, an important practical consideration for deployment.

## Tech Stack

- Python 3
- PyTorch
- Hugging Face `transformers` (`T5Tokenizer`, `T5ForConditionalGeneration`)
- Hugging Face `datasets` (`opus_books`)
- Hugging Face `evaluate` (BLEU)

## How to Run

```bash
pip install torch transformers datasets evaluate sentencepiece jupyter
jupyter notebook flan-translate.ipynb
```

GPU acceleration is strongly recommended; translating 500 sentences takes several minutes per model on CPU. Models and dataset download automatically on first run.
