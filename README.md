# Text Summarization Using NLP

### Deep Learning for Automated News Digest Generation

This repository contains an NLP-based text summarization project focused on reproducing and extending transformer-based abstractive summarization experiments. The project evaluates how well modern sequence-to-sequence models generate concise summaries from long news articles and compares their performance across in-domain and cross-domain datasets.

The main goal is to study automated news digest generation using deep learning models such as **T5**, **BART**, and **PEGASUS**, along with a classical **TF-IDF extractive baseline**. The project uses standard summarization datasets and evaluates generated summaries using widely used metrics such as **ROUGE**, **BLEU**, and **METEOR**.

---

## Project Overview

Text summarization is the process of converting long documents into shorter summaries while preserving the most important information. This project mainly focuses on **abstractive summarization**, where the model generates new summary sentences instead of simply copying sentences from the source document.

The repository reproduces summarization experiments on the **CNN/DailyMail** dataset and extends the work by testing cross-domain generalization on the **XSum** dataset. It also includes an extractive TF-IDF baseline and an additional BART fine-tuning experiment on XSum to analyze whether domain adaptation improves performance.

---

## Objectives

- Reproduce transformer-based summarization results on CNN/DailyMail.
- Fine-tune and evaluate T5, BART, and PEGASUS models.
- Compare pre-training and post-training summarization performance.
- Test zero-shot transfer from CNN/DailyMail to XSum.
- Implement a TF-IDF extractive baseline for comparison.
- Fine-tune BART on XSum to measure domain adaptation effects.
- Analyze model performance using ROUGE, BLEU, and METEOR.

---

## Models Used

| Model | Type | Description |
|------|------|-------------|
| T5-base | Transformer | Text-to-text summarization model |
| T5-large | Transformer | Larger T5 model for abstractive summarization |
| BART-large-CNN | Transformer | Denoising sequence-to-sequence model fine-tuned for news summarization |
| PEGASUS-CNN/DailyMail | Transformer | Summarization-focused transformer model |
| TF-IDF | Classical NLP | Extractive baseline using sentence scoring |

---

## Datasets

| Dataset | Purpose | Summary Style |
|--------|---------|---------------|
| CNN/DailyMail | Main reproduction dataset | Multi-sentence news highlights |
| XSum | Cross-domain evaluation dataset | Highly abstractive single-sentence summaries |

---

## Repository Structure

```text
Code Repo/
├── text_summarization.py                     # Main reproduction script for transformer models
├── bart.log                                  # BART training log
├── pegasus.log                               # PEGASUS training log
├── t5base.log                                # T5-base training log
├── t5large.log                               # T5-large training log
├── text_summarization_run.log                # Main experiment execution log
├── pre_training_metrics.png                  # Pre-training comparison plot
├── post_training_metrics.png                 # Post-training comparison plot
├── bart_vs_pegasus_pre_per_sample.csv        # Per-sample metric comparison before fine-tuning
├── bart_vs_pegasus_post_per_sample.csv       # Per-sample metric comparison after fine-tuning
├── bart_vs_t5large_post_per_sample.csv       # BART vs T5-large comparison
├── pegasus_vs_t5large_post_per_sample.csv    # PEGASUS vs T5-large comparison
│
├── TF-IDF/
│   ├── run_tfidf_cnn_dailymail.py            # TF-IDF extractive baseline
│   └── tfidf_cnn_dailymail_metrics.json      # TF-IDF evaluation results
│
├── XSUM Zero Shot Transfer/
│   ├── xsum_zero-shot.py                     # Zero-shot XSum evaluation script
│   ├── xsum_zero_shot_metrics.json           # XSum zero-shot metrics
│   ├── *_xsum_predictions.txt                # Generated summaries
│   ├── *_xsum_references.txt                 # Reference summaries
│   └── xsum_zero_shot_metric_bars.png        # XSum evaluation plot
│
└── fine tuning on XSum/
    ├── run_bart_xsum.py                      # BART fine-tuning script on XSum
    ├── bart_large_cnn_xsum_metrics.json      # BART XSum fine-tuning metrics
    ├── bart_large_cnn_xsum_pre_predictions.txt
    ├── bart_large_cnn_xsum_post_predictions.txt
    └── bart_large_cnn_xsum_references.txt
```

---

## Installation

Install the required dependencies:

```bash
pip install torch transformers datasets rouge-score nltk tqdm pandas numpy scikit-learn matplotlib
```

A CUDA-enabled GPU is strongly recommended for training large transformer models.

---

## How to Run

### 1. Run Main Summarization Reproduction

```bash
python text_summarization.py
```

This script evaluates and fine-tunes transformer-based summarization models on CNN/DailyMail.

### 2. Run TF-IDF Baseline

```bash
python "TF-IDF/run_tfidf_cnn_dailymail.py"
```

This script creates an extractive summarization baseline using TF-IDF sentence ranking.

### 3. Run XSum Zero-Shot Transfer

```bash
python "XSUM Zero Shot Transfer/xsum_zero-shot.py" --models all
```

This evaluates fine-tuned CNN/DailyMail models on XSum without additional training.

### 4. Fine-Tune BART on XSum

```bash
python "fine tuning on XSum/run_bart_xsum.py"
```

This fine-tunes BART-large-CNN directly on XSum and compares pre-training and post-training performance.

---

## Evaluation Metrics

The project uses the following metrics:

- **ROUGE-1**: unigram overlap between generated and reference summaries
- **ROUGE-2**: bigram overlap
- **ROUGE-L**: longest common subsequence overlap
- **BLEU**: n-gram precision-based evaluation
- **METEOR**: semantic-aware metric using stemming and synonym matching

---

## Key Results

### CNN/DailyMail Post-Training Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|------|---------|---------|---------|------|
| T5-base | 0.000 | 0.000 | 0.000 | 0.000 |
| T5-large | 0.418 | 0.193 | 0.294 | 0.076 |
| BART-large-CNN | 0.448 | 0.215 | 0.308 | 0.110 |
| PEGASUS-large | 0.445 | 0.214 | 0.313 | 0.108 |

### XSum Zero-Shot Transfer Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|------|---------|---------|---------|------|--------|
| BART-large-CNN | 0.210 | 0.036 | 0.138 | 0.012 | 0.172 |
| PEGASUS-large | 0.220 | 0.040 | 0.148 | 0.015 | 0.165 |
| T5-large | 0.213 | 0.035 | 0.142 | 0.012 | 0.159 |

### TF-IDF Baseline on CNN/DailyMail

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|------|---------|---------|---------|------|--------|
| TF-IDF Extractive Baseline | 0.360 | 0.147 | 0.235 | 0.076 | 0.293 |

### BART Fine-Tuning on XSum

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|----------|---------|---------|---------|------|--------|
| Zero-shot on XSum | 0.210 | 0.036 | 0.138 | 0.012 | 0.177 |
| Fine-tuned on XSum | 0.352 | 0.153 | 0.264 | 0.061 | 0.347 |

---

## Important Observations

- BART and PEGASUS achieved the strongest CNN/DailyMail results after fine-tuning.
- T5-base experienced training collapse during fine-tuning, producing zero post-training scores.
- All transformer models showed a major performance drop when transferred zero-shot from CNN/DailyMail to XSum.
- T5-large showed the smallest ROUGE-1 percentage drop on XSum, suggesting stronger cross-domain generalization.
- BART achieved the highest METEOR score on XSum, showing that semantic-aware metrics can reveal performance differences missed by ROUGE.
- The TF-IDF baseline performed competitively on CNN/DailyMail, showing that the dataset favors lexical overlap and extractive-style summaries.
- Fine-tuning BART directly on XSum significantly improved all metrics, proving that domain adaptation is important for abstractive summarization.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NLTK
- ROUGE Score
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Conclusion

This project demonstrates how transformer-based models can be used for automated text summarization and news digest generation. It reproduces summarization experiments on CNN/DailyMail, evaluates cross-domain transfer on XSum, compares transformer models with a TF-IDF baseline, and shows the importance of domain-specific fine-tuning.

The results highlight that strong performance on one dataset does not always guarantee strong generalization to another dataset. Evaluation with multiple metrics is necessary because ROUGE, BLEU, and METEOR capture different aspects of summary quality.

---

## Citation

If this project is used for academic purposes, cite the original summarization paper, the datasets, and the transformer models used in this reproduction.
