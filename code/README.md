# Code Folder

This folder contains the main implementation scripts for the **Text Summarization Using NLP** project. The code is used to run different summarization experiments, including traditional NLP baselines, zero-shot transfer, and transformer-based fine-tuning.

The scripts in this folder handle important steps such as dataset loading, text preprocessing, tokenization, model training, summary generation, and evaluation. They support experiments using models such as T5, BART, and PEGASUS, along with a TF-IDF baseline for comparison.

This folder may include code for:

- Loading summarization datasets such as CNN/DailyMail and XSum
- Preprocessing articles and reference summaries
- Running TF-IDF based extractive summarization
- Performing zero-shot summarization on XSum
- Fine-tuning transformer models on XSum
- Generating summaries from trained or pretrained models
- Evaluating summaries using ROUGE, BLEU, and METEOR
- Saving results, logs, and metric files

The code folder is the core part of the repository because it contains the actual logic used to reproduce and improve summarization experiments. Each script is organized to support a specific stage of the NLP pipeline, making the project easier to understand, run, debug, and extend.

Overall, this folder provides a structured implementation of automated text summarization using both traditional NLP methods and modern deep learning models.
