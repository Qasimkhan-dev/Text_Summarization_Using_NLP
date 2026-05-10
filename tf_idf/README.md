
# TF-IDF Baseline

This folder contains the TF-IDF baseline experiment for the **Text Summarization Using NLP** project. The purpose of this experiment is to compare transformer-based summarization models with a traditional NLP approach.

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It is used to identify important words in a document by measuring how frequently a word appears in one document and how rare it is across the complete dataset. Words with higher TF-IDF scores are considered more important.

In this baseline approach, the system selects the most important sentences from the original text based on TF-IDF scores. Unlike abstractive models such as T5, BART, or PEGASUS, this method performs extractive summarization, meaning it creates summaries by choosing existing sentences from the input document rather than generating new ones.

This folder may include scripts, generated summaries, evaluation results, and metric files related to the TF-IDF summarization experiment. The results are evaluated using standard metrics such as ROUGE, BLEU, and METEOR to compare the quality of TF-IDF summaries with deep learning-based summaries.

Overall, this experiment provides a simple and interpretable baseline for text summarization. It helps show the difference between traditional NLP techniques and modern transformer-based abstractive summarization models.
