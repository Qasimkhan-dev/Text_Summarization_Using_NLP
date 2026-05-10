
# Fine-Tuning on XSum

This folder contains the fine-tuning experiment performed on the XSum dataset for the **Text Summarization Using NLP** project. The purpose of this experiment is to improve the summarization performance of transformer-based models by training them directly on XSum articles and their reference summaries.

Unlike zero-shot evaluation, where a pretrained model is used without additional training, fine-tuning allows the model to learn the specific structure and style of the XSum dataset. XSum summaries are usually short, highly concise, and focused on the main idea of a news article. Because of this, fine-tuning helps the model generate more accurate and dataset-specific abstractive summaries.

This experiment may use models such as BART, T5, or PEGASUS. The training process includes loading the XSum dataset, tokenizing articles and summaries, training the model, generating predictions, and evaluating the generated summaries using standard metrics such as ROUGE, BLEU, and METEOR.

This folder may include training scripts, saved model checkpoints, generated summaries, evaluation results, logs, and metric files. These outputs help analyze how fine-tuning improves summarization quality compared to zero-shot and TF-IDF baseline approaches.

Overall, this experiment demonstrates how transformer models can be adapted to a specific summarization dataset to produce better, more fluent, and more relevant summaries.
