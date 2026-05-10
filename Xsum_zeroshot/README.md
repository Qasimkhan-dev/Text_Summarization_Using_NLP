# XSum Zero-Shot Transfer

This folder contains the zero-shot transfer experiment performed on the XSum dataset for the **Text Summarization Using NLP** project. The purpose of this experiment is to evaluate how well a pretrained summarization model can generate summaries for a new dataset without being fine-tuned on that dataset.

In this setup, models such as T5, BART, or PEGASUS are used directly on XSum articles to generate abstractive summaries. The generated summaries are then compared with the reference summaries using standard evaluation metrics such as ROUGE, BLEU, and METEOR.

The XSum dataset is challenging because it requires highly concise, one-sentence summaries that capture the main idea of a news article. Zero-shot evaluation helps measure the generalization ability of a model and shows whether it can perform well on unseen summarization tasks without additional training.

This folder may include scripts, generated summaries, metric files, logs, and result outputs related to the XSum zero-shot experiment. These files help analyze model performance, compare results with fine-tuned models, and understand the limitations of zero-shot summarization.

Overall, this experiment demonstrates the ability of transformer-based NLP models to perform abstractive summarization on a new dataset without task-specific fine-tuning.
