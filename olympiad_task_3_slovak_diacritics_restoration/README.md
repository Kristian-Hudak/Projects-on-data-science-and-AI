# Olympiad Task 3: Automated Slovak Diacritics Restoration with Transformers

**Competition:** Slovak National Olympiad in Artificial Intelligence (Olympiáda v umelej inteligencii — OAI)  
**Author:** Kristian Hudák  
**Round:** National Qualifier / Home Round

---

## Project Overview

In the Slovak language, diacritical marks (`á, ä, č, ď, é, í, ĺ, ľ, ň, ó, ô, ŕ, š, ť, ú, ý, ž`) are crucial for distinguishing lexical meaning and grammatical function (e.g. *sud* [barrel] vs. *súd* [courtroom]). Text entered in informal online contexts (SMS, chats, web searches) frequently omits diacritics altogether.

This project implements an end-to-end NLP system to automatically restore missing diacritics with high semantic and morphological precision.

---

## Pipeline & Directory Structure

```
olympiad_task_3_slovak_diacritics_restoration/
├── 01_data_preparation/
│   ├── corpus_extraction_and_cleaning.ipynb     # Wikipedia XML dump parsing & regex cleaning
│   ├── stats.json                               # Corpus descriptive metrics
│   └── vety_dataset.xlsx                        # Extracted parallel sentences dataset
├── 02_zeroshot_restoration/
│   ├── zeroshot_diacritics_restoration.ipynb    # Zero-shot SlovakBERT MLM inference
│   └── predictions_zeroshot.tsv                 # Validation predictions (~91.50% accuracy)
├── 03_finetuned_restoration/
│   ├── finetuned_diacritics_restoration.ipynb   # Domain-fine-tuned SlovakBERT with dictionary constraints
│   └── predictions_finetuned.tsv                # Validation predictions (~97.20% accuracy)
└── technical_report_diacritics_restoration.md   # Comprehensive technical report & ablation analysis
```

---

## Key Performance Results

Evaluated on the official validation dataset strictly across Slovak diacritic-bearing characters ($a, c, d, e, i, l, n, o, r, s, t, u, y, z$):

| Architecture / Technique | Character Accuracy | Key Advantage |
| :--- | :---: | :--- |
| **Zero-Shot SlovakBERT MLM** | ~91.50 % | Requires zero training; predicts valid grammatical forms |
| **Fine-Tuned SlovakBERT + Vocabulary Constraints** | **~97.20 %** | **+5.7% accuracy boost;** resolves complex homonyms and casing |

---

## Dataset & Checkpoint Access

The full Wikipedia evaluation sets and model fine-tuning outputs can be requested for local reproduction by reaching out on [LinkedIn](https://www.linkedin.com/in/kristi%C3%A1n-hud%C3%A1k-6260a7446).

