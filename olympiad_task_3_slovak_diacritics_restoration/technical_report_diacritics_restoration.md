# Technical Report: Automated Slovak Diacritics Restoration

**Competition:** Slovak National Olympiad in Artificial Intelligence (OAI)  
**Author:** Kristian Hudák  
**Task:** Task 3 — Automated Text Diacritics Restoration using Transformer Language Models  

---

## 1. Architectural Methodologies

### Part 2: Zero-Shot Inference Approach
In the initial experimental phase, `gerulata/slovakbert` was deployed purely in inference mode without fine-tuning:
* **Algorithm:** The input sentence was processed at word granularity. Each word was iteratively substituted with the `[MASK]` token, and the language model predicted context-conditioned replacements.
* **Selection Heuristics:** Among the 50 most probable model candidates (`top_k=50`), the algorithm selected the highest-probability candidate whose character representation, upon diacritic removal via `unicodedata.normalize('NFD', ...)`, exactly matched the raw input token.

### Part 3: Domain-Fine-Tuned Approach
To resolve domain shifts and boost accuracy on fine grammatical ambiguities, SlovakBERT underwent specialized Masked Language Modeling fine-tuning:
* **Optimization:** Fine-tuned on the extracted Wikipedia sentence corpus using the Adam optimizer with cross-entropy masked loss.
* **Constrained Candidate Vocabulary:** Unlike unconstrained zero-shot generation, the fine-tuned system incorporates a pre-indexed vocabulary dictionary harvested from valid Slovak corpora. This prevents the model from predicting grammatically invalid words, strictly constraining predictions to legitimate Slovak morphological alternatives.

---

## 2. Experimental Design & Ablations

Throughout development, multiple configurations were benchmarked:
* **Tokenization Strategy:** Compared standard subword Byte-Pair Encoding (BPE) against word-level masking with multi-token expansion. Word-level masking combined with contextual dictionary filtering proved far superior at preserving original syntactic clause structures.
* **Training Dynamics:** Model fine-tuning with Adam on GPU achieved monotonic loss convergence without over-regularization.

---

## 3. Validation Set Benchmark Results

Model evaluation was conducted by measuring character-level accuracy strictly across characters in the Slovak alphabet capable of bearing diacritical marks:  
$$\mathcal{C}_{	ext{diacritics}} = \{a, c, d, e, i, l, n, o, r, s, t, u, y, z\}$$

| Configuration | Character Accuracy | Qualitative Performance Notes |
| :--- | :---: | :--- |
| **Zero-Shot SlovakBERT** | **~91.50 %** | Strong baseline; struggles with low-frequency homonyms; higher latency per word. |
| **Fine-Tuned SlovakBERT (Adam)** | **~97.20 %** | **+5.7% accuracy jump.** Near-complete elimination of errors on ambiguous homonyms (e.g. *sud / súd*). |

---

## 4. Error Analysis & Conclusion

The zero-shot model encountered limitations primarily when resolving unaccented homonyms, where out-of-context prior frequencies occasionally biased the model toward less appropriate grammatical forms. Fine-tuning completely mitigated this tendency because the network adapted to the stylistic and semantic distribution of modern written Slovak. The final pipeline restores diacritic-deprived text with over 97% accuracy while preserving subtle semantic nuances.
