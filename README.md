# Artificial Intelligence & Deep Learning Portfolio

**Author:** Kristian Hudák  
**Focus:** Deep Learning, Natural Language Processing, Computer Vision & Generative AI  
**Context:** Slovak National Olympiad in Artificial Intelligence (OAI) & Applied ML Projects  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Summary

This repository brings together a curated collection of machine learning, deep learning, computer vision, and NLP projects developed during the **Slovak National Olympiad in Artificial Intelligence (Olympiáda v umelej inteligencii — OAI)** and advanced AI coursework.

Every project includes clean, self-contained implementations, structured technical discussions, empirical evaluations, and rich visual outputs.

---

## Featured Competition Qualifier Projects

| Project | Domain | Architecture / Tech | Key Results |
| :--- | :--- | :--- | :--- |
| [**Olympiad Task 1: Recipe Nutrition & Calorie Prediction**](olympiad_task_1_nutritional_optimization/) | Tabular ML & Heuristics | Deep MLP vs Linear Regression, Fuzzy Matching (`thefuzz`), Multi-Objective Penalty Loss | Disproved additive linear models for multiplicative physics ($W \times D$); automated multi-attribute dietary substitution. |
| [**Olympiad Task 2: MNIST Digit Inpainting & Elastic Repair**](olympiad_task_2_mnist_digit_restoration/) | Computer Vision & Inpainting | Convolutional Autoencoder (`Conv2d`, `ConvTranspose2d`), Bézier curve modeling | Fine-tuned denoising autoencoder repairing 2,000 severely corrupted handwritten digits. |
| [**Olympiad Task 3: Slovak Diacritics Restoration**](olympiad_task_3_slovak_diacritics_restoration/) | NLP & Transformers | `gerulata/slovakbert` Masked Language Modeling, Wikipedia ETL, Constrained Decoding | **~97.20% character-level accuracy** (+5.7% over zero-shot baseline) across all Slovak diacritic characters. |

---

## Complete Project Catalog

### 1. Natural Language Processing & Transformers
- [**06_nlp_text_processing_nltk.ipynb**](06_nlp_text_processing_nltk.ipynb)  
  *Web Scraping, Punctuation Sanitization & Word Frequency Distribution.* Ingests full-text literature from Project Gutenberg, removes stopwords, and plots token frequency distributions with NLTK.
- [**07_tfidf_speech_similarity_search.ipynb**](07_tfidf_speech_similarity_search.ipynb)  
  *Corpus Scraping, TF-IDF Vectorization & Semantic Similarity Search.* Scrapes an online political speech repository, vectorizes text with TF-IDF, and computes pairwise Cosine Similarity matrices to find top semantic matches.
- [**09_transformers_slovak_nlp_classification_qa.ipynb**](09_transformers_slovak_nlp_classification_qa.ipynb)  
  *Slovak NLP with Transformers: Hate Speech Classification, Model Comparison, SK-QuAD QA & Error Analysis.* Fine-tunes `slovakbert`, benchmarks against `xlm-roberta` and multilingual `distilbert`, trains Slovak question answering, and provides a 50-sample error taxonomy.
- [**olympiad_task_3_slovak_diacritics_restoration/**](olympiad_task_3_slovak_diacritics_restoration/)  
  *Automated Diacritics Restoration with SlovakBERT.* Wikipedia data cleaning pipeline, zero-shot MLM inference (~91.5%), and fine-tuned contextual candidate-constrained decoder (~97.2%).

---

### 2. Computer Vision & Metric Learning
- [**04_cnn_image_binarization_weights.ipynb**](04_cnn_image_binarization_weights.ipynb)  
  *CNN Binarization & Weight Initialization Dynamics.* Handcrafted Heaviside step approximation ($W=10, b=-5$) contrasted with learning binarization boundaries from zero-initialized weights via backpropagation.
- [**05_variational_autoencoder_latent_space.ipynb**](05_variational_autoencoder_latent_space.ipynb)  
  *Convolutional Variational Autoencoder (CVAE) on MNIST.* Compressing digits into a 2D continuous latent manifold and generating a smooth 20x20 reconstructed digit grid.
- [**08_yolov3_tiny_object_detection.ipynb**](08_yolov3_tiny_object_detection.ipynb)  
  *Real-Time Video Object Detection with YOLOv3-Tiny.* Loading custom trained PyTorch weights (`OVLADACE_detector_yolov3_tiny.pth`) for frame-by-frame remote controller detection with Non-Maximum Suppression (NMS).
- [**11_vision_transformer_attention_segmentation.ipynb**](11_vision_transformer_attention_segmentation.ipynb)  
  *Vision Transformer (ViT-B/16) Attention Probing & Interactive Prompt-Based Segmentation.* Model surgery hooking into ViT attention maps, with an interactive JavaScript point-click interface and max-similarity patch feature aggregation for zero-shot foreground segmentation.
- [**14_dinov2_embeddings_few_shot_classification.ipynb**](14_dinov2_embeddings_few_shot_classification.ipynb)  
  *Self-Supervised DINOv2 ViT Embeddings & Nearest-Prototype Classification.* Segmenting handwriting variants with OpenCV connected components and performing few-shot centroid prototype classification in 384D embedding space.
- [**olympiad_task_2_mnist_digit_restoration/**](olympiad_task_2_mnist_digit_restoration/)  
  *Digit Inpainting Autoencoder.* Reconstructing 2,000 handwritten digits deformed by synthetic cubic Bézier rubbering curves.

---

### 3. Generative Adversarial Networks (GANs)
- [**10_gan_mnist_baseline.ipynb**](10_gan_mnist_baseline.ipynb)  
  *Balanced Learning Rates ($G_{lr} = 2 \times 10^{-4}, D_{lr} = 2 \times 10^{-4}$).* Minimax optimization, training loss trajectories, synthetic generation, and latent space linear interpolation ($z_1 \to z_2$).
- [**10_gan_mnist_fast_discriminator.ipynb**](10_gan_mnist_fast_discriminator.ipynb)  
  *Fast Discriminator Dynamics.* Analyzing gradient saturation and generator performance when the discriminator outpaces the generator.
- [**10_gan_mnist_fast_generator.ipynb**](10_gan_mnist_fast_generator.ipynb)  
  *Fast Generator Dynamics.* Examining mode collapse, discriminator loss degradation, and sample diversity under aggressive generator learning rates.

---

### 4. Classical Machine Learning & Optimization
- [**01_pandas_numpy_data_analysis.ipynb**](01_pandas_numpy_data_analysis.ipynb)  
  *Exploratory Data Analysis with Pandas & NumPy.* Groupby aggregations, Olympic records sorting, and vectorized grade statistics.
- [**02_polynomial_regression_pytorch_autograd.ipynb**](02_polynomial_regression_pytorch_autograd.ipynb)  
  *4th-Degree Polynomial Regression with PyTorch Autograd from Scratch.* Custom PyTorch Dataset, analytical autograd updates, and fitted polynomial regression plotting.
- [**03_deep_learning_hyperparameter_tuning.ipynb**](03_deep_learning_hyperparameter_tuning.ipynb)  
  *Deep Neural Network Depth, Width, and Hyperparameter Study.* Benchmarking 5 network architectures, activation functions (ReLU, Tanh, LeakyReLU), learning rates (0.01, 0.1, 1.0), and batch sizes (16, 32, 64) on football betting odds data.
- [**12_decision_trees_hyperparameter_optimization.ipynb**](12_decision_trees_hyperparameter_optimization.ipynb)  
  *Decision Tree Optimization & Splitting Criteria.* Evaluating depth regularization (`max_depth = 3, 8, 13`), Gini Impurity vs. Shannon Entropy, Graphviz DOT visualizations, and clinical confusion matrix metrics.
- [**13_chaid_decision_tree_chefboost.ipynb**](13_chaid_decision_tree_chefboost.ipynb)  
  *CHAID Decision Tree Regression with Chefboost.* Multi-way chi-squared decision tree regression and rule extraction on medical insurance expenses.

---

### 5. Autonomous Agents & Clinical AI Systems
- [**medical_qa_pipeline_llama_whisper.py**](medical_qa_pipeline_llama_whisper.py)  
  *Real-Time Medical Dialogue QA with 4-bit Llama-3.1-8B-Instruct & Faster-Whisper ASR.* End-to-end clinical pipeline that transcribes doctor-patient consultation audio with word timestamps, prompts quantized Llama-3.1 for binary QA, extracts precise evidence timestamps, and evaluates against strict latency benchmarks.
- [**build_a_coding_agent_from_scratch.ipynb**](build_a_coding_agent_from_scratch.ipynb)  
  *Building an Autonomous Coding Agent.* Implementing the four essential tools (`read_file`, `list_directory`, `edit_file`, `run_command`), an autonomous execution loop with system prompt conditioning, and multi-step code self-correction.

---

## Technical Skills Matrix

```
┌─────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Area                            │ Core Frameworks & Tools                                │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Deep Learning Frameworks        │ PyTorch, Torchvision, PyTorch Hub, Autograd            │
│ Pretrained Foundation Models    │ DINOv2 (Meta), ViT-B/16, Llama-3.1-8B, SlovakBERT      │
│ Speech & Audio AI               │ Faster-Whisper (ASR), CTC/Transformer speech decode    │
│ Computer Vision                 │ OpenCV, YOLOv3-Tiny, Morphological Image Processing   │
│ Natural Language Processing     │ Hugging Face Transformers, Tokenizers, NLTK, thefuzz   │
│ Classical Machine Learning      │ Scikit-Learn, Decision Trees, Chefboost (CHAID)        │
│ Generative Models               │ DCGAN / Deep GANs, Variational Autoencoders (VAE)      │
│ Data Engineering & Analysis     │ Pandas, NumPy, Scipy, BeautifulSoup, OpenPyXL          │
│ Visualization & Tooling         │ Matplotlib, Seaborn, Graphviz (DOT), Git, Google Colab │
└─────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## Notes on Running the Notebooks

Many notebooks were authored for execution in **Google Colab** with GPU acceleration:
- Paths beginning with `/content/` reflect standard Colab session storage environments.
- Dataset files used in the projects (e.g. `recepty(75).xlsx`, `priklad na zdravsie suroviny.csv`, `2000_fixed_digits.pt`) are included directly within their corresponding project directories.
- All notebooks have their pre-computed cell outputs, plots, and metrics preserved for immediate preview on GitHub without requiring re-execution.
 
---
 
## License
 
This project portfolio is licensed under the [MIT License](LICENSE) &copy; 2026 Kristian Hudák.
