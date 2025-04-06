# Empowering Vision: Multimodal Image Captioning & Audio Narration Web App

A deep learning-powered web application that generates *descriptive captions and audio narrations* from input images. Built to improve *accessibility for visually impaired users*, this project transforms visual data into rich, human-like narratives using advanced AI models.

---

## 🚀 Live Demo

*Web Application:*  
[Launch on Hugging Face Spaces](https://huggingface.co/spaces/ayushraj2349-2/IC)

---

## 💡 Project Motivation

The primary goal of this project is to *bridge the visual gap for the visually impaired* by converting static visual content (images) into meaningful, spoken descriptions. The system allows users to upload one or more images and receive natural language captions, which are then converted to **real-time audio using Google TTS**. This enables blind users to gain insights into visual content—photos, documents, surroundings—that they would not otherwise be able to perceive.

---

## 🧠 Project Overview

This system implements **Multimodal Image Captioning**—converting a single image into both *natural language text* and *spoken audio output*. The model architecture is based on a *fine-tuned CNN-LSTM with Attention* mechanism, using *Beam Search* decoding to generate fluent and contextually rich captions.

---

## ✨ Features

- *Dual Output:* Caption generation in both *text* and *audio* (via Google TTS)
- *Batch Upload Support:* Generate captions for up to *10 images* at once
- *Interactive Parameters:* Sliders for *beam width* and *maximum caption length*
- *Deployed Web App:* Built with *Gradio*, hosted on *Hugging Face Spaces*
- *Fully Notebook-Based:* Implementation and training logic entirely within *Colab notebooks*

---

## 📁 Main Notebook

| Notebook | Description |
|----------|-------------|
| [ic_with_beam_attention.ipynb](ic_with_beam_attention.ipynb) | Complete pipeline including preprocessing, training, inference, attention visualization, and TTS integration |

---

## 🏗 Model Architecture

- *Encoder:* Pre-trained *InceptionV3* CNN, fine-tuned on Flickr8k for feature extraction
- *Decoder:* *LSTM* with *Bahdanau Attention*, trained to generate context-aware captions
- *Decoding Method:* *Beam Search* (user-tunable) to improve caption quality
- *TTS Engine:* *Google Text-to-Speech (gTTS)* for converting text captions into audio

---

## 🧾 Dataset

- *Dataset Name:* Flickr8k  
- *Size:* 8,000+ images, each with 5 captions  
- *Preprocessing:* Tokenization, vocabulary filtering, padding, image normalization and transformation

---

## ⚙ Training Details

- *Epochs:* 100  
- *Optimizer:* Adam  
- *Regularization:* Label Smoothing, Data Augmentation  
- *LR Scheduler:* ReduceLROnPlateau  
- *Early Stopping:* Triggered on validation loss stagnation  
- *Checkpointing:* Best model saved when val loss improves  
- *Evaluation:* sacreBLEU score of *~22.5*

---

## 🧪 Deployment Stack

- *Frontend:* Gradio  
- *Backend:* PyTorch, gTTS  
- *Hosting:* Hugging Face Spaces  
- *Environment:* Google Colab Notebooks

---

## 🔮 Future Improvements

- Support for *multilingual captioning and audio synthesis*
- Explore *Transformer-based architectures* (e.g., ViT + GPT-2)
- Package into a *mobile-first PWA* for real-world assistive use
- Add *caption customization* (tone, verbosity, etc.)

---

## 🙏 Acknowledgements

- Inspired by the *"Show, Attend and Tell"* framework for attention-based image captioning.
- *Flickr8k dataset* used for training, courtesy of the University of Illinois at Urbana-Champaign.
- Model architecture adapted and customized from *TensorFlow, **PyTorch*, and research community examples.
- *Google Text-to-Speech (gTTS)* API used for converting generated captions to audio.
- Web deployment powered by *Gradio* and hosted on *Hugging Face Spaces*.

Special thanks to the open-source community for providing pre-trained models, datasets, and tooling that made this project possible.

---

## 🪪 License

This project is licensed under the *MIT License*.

---

## 🙋‍♂ Author

**Ayush Raj**

[![gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](ayushraj2349@gmail.com)
[![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayushraj2349)
[![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ayushraj2349)
[![kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/ayushraj2349)
[![medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@ayushraj2349)
