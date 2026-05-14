---
title: Reconocedor de estructuras químicas
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Reconocedor de estructuras químicas

Demo del proyecto ADNE — Práctica 2 — Máster en Big Data, ICAI.

Reconoce 196 compuestos químicos (orgánicos + inorgánicos) a partir de un dibujo en el canvas.
Modelo: ResNet18 fine-tuned con augmentación agresiva (HANDWRITTEN_TRAIN_TRANSFORM), 98,9% de
accuracy en el test del dataset.

Código completo: https://github.com/Miguelmotacava/adne-chemistry-recognizer
