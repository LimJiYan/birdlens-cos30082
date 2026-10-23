---
title: BirdLens - Bird Species Classification
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
python_version: "3.12"
app_file: app.py
suggested_hardware: zero-a10g
---

# BirdLens — 200 Bird Species

COS30082 Applied Machine Learning: interactive demonstration of the selected
Model 8 (ImageNet-pretrained ResNet18 trained with Mixup; epoch 18).

Upload a bird photograph to view the predicted species and five ranked scores.
The model supports 200 species. Scores are not guarantees.

**Held-out test:** Top-1 58.55%, mean per-class accuracy 57.96% (1,204 images).

**Hardware:** Use the **ZeroGPU** hardware option when creating the Gradio Space.
The `suggested_hardware` field is not itself a hardware reservation.
