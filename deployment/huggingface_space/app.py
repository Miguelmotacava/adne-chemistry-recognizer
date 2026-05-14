"""Gradio app para Hugging Face Spaces.

Sube los siguientes ficheros al Space (estructura plana en la raiz):
    app.py
    requirements.txt
    README.md
    best_model_handwritten.pt
    best_model_handwritten_config.json
    compounds.py
    src_augmentation.py  (renombrado de src/augmentation.py)
    src_models.py        (renombrado de src/models.py)

Y opcionalmente para el render de referencia (RDKit):
    src_generate_dataset.py (extracto de data/generate_dataset.py con
                              render_base, render_rdkit, render_formula_text)

El Space tiene que ser de tipo Gradio en la creacion.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

# --- Imports del modelo ----------------------------------------------------- #
from src_models import PretrainedModel
from src_augmentation import VAL_TRANSFORM
from compounds import COMPOUNDS, TAXONOMY, get_compounds

try:
    from src_generate_dataset import render_base
    HAS_RDKIT_RENDER = True
except Exception as e:
    HAS_RDKIT_RENDER = False
    print(f'Render RDKit no disponible: {e}')


# --- Setup ------------------------------------------------------------------ #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / 'best_model_handwritten_config.json'
CKPT_PATH = HERE / 'best_model_handwritten.pt'

CFG = json.loads(CFG_PATH.read_text(encoding='utf-8'))
NUM_CLASSES = CFG['num_classes']
CLASS_NAMES = CFG['class_names']
MODEL = PretrainedModel(backbone=CFG['backbone'],
                        num_classes=NUM_CLASSES,
                        strategy=CFG['strategy'])
MODEL.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
MODEL.eval().to(DEVICE)

COMPOUND_BY_ID = {c['id']: c for c in COMPOUNDS}
ALL_IDS = sorted(COMPOUND_BY_ID.keys())


# --- Preprocesado del dibujo ------------------------------------------------ #

def preprocess_drawing(pil: Image.Image) -> Image.Image:
    """Binariza, recorta al bounding box, centra en cuadrado y resize a 224."""
    g = np.array(pil.convert('L'))
    mask = g < 200
    if not mask.any():
        return pil.convert('RGB')
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    r0 = max(0, rmin - pad); r1 = min(g.shape[0], rmax + pad + 1)
    c0 = max(0, cmin - pad); c1 = min(g.shape[1], cmax + pad + 1)
    crop = g[r0:r1, c0:c1]
    crop = np.where(crop < 200, 0, 255).astype(np.uint8)
    h, w = crop.shape
    s = max(h, w)
    square = np.full((s, s), 255, dtype=np.uint8)
    y_off = (s - h) // 2; x_off = (s - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop
    return Image.fromarray(square).resize((224, 224), Image.LANCZOS).convert('RGB')


def predict(pil: Image.Image):
    arr = np.array(pil.convert('RGB'))
    x = VAL_TRANSFORM(image=arr)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    top5_idx = np.argsort(probs)[::-1][:5]
    return {CLASS_NAMES[i]: float(probs[i]) for i in top5_idx}


def get_reference_image(compound_id: str) -> Image.Image | None:
    if not HAS_RDKIT_RENDER:
        return None
    c = COMPOUND_BY_ID.get(compound_id)
    if c is None:
        return None
    try:
        img, _ = render_base(c)
        return img.resize((220, 220), Image.LANCZOS)
    except Exception:
        return None


# --- UI: modo dibujo libre -------------------------------------------------- #

def on_compound_change(compound_id: str):
    c = COMPOUND_BY_ID.get(compound_id)
    if c is None:
        return None, 'Selecciona un compuesto.', ''
    label = f"**{c['name_display']}** ({c['formula_display']}) — dificultad: {c['difficulty']}"
    if c.get('ionic', False):
        warning = ('⚠️ **Compuesto iónico — renderizado en el dataset como texto-fórmula con la fuente '
                   '`DejaVuSansMono-Bold`.** El modelo no fue entrenado a reconocer escritura manuscrita, '
                   'así que es muy probable que falle. Para una demo más fiable, prueba compuestos '
                   'orgánicos o inorgánicos con estructura 2D (óxidos, anhídridos, oxoácidos).')
    else:
        warning = ''
    return get_reference_image(compound_id), label, warning


def classify_drawing(sketch, compound_id: str):
    if sketch is None:
        return None, {}, 'Dibuja primero algo en el canvas.'
    # Gradio devuelve un dict con keys 'composite', 'background', 'layers'
    if isinstance(sketch, dict):
        img_arr = sketch.get('composite', sketch.get('image', None))
    else:
        img_arr = sketch
    if img_arr is None:
        return None, {}, 'Lienzo vacío.'
    if img_arr.shape[2] == 4:
        # Convertir RGBA a RGB sobre fondo blanco
        rgba = img_arr.astype(np.uint8)
        rgb = np.full(rgba.shape[:2] + (3,), 255, dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(float) / 255.0
        rgb = (rgba[:, :, :3] * alpha + rgb * (1 - alpha)).astype(np.uint8)
    else:
        rgb = img_arr.astype(np.uint8)
    pil = Image.fromarray(rgb)
    processed = preprocess_drawing(pil)
    top5 = predict(processed)

    c = COMPOUND_BY_ID.get(compound_id)
    expected = c['id'] if c else '—'
    pred = max(top5, key=top5.get)
    ok = (pred == expected)
    icon = '✅' if ok else '❌'
    msg = f"{icon}  Predicción: **{pred}** ({top5[pred]:.1%})  ·  Esperado: **{expected}**"
    return processed, top5, msg


# --- UI: modo "ver al modelo trabajar" sobre dataset embebido -------------- #
# Como no tenemos el dataset en el Space, generamos sobre la marcha con
# render_base. El modelo realmente reconoce estos renders al ~99%.

def quiz_random():
    if not HAS_RDKIT_RENDER:
        return None, {}, 'RDKit no disponible en el Space.'
    cid = random.choice(ALL_IDS)
    c = COMPOUND_BY_ID[cid]
    ref, _ = render_base(c)
    top5 = predict(ref)
    pred = max(top5, key=top5.get)
    ok = (pred == cid)
    icon = '✅' if ok else '❌'
    msg = (f"{icon}  Predicción: **{pred}** ({top5[pred]:.1%})  ·  "
           f"Compuesto real: **{cid}** — *{c['name_display']}* ({c['formula_display']})")
    return ref.resize((300, 300), Image.LANCZOS), top5, msg


# --- Layout ---------------------------------------------------------------- #

with gr.Blocks(title='Reconocedor de estructuras químicas') as demo:
    gr.Markdown("""
    # Reconocedor de estructuras químicas

    Demo del proyecto **ADNE — Análisis de Datos No Estructurados** (Máster en Big Data, ICAI).
    Reconoce 196 compuestos de bachillerato a partir de un dibujo. El modelo es un **ResNet18 fine-tuned**
    entrenado con augmentación agresiva (`HANDWRITTEN_TRAIN_TRANSFORM`); accuracy 98,9% en test del dataset.

    > **Aviso honesto:** el 98,9% es sobre los renders de RDKit aumentados con `Albumentations`,
    > no sobre escritura humana real. La demo funciona bien con compuestos orgánicos (estructura 2D);
    > falla con los iónicos (renderizados como texto-fórmula con fuente concreta).
    [Repositorio en GitHub](https://github.com/Miguelmotacava/adne-chemistry-recognizer)
    """)

    with gr.Tab('Dibujar'):
        with gr.Row():
            with gr.Column(scale=2):
                compound_dropdown = gr.Dropdown(
                    choices=ALL_IDS, label='Compuesto a dibujar',
                    value='metano', interactive=True)
                compound_label = gr.Markdown('')
                warning_md = gr.Markdown('')
                sketch = gr.Sketchpad(
                    label='Dibuja aquí (imita el render de referencia)',
                    type='numpy',
                    canvas_size=(400, 400),
                    brush=gr.Brush(default_size=4, colors=['#000000']),
                )
                btn_check = gr.Button('Comprobar', variant='primary')
            with gr.Column(scale=1):
                ref_image = gr.Image(label='Render de referencia (RDKit)',
                                     interactive=False, height=240)
                processed_view = gr.Image(label='Tu dibujo tras preprocesar (lo que ve el modelo)',
                                          interactive=False, height=240)
                top5_label = gr.Label(label='Top-5 confianzas', num_top_classes=5)
                result_md = gr.Markdown('')

        compound_dropdown.change(
            fn=on_compound_change, inputs=compound_dropdown,
            outputs=[ref_image, compound_label, warning_md])
        demo.load(
            fn=on_compound_change, inputs=compound_dropdown,
            outputs=[ref_image, compound_label, warning_md])
        btn_check.click(
            fn=classify_drawing, inputs=[sketch, compound_dropdown],
            outputs=[processed_view, top5_label, result_md])

    with gr.Tab('Ver al modelo trabajar (modo dataset)'):
        gr.Markdown("""Pulsa **Generar imagen aleatoria** para que aparezca un render de RDKit
        de un compuesto cualquiera. El modelo lo clasificará: aquí la accuracy real ronda el 99%.""")
        btn_random = gr.Button('Generar imagen aleatoria', variant='primary')
        with gr.Row():
            quiz_image = gr.Image(label='Imagen para el modelo',
                                  interactive=False, height=300)
            quiz_top5 = gr.Label(label='Top-5 confianzas', num_top_classes=5)
        quiz_result = gr.Markdown('')
        btn_random.click(fn=quiz_random, inputs=None,
                         outputs=[quiz_image, quiz_top5, quiz_result])


if __name__ == '__main__':
    demo.launch()
