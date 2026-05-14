"""Demo interactiva en Streamlit para Streamlit Cloud / Hugging Face Spaces.

Replica las dos secciones del notebook 06 — dibujo libre + ver al modelo
trabajar sobre el render canonico de RDKit — en una aplicacion web standalone
sin dependencias de Jupyter/ipycanvas.

Entrypoint para Streamlit Cloud (lo busca por defecto). Para correrlo local:
    streamlit run streamlit_app.py
"""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Permitimos importar el paquete src/ desde la raiz del repo
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import PretrainedModel, VAL_TRANSFORM
from data.compounds import COMPOUNDS, TAXONOMY, get_compounds
from data.generate_dataset import render_base


# ------------------------------------------------------------------ #
#  Configuracion de pagina y carga unica del modelo
# ------------------------------------------------------------------ #

st.set_page_config(page_title="Reconocedor de estructuras quimicas",
                   page_icon="🧪", layout="wide")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS_DIR = ROOT / 'saved_models'


@st.cache_resource(show_spinner='Cargando modelo en memoria...')
def load_model():
    """Carga el mejor modelo disponible. Prefiere el handwritten-aug."""
    candidates = [
        (MODELS_DIR / 'best_model_handwritten_config.json',
         MODELS_DIR / 'best_model_handwritten.pt',
         'handwritten-aug (robusto a dibujos a mano)'),
        (MODELS_DIR / 'best_model_config.json',
         MODELS_DIR / 'best_model.pt',
         'estandar (notebook 04)'),
    ]
    for cfg_path, ckpt_path, label in candidates:
        if cfg_path.exists() and ckpt_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            model = PretrainedModel(backbone=cfg['backbone'],
                                    num_classes=cfg['num_classes'],
                                    strategy=cfg['strategy'])
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.eval().to(DEVICE)
            return model, cfg['class_names'], cfg, label
    return None, None, None, None


MODEL, CLASS_NAMES, CFG, MODEL_LABEL = load_model()

if MODEL is None:
    st.error('No se ha encontrado ningun modelo entrenado en `saved_models/`. '
             'Ejecuta primero el notebook 04 o `scripts/retrain_handwritten.py`.')
    st.stop()

COMPOUND_BY_ID = {c['id']: c for c in COMPOUNDS}


# ------------------------------------------------------------------ #
#  Logica de preprocesado e inferencia
# ------------------------------------------------------------------ #

def preprocess_drawing(pil: Image.Image) -> Image.Image:
    """Binariza, recorta y centra el dibujo para que se parezca a un render
    de RDKit antes de pasarlo al modelo."""
    g = np.array(pil.convert('L'))
    mask = g < 200
    if not mask.any():
        return pil.convert('RGB').resize((224, 224))
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
    square[(s - h) // 2:(s - h) // 2 + h,
           (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(square).resize((224, 224), Image.LANCZOS).convert('RGB')


def predict(img: Image.Image):
    """Devuelve top-5 [(compound_id, prob), ...] para una imagen 224x224 RGB."""
    arr = np.array(img.convert('RGB'))
    x = VAL_TRANSFORM(image=arr)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top5_idx]


# ------------------------------------------------------------------ #
#  Cabecera + sidebar
# ------------------------------------------------------------------ #

st.title('🧪 Reconocedor de estructuras quimicas')
st.caption('Practica 2 — Analisis de Datos No Estructurados. Master en Big Data, ICAI.')

with st.sidebar:
    st.subheader('Modelo en uso')
    st.write(f'**Backbone**: {CFG["backbone"]}')
    st.write(f'**Estrategia**: {CFG["strategy"]}')
    st.write(f'**Clases**: {CFG["num_classes"]}')
    st.write(f'**Variante**: {MODEL_LABEL}')
    if 'test_accuracy' in CFG:
        st.metric('Test accuracy (dataset)', f'{CFG["test_accuracy"]*100:.2f}%')
    st.divider()
    st.markdown(
        '### Sobre la demo\n'
        'El modelo se entreno con renders generados por RDKit. Reconoce muy bien '
        'imagenes del dominio para el que fue entrenado (modo *Test*), pero su '
        'precision **cae** cuando se le pasan dibujos hechos a mano, sobre todo '
        'si el compuesto se renderizo originalmente como texto-formula. Esta '
        'limitacion esta documentada con detalle en el [README del repositorio]'
        '(https://github.com/Miguelmotacava/adne-chemistry-recognizer).'
    )


# ------------------------------------------------------------------ #
#  Pestañas
# ------------------------------------------------------------------ #

tab_test, tab_draw = st.tabs(['🔬 Test del modelo (RDKit)',
                              '✏️ Dibujar a mano'])


# ----- Tab 1: Test sobre RDKit render canonico ----- #

with tab_test:
    st.markdown('### El modelo trabajando sobre el dominio para el que fue entrenado')
    st.markdown(
        'Elige un compuesto del catalogo o pulsa **Sorprendeme** para uno aleatorio. '
        'Generamos su render canonico con RDKit y el modelo lo clasifica al instante. '
        'Aqui la accuracy es la real del modelo (~99%).'
    )

    cols = st.columns([2, 1])
    with cols[0]:
        compound_id = st.selectbox(
            'Compuesto',
            options=sorted(c['id'] for c in COMPOUNDS),
            format_func=lambda cid: f"{cid} — {COMPOUND_BY_ID[cid]['name_display']}",
            key='test_compound_select',
        )
    with cols[1]:
        st.write('')
        st.write('')
        if st.button('🎲 Sorprendeme', use_container_width=True):
            st.session_state['test_compound_select'] = random.choice(
                [c['id'] for c in COMPOUNDS])
            st.rerun()

    c = COMPOUND_BY_ID[compound_id]
    try:
        rendered_pil, _ = render_base(c)
    except Exception as e:
        st.error(f'No se pudo renderizar: {e}')
        st.stop()

    top5 = predict(rendered_pil)

    img_col, pred_col = st.columns([1, 1])
    with img_col:
        st.image(rendered_pil, caption=f'Render RDKit de {c["formula_display"]}',
                 width=300)
        st.markdown(f'**Nombre real**: {c["name_display"]}  ')
        st.markdown(f'**IUPAC**: {c["name_iupac"]}')

    with pred_col:
        st.subheader('Top-5 predicciones')
        pred = top5[0][0]
        ok = (pred == c['id'])
        if ok:
            st.success(f'✅ Predicho correctamente: **{pred}**')
        else:
            st.error(f'❌ Predicho: **{pred}** — esperado **{c["id"]}**')

        for i, (cid, prob) in enumerate(top5):
            label = COMPOUND_BY_ID.get(cid, {}).get('name_display', cid)
            st.write(f'**{i+1}. {cid}** — {label}')
            st.progress(prob, text=f'{prob*100:.1f}%')


# ----- Tab 2: Dibujar a mano ----- #

with tab_draw:
    st.markdown('### Dibuja un compuesto a mano')
    st.markdown(
        'Elige un compuesto, mira el render de referencia a la derecha e intenta '
        'reproducirlo en el lienzo. Despues pulsa **Comprobar**.\n\n'
        '⚠️ **Aviso honesto**: el modelo se entreno con renders de RDKit, no con '
        'dibujos a mano. La accuracy aqui es notablemente inferior a la del test. '
        'Funciona mejor con estructuras 2D (organicos, oxidos, anhidridos, oxoacidos) '
        'que con compuestos ionicos (sales, hidroxidos, hidracidos).'
    )

    try:
        from streamlit_drawable_canvas import st_canvas
        HAS_CANVAS = True
    except ImportError:
        HAS_CANVAS = False
        st.warning('`streamlit-drawable-canvas` no esta instalado. '
                   'Anadelo a requirements.txt para activar el lienzo de dibujo.')

    cols = st.columns([2, 1])
    with cols[0]:
        target_id = st.selectbox(
            'Compuesto a dibujar',
            options=sorted(c['id'] for c in COMPOUNDS),
            format_func=lambda cid: f"{cid} — {COMPOUND_BY_ID[cid]['name_display']}",
            key='draw_compound_select',
        )
    with cols[1]:
        st.write('')
        st.write('')
        if st.button('🎲 Aleatorio', key='draw_random', use_container_width=True):
            st.session_state['draw_compound_select'] = random.choice(
                [c['id'] for c in COMPOUNDS])
            st.rerun()

    target = COMPOUND_BY_ID[target_id]

    if target.get('ionic', False):
        st.warning(
            f'⚠️ **{target["formula_display"]}** se renderizo en el dataset como '
            f'texto-formula. El modelo aprendio a leer una fuente concreta, no '
            f'escritura manuscrita. Es muy probable que aqui falle. Si quieres '
            f'una demo mas fluida, prueba compuestos organicos o oxoacidos.'
        )

    draw_col, ref_col = st.columns([1, 1])

    with draw_col:
        st.markdown('#### 1. Lienzo')
        stroke_width = st.slider('Grosor del trazo', 1, 25, 4)

        if HAS_CANVAS:
            canvas_result = st_canvas(
                fill_color='rgba(255, 255, 255, 0)',
                stroke_width=stroke_width,
                stroke_color='#000000',
                background_color='#FFFFFF',
                update_streamlit=True,
                height=400, width=400,
                drawing_mode='freedraw',
                key=f'canvas_{target_id}',
            )
        else:
            canvas_result = None
            uploaded = st.file_uploader(
                'O sube una imagen', type=['png', 'jpg', 'jpeg'])

    with ref_col:
        st.markdown('#### 2. Render de referencia (RDKit)')
        try:
            ref_pil, _ = render_base(target)
            st.image(ref_pil, width=300,
                     caption=f'Asi lo dibuja RDKit en el dataset')
        except Exception as e:
            st.error(f'No se pudo renderizar el de referencia: {e}')

        st.markdown(f'**Nombre**: {target["name_display"]}  ')
        st.markdown(f'**Formula**: `{target["formula_display"]}`  ')
        st.markdown(f'**Dificultad**: `{target["difficulty"]}`')

    st.divider()
    if st.button('🔍 Comprobar', type='primary', use_container_width=True):
        raw_pil = None
        if HAS_CANVAS and canvas_result is not None and canvas_result.image_data is not None:
            arr = canvas_result.image_data.astype(np.uint8)
            # streamlit-drawable-canvas devuelve RGBA con fondo transparente —
            # componemos sobre blanco para que el modelo no vea negro alrededor.
            rgb = np.ones((arr.shape[0], arr.shape[1], 3), dtype=np.uint8) * 255
            alpha = arr[:, :, 3:4] / 255.0
            rgb = (arr[:, :, :3] * alpha + rgb * (1 - alpha)).astype(np.uint8)
            raw_pil = Image.fromarray(rgb)
        elif not HAS_CANVAS:
            if uploaded is not None:
                raw_pil = Image.open(uploaded)

        if raw_pil is None or np.array(raw_pil.convert('L')).min() > 240:
            st.warning('Dibuja algo en el lienzo antes de comprobar.')
        else:
            processed = preprocess_drawing(raw_pil)
            top5 = predict(processed)
            pred = top5[0][0]
            ok = (pred == target['id'])

            res_col, prev_col = st.columns([1, 1])
            with res_col:
                if ok:
                    st.success(f'✅ Acierto. El modelo dice **{pred}**, que es lo esperado.')
                else:
                    st.error(f'❌ Fallo. El modelo dice **{pred}**, esperado **{target["id"]}**.')

                st.markdown('**Top-5 predicciones**:')
                for i, (cid, prob) in enumerate(top5):
                    name = COMPOUND_BY_ID.get(cid, {}).get('name_display', cid)
                    st.write(f'{i+1}. **{cid}** — {name}')
                    st.progress(prob, text=f'{prob*100:.1f}%')

            with prev_col:
                st.markdown('**Tu dibujo tras preprocesar (224×224)**')
                st.image(processed, width=224,
                         caption='Esto es lo que ve el modelo')
