"""Funciones de renderizado autocontenidas para el Space de Hugging Face.

Replica las funciones esenciales de data/generate_dataset.py:
    render_rdkit(smiles)            -> PIL.Image | None
    render_formula_text(formula)    -> PIL.Image
    render_base(compound_dict)      -> (PIL.Image, render_mode_str)
"""
from __future__ import annotations

import io
import warnings
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except Exception as e:
    warnings.warn(f'RDKit no disponible: {e}')
    RDKIT_AVAILABLE = False

IMG_SIZE = 224


def render_rdkit(smiles: str, size: int = IMG_SIZE) -> Optional[Image.Image]:
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_bytes)).convert('RGB')
    except Exception:
        return None


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ('DejaVuSansMono-Bold.ttf', 'DejaVuSansMono.ttf',
                 'Consolas.ttf', 'Courier New.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_formula_text(formula_display: str, size: int = IMG_SIZE) -> Image.Image:
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    font_size = 48
    text_w = text_h = x_off = y_off = 0
    font = None
    while font_size >= 14:
        font = _load_font(font_size)
        try:
            bbox = draw.textbbox((0, 0), formula_display, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x_off = -bbox[0]
            y_off = -bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(formula_display, font=font)
            x_off = y_off = 0
        if text_w <= size * 0.88 and text_h <= size * 0.7:
            break
        font_size -= 4
    x = (size - text_w) // 2 + x_off
    y = (size - text_h) // 2 + y_off
    draw.text((x, y), formula_display, fill='black', font=font)
    return img


def render_base(compound: Dict) -> Tuple[Image.Image, str]:
    if compound.get('ionic', False):
        return render_formula_text(compound['formula_display']), 'formula_text'
    img = render_rdkit(compound['smiles'])
    if img is None:
        return render_formula_text(compound['formula_display']), 'formula_text'
    return img, 'rdkit_2d'
