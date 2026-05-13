"""
Dataset generator for the Chemistry Recognizer project.

CLI:
    python data/generate_dataset.py [OPTIONS]

For every selected compound it:
  1. Renders a base 224x224 image — either via RDKit 2D depiction or by
     drawing the formula_display string (for ionic / unparseable SMILES).
  2. Applies N augmented variants (Albumentations geometric + photometric).
  3. Saves PNGs to <output_dir>/<category>/<subcategory>/<compound_id>/img_NNNN.png
  4. Writes/overwrites data/metadata.csv with one row per generated image,
     including a stratified train/val/test split.

This script is STANDALONE (does not depend on src/ at import time) — but if
src/ is available it will use src.augmentation.AUGMENT_ONLY for the augmentation
pipeline; otherwise it falls back to a local equivalent.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Make repo root importable so we can do `from data.compounds import ...`
# and (optionally) `from src.augmentation import AUGMENT_ONLY`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.compounds import COMPOUNDS, TAXONOMY, get_compounds  # noqa: E402

# ---- Optional imports (RDKit + Albumentations) --------------------------- #
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except Exception as e:
    warnings.warn(f"RDKit not available ({e}); all compounds will use formula-text rendering.")
    RDKIT_AVAILABLE = False

try:
    from src.augmentation import AUGMENT_ONLY as _AUG
except Exception:
    import albumentations as A
    _AUG = A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.9),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.8),
        A.ElasticTransform(alpha=50, sigma=7, alpha_affine=7, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    ])

IMG_SIZE = 224


# ---------- Rendering helpers --------------------------------------------- #

def render_rdkit(smiles: str, size: int = IMG_SIZE) -> Optional[Image.Image]:
    """Try to render a SMILES as a 2D structure. Return None on failure."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try a few common monospace fonts; fall back to PIL default."""
    candidates = [
        "DejaVuSansMono-Bold.ttf",
        "DejaVuSansMono.ttf",
        "Consolas.ttf",
        "consola.ttf",
        "Courier New.ttf",
        "cour.ttf",
        "Arial.ttf",
        "arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_formula_text(formula_display: str, size: int = IMG_SIZE) -> Image.Image:
    """Render the formula string centred on a white 224x224 PIL image."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font_size = 48
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
            x_off, y_off = 0, 0
        if text_w <= size * 0.88 and text_h <= size * 0.7:
            break
        font_size -= 4
    x = (size - text_w) // 2 + x_off
    y = (size - text_h) // 2 + y_off
    draw.text((x, y), formula_display, fill="black", font=font)
    return img


def render_base(compound: Dict) -> Tuple[Image.Image, str]:
    """Render the base 224x224 image. Returns (image, render_mode)."""
    if compound.get("ionic", False):
        return render_formula_text(compound["formula_display"]), "formula_text"
    img = render_rdkit(compound["smiles"])
    if img is None:
        warnings.warn(
            f"SMILES not renderable for '{compound['id']}' "
            f"('{compound['smiles']}'); falling back to formula text."
        )
        return render_formula_text(compound["formula_display"]), "formula_text"
    return img, "rdkit_2d"


def augment(img: Image.Image) -> Image.Image:
    """Apply an Albumentations augmentation that DOES NOT normalize/to-tensor."""
    arr = np.array(img)
    out = _AUG(image=arr)["image"]
    return Image.fromarray(out)


# ---------- Split logic --------------------------------------------------- #

def stratified_indices(n: int,
                       val_split: float,
                       test_split: float,
                       seed: int) -> List[str]:
    """Return a list of length n with values 'train'|'val'|'test'."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_split))) if n >= 5 else max(0, n // 5)
    n_test = max(1, int(round(n * test_split))) if n >= 5 else max(0, n // 5)
    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n - 2)
        n_val = (n - n_train) // 2
        n_test = n - n_train - n_val
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    out = [""] * n
    for pos, original_idx in enumerate(idx):
        out[original_idx] = splits[pos]
    return out


# ---------- Main CLI ------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the chemistry image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--categories", default="all",
                   help="'inorganica', 'organica', or 'all'")
    p.add_argument("--subcategories", default=None,
                   help="Comma-separated subcategory slugs (default: all in selected category)")
    p.add_argument("--difficulty", default=None,
                   help="Comma-separated difficulties (basico,intermedio,avanzado)")
    p.add_argument("--n_per_class", type=int, default=300,
                   help="Number of images per compound (incl. the base image).")
    p.add_argument("--output_dir", default="data/raw",
                   help="Where to save generated images.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--metadata_path", default="data/metadata.csv")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the plan and exit without generating images.")
    return p.parse_args()


def select_compounds(args: argparse.Namespace) -> List[Dict]:
    cat = None if args.categories == "all" else args.categories
    subs = None
    if args.subcategories:
        subs = [s.strip() for s in args.subcategories.split(",") if s.strip()]
    diffs = None
    if args.difficulty:
        diffs = [s.strip() for s in args.difficulty.split(",") if s.strip()]
    return get_compounds(category=cat, subcategories=subs, difficulty=diffs)


def main() -> int:
    args = parse_args()
    compounds = select_compounds(args)
    if not compounds:
        print("[ERROR] No compounds matched the given filters.", file=sys.stderr)
        return 1

    out_root = (ROOT / args.output_dir) if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    metadata_path = (ROOT / args.metadata_path) if not Path(args.metadata_path).is_absolute() \
        else Path(args.metadata_path)

    print(f"=== Dataset Generation ===")
    print(f"  Output dir   : {out_root}")
    print(f"  Compounds    : {len(compounds)}")
    print(f"  Images/class : {args.n_per_class}")
    print(f"  Total images : {len(compounds) * args.n_per_class}")
    print(f"  Splits       : train={1 - args.val_split - args.test_split:.2f}"
          f"  val={args.val_split}  test={args.test_split}")
    print(f"  Seed         : {args.seed}")
    if args.dry_run:
        for c in compounds:
            print(f"    [{c['category']}/{c['subcategory']}] {c['id']:20s} {c['smiles']}")
        print("\n[dry_run] Exiting without writing files.")
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    per_subcat: Dict[str, Dict[str, int]] = {}

    for ci, c in enumerate(compounds, 1):
        cat = c["category"]; sub = c["subcategory"]; cid = c["id"]
        target_dir = out_root / cat / sub / cid
        target_dir.mkdir(parents=True, exist_ok=True)

        base_img, render_mode = render_base(c)

        n = max(1, int(args.n_per_class))
        splits = stratified_indices(n, args.val_split, args.test_split, args.seed + ci)

        for k in range(n):
            img = base_img if k == 0 else augment(base_img)
            fname = f"img_{k:04d}.png"
            fpath = target_dir / fname
            img.save(fpath, "PNG")
            rel = fpath.resolve().relative_to(ROOT.resolve())
            rows.append({
                "filepath": str(rel).replace("\\", "/"),
                "compound_id": cid,
                "name_es": c["name_es"],
                "name_display": c["name_display"],
                "formula": c["formula"],
                "formula_display": c["formula_display"],
                "category": cat,
                "subcategory": sub,
                "difficulty": c["difficulty"],
                "ionic": int(bool(c.get("ionic", False))),
                "split": splits[k],
                "render_mode": render_mode,
            })

        per_subcat.setdefault(cat, {}).setdefault(sub, 0)
        per_subcat[cat][sub] += 1

        if ci % 10 == 0 or ci == len(compounds):
            print(f"  [{ci}/{len(compounds)}] {cat}/{sub}/{cid}: {n} images ({render_mode})")

    # ---- Write metadata.csv -------------------------------------------- #
    cols = ["filepath", "compound_id", "name_es", "name_display",
            "formula", "formula_display", "category", "subcategory",
            "difficulty", "ionic", "split", "render_mode"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows to {metadata_path}")

    # ---- Summary table -------------------------------------------------- #
    from collections import Counter
    split_counts = Counter(r["split"] for r in rows)
    print("\nSummary:")
    print(f"  Total images       : {len(rows)}")
    print(f"  Train / Val / Test : "
          f"{split_counts.get('train',0)} / "
          f"{split_counts.get('val',0)} / "
          f"{split_counts.get('test',0)}")
    print("\n  Category | Subcategory                 | #Compounds | #Images")
    print("  " + "-" * 70)
    for cat, subs in per_subcat.items():
        for sub, ncomp in subs.items():
            imgs = sum(1 for r in rows if r["category"] == cat and r["subcategory"] == sub)
            print(f"  {cat:9s}| {sub:28s}| {ncomp:10d} | {imgs:7d}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
