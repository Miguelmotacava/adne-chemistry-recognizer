"""Reentrena el modelo ResNet18 con augmentacion agresiva para que sea
mas robusto a dibujos a mano. Parte de ImageNet (no del checkpoint anterior)
para que el modelo aprenda directamente la distribucion expandida.

Guarda el modelo en saved_models/best_model_handwritten.pt + config.
"""
import sys
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (PretrainedModel, get_dataloaders, train_model,
                 full_evaluation_report, HANDWRITTEN_TRAIN_TRANSFORM, VAL_TRANSFORM)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
METADATA = ROOT / 'data' / 'metadata.csv'
MODELS = ROOT / 'saved_models'
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# OJO: num_workers=0 obligatorio en Windows para que el spawn del DataLoader
# no se quede colgado con las clases de src/dataset.py.
EPOCHS = 8
BATCH = 128 if DEVICE == 'cuda' else 32
NUM_WORKERS = 0

print(f'=== Reentreno con HANDWRITTEN_TRAIN_TRANSFORM ===')
print(f'Device: {DEVICE}, batch={BATCH}, epochs={EPOCHS}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

train_loader, val_loader, test_loader, class_names, _ = get_dataloaders(
    metadata_path=str(METADATA), batch_size=BATCH, num_workers=NUM_WORKERS,
    root_dir=ROOT,
    train_transform=HANDWRITTEN_TRAIN_TRANSFORM,
    eval_transform=VAL_TRANSFORM,
)
NUM_CLASSES = len(class_names)
print(f'NUM_CLASSES = {NUM_CLASSES}')
print(f'Batches train/val/test: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}')

# Mismo modelo que el notebook 04 ganador: ResNet18 fine-tune
model = PretrainedModel(backbone='resnet18', num_classes=NUM_CLASSES, strategy='finetune')
opt = torch.optim.Adam(model.get_optimizer_groups(lr_backbone=1e-5, lr_head=1e-3))
sched = CosineAnnealingLR(opt, T_max=EPOCHS)

save_path = str(MODELS / 'best_model_handwritten.pt')
t0 = time.time()
history = train_model(model, train_loader, val_loader, epochs=EPOCHS,
                      optimizer=opt, scheduler=sched, device=DEVICE,
                      patience=EPOCHS + 1, save_path=save_path,
                      experiment_name='handwritten-aug', verbose=True)
elapsed = time.time() - t0
print(f'\nEntrenamiento total: {elapsed/60:.1f} min, mejor epoch={history["best_epoch"]}')

# Cargar el mejor checkpoint y evaluar
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model = model.to(DEVICE).eval()

print('\nEvaluacion en los tres splits (con VAL_TRANSFORM):')
report = {}
# Re-construimos loaders sin sampler ni augmentation
train_loader_eval, val_loader_eval, test_loader_eval, _, _ = get_dataloaders(
    metadata_path=str(METADATA), batch_size=BATCH, num_workers=NUM_WORKERS,
    root_dir=ROOT,
    train_transform=VAL_TRANSFORM, eval_transform=VAL_TRANSFORM,
    use_weighted_sampler=False,
)
for split, loader in [('train', train_loader_eval),
                      ('val', val_loader_eval),
                      ('test', test_loader_eval)]:
    rep = full_evaluation_report(model, loader, class_names, device=DEVICE)
    report[split] = {'accuracy': rep['accuracy'], 'macro_f1': rep['macro_f1']}
    print(f'  {split:5s}: acc={rep["accuracy"]:.4f}  macro_F1={rep["macro_f1"]:.4f}')

cfg = {
    'architecture': 'PretrainedModel',
    'backbone': 'resnet18',
    'strategy': 'finetune',
    'num_classes': int(NUM_CLASSES),
    'class_names': list(class_names),
    'training': 'HANDWRITTEN_TRAIN_TRANSFORM (aug agresivo: rotacion 25, scale 0.7-1.3, elastic alpha 120, grid distortion, optical distortion, coarse dropout, ruido fuerte)',
    'epochs_trained': len(history['train_loss']),
    'best_epoch': history['best_epoch'],
    'train_accuracy': report['train']['accuracy'],
    'val_accuracy': report['val']['accuracy'],
    'test_accuracy': report['test']['accuracy'],
    'train_macro_f1': report['train']['macro_f1'],
    'val_macro_f1': report['val']['macro_f1'],
    'test_macro_f1': report['test']['macro_f1'],
}
with open(MODELS / 'best_model_handwritten_config.json', 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f'\nGuardado en {save_path} + best_model_handwritten_config.json')
