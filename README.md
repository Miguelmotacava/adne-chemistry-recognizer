# Reconocimiento de estructuras químicas

Práctica 2 — *Análisis de Datos No Estructurados* — ICAI, Máster en Big Data.

## De qué va el proyecto

El objetivo es construir, desde cero, un clasificador capaz de identificar compuestos químicos a partir de un dibujo a mano (o de una fórmula escrita) y ofrecer una pequeña aplicación interactiva donde un alumno de bachillerato pueda practicar formulación. Hemos elegido este caso porque cubre todos los puntos que pide la guía de la asignatura —EDA, ML clásico, DL discriminativo (from scratch + transfer learning), DL generativo, comparación con herramientas del mercado— y porque tiene una utilidad real fuera del aula: cualquier alumno de 1º de Bachillerato puede usarlo para auto-evaluarse.

El catálogo contiene unos 200 compuestos divididos en Química Inorgánica (óxidos, anhídridos, peróxidos, hidruros, sales, hidróxidos, oxoácidos, oxisales) y Química Orgánica (alcanos, alquenos, alquinos, cicloalcanos, aromáticos, halogenuros, alcoholes, éteres, aldehídos, cetonas, ácidos carboxílicos, ésteres, anhídridos, aminas, amidas, nitrilos).

## Requisitos

- Windows 10/11, macOS 12+ o Ubuntu 20.04+
- Conda (Miniconda o Anaconda) — necesario para que RDKit se instale sin dolor
- Python 3.10
- Opcional pero recomendado: GPU con CUDA (entrenar transfer learning en CPU es viable pero lento)
- ~5 GB libres si vas a generar el dataset completo

> **Nota sobre RDKit:** en muchas plataformas no hay wheel de pip estable, así que la instalación oficial es por conda-forge. Si no usas conda, salta a la sección *Sin conda* más abajo.

## Cómo arrancar

```bash
git clone <repo>
cd chemistry-recognizer

# Linux/macOS
bash scripts/setup_env.sh
# Windows
scripts\setup_env.bat

# o si lo prefieres manual:
conda env create -f environment.yml
conda activate chem-adne
pip install -e .
```

El script de setup crea el entorno, instala el paquete `src/` en modo editable, habilita los widgets de Jupyter y genera un dataset de prueba de 5 imágenes/clase para verificar que todo funciona.

### Generar el dataset completo

```bash
python data/generate_dataset.py --categories all --n_per_class 300
# o
make data-full
```

Tarda 15-25 minutos en un portátil normal. Renderiza con RDKit las estructuras 2D de los compuestos covalentes y dibuja como texto los iónicos. Para cada uno produce 300 variantes con `Albumentations` (rotación, ruido gaussiano, deformación elástica, brillo/contraste) y las guarda en `data/raw/<categoria>/<subcategoria>/<id>/`. El CSV de metadatos sale a `data/metadata.csv` con el split 70/15/15 ya hecho.

### Lanzar los notebooks

```bash
jupyter notebook
```

y abrir la carpeta `notebooks/` en el navegador.

## Cómo está organizado

```
chemistry-recognizer/
├── data/
│   ├── compounds.py          # catálogo + taxonomía (módulo standalone)
│   ├── generate_dataset.py   # CLI para generar imágenes y metadata.csv
│   ├── metadata.csv          # se crea al ejecutar lo anterior
│   └── raw/                  # imágenes generadas
├── src/                      # paquete instalable (pip install -e .)
│   ├── augmentation.py       # pipelines de Albumentations
│   ├── dataset.py            # ChemDataset + factory get_dataloaders
│   ├── models.py             # ChemCNN + PretrainedModel
│   ├── train.py              # bucle de entrenamiento (con AMP en GPU)
│   ├── evaluate.py           # métricas, curvas, matriz de confusión
│   └── vae.py                # Conditional VAE (notebook 04b)
├── notebooks/
│   ├── 00_dataset_generation.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_classical_ML.ipynb
│   ├── 03_CNN_scratch.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 04b_generative.ipynb        # CVAE — parte generativa
│   ├── 05_market_comparison.ipynb
│   └── 06_interactive_demo.ipynb
├── saved_models/
│   ├── best_model.pt
│   └── best_model_config.json
├── tests/                    # pytest — compounds, models, dataset
├── scripts/                  # setup_env, generate_full_dataset, run_all_notebooks
├── environment.yml
├── requirements.txt
├── setup.py
└── Makefile
```

## Resultados

Con el dataset completo (300 imágenes/clase, 58.800 imágenes, 196 clases) y el pipeline ejecutado de extremo a extremo en una GPU RTX 4060:

| Notebook | Modelo | Métrica | Resultado |
|---|---|---|---|
| 02 | SVM-linear + embeddings ResNet18 | test accuracy | 69,0% |
| 02 | SVM-linear + embeddings ResNet18 | test F1 macro | 0,683 |
| 03 | ChemCNN (4 bloques, Exp2) | best val acc | 98,2% |
| **04** | **ResNet18 fine-tune** | **test accuracy** | **99,4%** |
| **04** | **ResNet18 fine-tune** | **test F1 macro** | **0,994** |
| 04 | EfficientNet-B0 fine-tune | best val acc | 98,9% |
| 04b | Conditional VAE | best val loss | 156,4 (ELBO neg.) |

El modelo ganador (ResNet18 fine-tuned) está serializado en [`saved_models/best_model.pt`](saved_models/best_model.pt) + [`saved_models/best_model_config.json`](saved_models/best_model_config.json) y es el que carga el notebook 06 para la demo interactiva.

## Orden de ejecución de los notebooks

1. **`00_dataset_generation`** — UI para lanzar el generador y verificar que las imágenes generadas tienen sentido.
2. **`01_EDA`** — análisis exploratorio: distribución, desbalance (índice de Gini), proyección PCA/t-SNE y pares confundibles por SSIM. Las decisiones tomadas aquí justifican las elecciones de los notebooks siguientes.
3. **`02_classical_ML`** — baseline. Comparamos tres tipos de features (píxeles, HOG, embeddings de ResNet18 congelada) con cuatro modelos (SVM-lin, SVM-RBF, RandomForest, KNN). Sirve para saber qué techo razonable tiene el ML clásico antes de meterse en DL.
4. **`03_CNN_scratch`** — CNN desde cero (`ChemCNN`) con 5 experimentos progresivos para ver cómo afecta cada decisión (profundidad, dropout, augmentación, learning rate scheduler).
5. **`04_transfer_learning`** — comparativa de 4 configuraciones: ResNet18 y EfficientNet-B0, cada una en modo *feature extraction* y *fine-tuning*. El ganador se guarda en `saved_models/best_model.pt` para los notebooks 05 y 06.
6. **`04b_generative`** — Conditional VAE. La parte generativa del pipeline. Reconstrucción, generación condicional por clase e interpolación en el espacio latente.
7. **`05_market_comparison`** — comparativa contra DECIMER (OCSR open source) y Claude Sonnet (LLM con visión). Discusión sobre accuracy, latencia, coste y despliegue offline.
8. **`06_interactive_demo`** — la aplicación final para el alumno: canvas para dibujar, inferencia con el `best_model.pt`, marcador por subcategoría.

## Variables de entorno opcionales

```bash
# Necesario sólo para la sección de LLM en el notebook 05
export ANTHROPIC_API_KEY=sk-ant-...
```

Si no está definida, el notebook 05 omite la fila del LLM sin fallar.

## Notas técnicas

- **GPU**: el código detecta automáticamente CUDA. En GPU activa `torch.backends.cudnn.benchmark`, usa `torch.amp` (mixed precision) y sube `batch_size` a 128. En CPU usa batch 32 y precisión completa.
- **Dataset en disco vs. on-the-fly**: hemos optado por escribir las variantes aumentadas a disco. Es menos elegante pero ahorra ~30% de tiempo por época (la augmentación con `ElasticTransform` no es gratis).
- **`WeightedRandomSampler`** activado por defecto en `get_dataloaders()`. Con el dataset balanceado actual es redundante, pero queda como red de seguridad si en el futuro alguien filtra por subcategoría o dificultad.

## Sin conda

Si no quieres instalar conda:

```bash
python -m pip install -r requirements.txt
# Y aparte:
python -m pip install rdkit          # puede que no haya wheel para tu plataforma
```

En sistemas donde el wheel de `rdkit` para pip no funciona, no hay alternativa razonable: hay que usar conda.

## Tests

```bash
make test
# o
python -m pytest tests/ -v
```

`tests/test_compounds.py` valida los SMILES con RDKit y que los IDs son únicos. `tests/test_models.py` comprueba forward shapes de `ChemCNN` y `PretrainedModel` (todas las combinaciones backbone × strategy). `tests/test_dataset.py` carga el `DataLoader` y comprueba las shapes de un batch (se omite si no se ha generado el dataset todavía).

## Referencias

- [RDKit](https://www.rdkit.org/) — renderizado 2D y validación de SMILES.
- [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) — OCSR open source, usado como referencia en el notebook 05.
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/) — DL backend y modelos pre-entrenados (ResNet18, EfficientNet-B0).
- [Albumentations](https://albumentations.ai/) — pipeline de aumentación.
- [formulacionquimica.com](https://www.formulacionquimica.com/) — referencia para nomenclatura tradicional e IUPAC.

## Autor

Miguel Mota Cava — ICAI, Universidad Pontificia Comillas, 2025-2026.
