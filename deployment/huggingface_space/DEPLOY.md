# Despliegue a Hugging Face Spaces

Esta carpeta contiene **todo lo que necesita un Space de Hugging Face** para servir la demo en una URL pública gratuita.

## Despliegue paso a paso

### 1. Crear el Space en huggingface.co

1. Ir a https://huggingface.co (registro gratuito si no tienes cuenta).
2. Pulsar tu avatar → **New Space**.
3. Rellenar:
   - **Owner**: tu usuario
   - **Space name**: `adne-chemistry-recognizer` (o el que quieras)
   - **License**: MIT
   - **SDK**: **Gradio**
   - **Hardware**: CPU basic — free (suficiente; el modelo es ~45 MB)
   - **Visibility**: Public
4. Pulsar **Create Space**. HF crea un repositorio git vacío.

### 2. Instalar el CLI de Hugging Face (sólo la primera vez)

```bash
pip install huggingface_hub
huggingface-cli login
```

Te pedirá un *token* de acceso: lo creas en https://huggingface.co/settings/tokens con permisos de *Write*.

### 3. Clonar el Space y copiar los ficheros

Desde **el directorio raíz** del proyecto (no desde `deployment/`):

```bash
# Sustituye <USUARIO> por tu usuario de HF
git clone https://huggingface.co/spaces/<USUARIO>/adne-chemistry-recognizer hf_space_repo
cp deployment/huggingface_space/* hf_space_repo/
cd hf_space_repo
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add .
git commit -m "Initial commit"
git push
```

El push puede tardar 2-3 min porque hay que subir el modelo de 45 MB con Git LFS.

### 4. Esperar el build

En la página del Space verás un log de build. HF instalará todas las dependencias de `requirements.txt` (PyTorch + RDKit + Gradio), suele tardar 5-10 minutos la primera vez. Cuando termine aparece el botón **"App"** y la URL pública será:

```
https://huggingface.co/spaces/<USUARIO>/adne-chemistry-recognizer
```

Esa es la URL que puedes compartir con cualquiera para que pruebe la demo sin instalar nada.

## Estructura de ficheros en este folder

| Fichero | Origen | Para qué |
|---|---|---|
| `app.py` | nuevo | UI Gradio con sketchpad y modo dataset |
| `requirements.txt` | nuevo | Versiones pinned para el Space |
| `README.md` | nuevo | Metadata del Space (frontmatter YAML) + descripción |
| `best_model_handwritten.pt` | copiado de `saved_models/` | Pesos del modelo (45 MB) |
| `best_model_handwritten_config.json` | copiado | Configuración del modelo |
| `compounds.py` | copiado de `data/compounds.py` | Catálogo de 196 compuestos |
| `src_models.py` | copiado de `src/models.py` | Clase `PretrainedModel` |
| `src_augmentation.py` | copiado de `src/augmentation.py` | `VAL_TRANSFORM` |
| `src_generate_dataset.py` | reescrito | Funciones `render_base`, `render_rdkit`, `render_formula_text` autocontenidas |

Los renombres `src_*.py` son porque el Space tiene una estructura plana (todo en la raíz) y queríamos evitar colisión con el módulo `src/` del proyecto principal.

## Notas

- El modelo se carga una vez al arrancar el Space y se mantiene en memoria. La inferencia tarda ~50 ms por dibujo.
- El Space *queda dormido* tras 48 h de inactividad — al volver a acceder tarda 30-60 s en arrancar de nuevo.
- Si quieres GPU gratis durante un tiempo limitado, en la configuración del Space puedes activar el plan "Community GPU" (5 min/día, suficiente para una demo).
