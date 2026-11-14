# Asistente de preguntas sobre documentos del curso (RAG)

RAG 100 % local: ChromaDB + embeddings multilingües + **DeepSeek-R1 (vía Ollama)**, con
interfaz de escritorio **GTK 3** (`app_gtk.py`) y la interfaz web Gradio original (`app.py`).

## Modelo elegido (Latitude 5420 · i5-1135G7 · 16 GB · sin GPU dedicada)
`deepseek-r1:8b` (5.2 GB, DeepSeek-R1-0528 destilado en Qwen3-8B). Corre en CPU; espera
unos 4-6 tokens/s. Si vas justo de RAM usa `deepseek-r1:7b` (4.7 GB), y no uses el 14b
(9 GB, ~2 tok/s). Se cambia en `config.py` → `MODELO_LLM`.
Ojo: tu neofetch muestra ~11.7 GB en uso; cierra el navegador y otras apps pesadas antes
de usarlo, porque el modelo necesita ~6 GB libres.

## 1. Instalar Ollama y el modelo (una sola vez)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:8b
```

## 2. Dependencias del sistema (GTK)
```bash
sudo apt install python3-gi gir1.2-gtk-3.0
```
(En Ubuntu Cinnamon normalmente ya vienen instalados.)

## 3. Entorno de Python
`--system-site-packages` es imprescindible: así el venv ve el PyGObject del sistema.
```bash
cd rag-curso
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU
pip install -r requirements.txt
```

## 4. Documentos e indexado
Copia tus PDF, TXT o MD a `documentos/` y ejecuta `python indexar.py`
(o usa el botón ↻ de la interfaz). La carpeta `db/` incluida ya está indexada y es
compatible: **no hace falta reindexar** para empezar.

## 5. Usar
```bash
python app_gtk.py   # interfaz de escritorio GTK
python app.py       # interfaz web (requiere: pip install gradio)
python rag.py       # terminal
```

### Interfaz GTK
- La respuesta aparece token a token; el botón **Detener** corta la generación.
- Interruptor **Razonar**: DeepSeek-R1 "piensa" antes de responder. Mejora preguntas
  complejas, pero en CPU añade minutos; por defecto está apagado (`RAZONAR_POR_DEFECTO`).
- Botones de cabecera: limpiar conversación, reindexar, abrir carpeta `documentos/`.
- La primera pregunta tras abrir la app es la más lenta: el modelo se precarga al inicio.

## Archivos
- `config.py`    parámetros (modelo, opciones del LLM, fragmentos, umbral)
- `indexar.py`   Fase 1: documentos -> fragmentos -> embeddings -> ChromaDB
- `rag.py`       Fase 2: pregunta -> búsqueda -> prompt -> LLM (streaming + razonamiento)
- `app_gtk.py`   interfaz GTK 3
- `app.py`       interfaz web con Gradio
