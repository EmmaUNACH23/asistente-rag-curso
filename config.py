"""Configuración central del asistente RAG. Ajusta aquí los parámetros."""

CARPETA_DOCS = "documentos"      # donde pones tus PDF / TXT / MD
CARPETA_DB = "db"                # donde se guarda la base vectorial
NOMBRE_COLECCION = "curso"

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"  # funciona bien en español

# --- LLM: DeepSeek-R1 a través de Ollama -------------------------------------
# Equipo objetivo: Dell Latitude 5420, i5-1135G7 (4 núcleos), 16 GB RAM, sin GPU
# dedicada -> la inferencia corre en CPU. Opciones (todas vía `ollama pull`):
#   deepseek-r1:8b   5.2 GB  DeepSeek-R1-0528 destilado en Qwen3-8B. Mejor calidad y
#                            español; es la opción recomendada para 16 GB de RAM.
#   deepseek-r1:7b   4.7 GB  R1-Distill-Qwen-7B. Algo más ligero si vas justo de RAM.
#   deepseek-r1:1.5b 1.1 GB  Muy rápido pero poco fiable; solo para pruebas.
#   (el 14b pesa ~9 GB: en este equipo iría a ~2 tok/s y llenaría la RAM)
MODELO_LLM = "qwen3:4b"

# Los modelos R1 "piensan" antes de responder. En CPU eso puede tardar minutos, así que
# por defecto va desactivado; se puede activar desde la interfaz GTK.
RAZONAR_POR_DEFECTO = False

LLM_OPCIONES = {
    "temperature": 0.6,          # valor recomendado por DeepSeek para R1
    "num_ctx": 4096,             # contexto: suficiente para TOP_K fragmentos + pregunta
    "num_thread": 4,             # = núcleos físicos del i5-1135G7 (más hilos no ayuda)
}
LLM_KEEP_ALIVE = "30m"           # mantiene el modelo en RAM entre preguntas

TAM_FRAGMENTO = 800              # caracteres por fragmento
TRASLAPE = 100                   # caracteres compartidos entre fragmentos
TOP_K = 4                        # cuántos fragmentos se recuperan por pregunta
UMBRAL_DISTANCIA = 0.75          # si el mejor fragmento está más lejos que esto, se responde "no sé"
