"""FASE 2: dada una pregunta, recupera fragmentos relevantes y le pide al LLM que responda con ellos.

Los modelos pesados (embeddings y base vectorial) se cargan con `inicializar()` y no al
importar el módulo, para que una interfaz gráfica pueda hacerlo en segundo plano.
"""
import re
import threading

import config

SIN_INFO = "No encontré esa información en los documentos del curso."

PROMPT_SISTEMA = (
    "Eres un asistente que ayuda a estudiantes a consultar el material de su curso. "
    "Responde en español, de forma clara y concisa, usando ÚNICAMENTE el contexto proporcionado. "
    f"Si la respuesta no está en el contexto, di exactamente: '{SIN_INFO}' "
    "No inventes datos."
)

_modelo = None
_coleccion = None
_bloqueo = threading.Lock()


def inicializar():
    """Carga el modelo de embeddings y abre la base vectorial (idempotente)."""
    global _modelo, _coleccion
    with _bloqueo:
        if _modelo is None:
            from sentence_transformers import SentenceTransformer
            _modelo = SentenceTransformer(config.MODELO_EMBEDDINGS)
        if _coleccion is None:
            import chromadb
            _coleccion = chromadb.PersistentClient(path=config.CARPETA_DB).get_collection(
                config.NOMBRE_COLECCION)


def reiniciar_coleccion():
    """Descarta la colección abierta (úsalo después de reindexar)."""
    global _coleccion
    with _bloqueo:
        _coleccion = None


def buscar(pregunta: str):
    """Devuelve (fragmentos, metadatos, distancias) de los TOP_K más parecidos."""
    inicializar()
    emb = _modelo.encode(pregunta).tolist()
    res = _coleccion.query(query_embeddings=[emb], n_results=config.TOP_K)
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def _mensajes(pregunta, fragmentos, metas):
    contexto = "\n\n---\n\n".join(
        f"[{m['archivo']}, pág. {m['pagina']}]\n{f}" for f, m in zip(fragmentos, metas)
    )
    return [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"},
    ]


def _fuentes(metas):
    return sorted({f"{m['archivo']} (pág. {m['pagina']})" for m in metas})


_RE_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def _quitar_think(texto: str) -> str:
    """Respaldo para versiones de Ollama que devuelven el razonamiento dentro de <think>."""
    return _RE_THINK.sub("", texto).strip()


def preguntar_stream(pregunta: str, razonar: bool = None):
    """Generador para interfaces: produce tuplas (tipo, dato).

    tipo = 'fuentes'      -> dato: lista de fuentes (se emite antes de generar)
           'razonamiento' -> dato: trozo del razonamiento interno del modelo
           'respuesta'    -> dato: trozo de la respuesta final
           'fin'          -> dato: None
    """
    import ollama

    razonar = config.RAZONAR_POR_DEFECTO if razonar is None else razonar
    fragmentos, metas, distancias = buscar(pregunta)

    # Si ni el mejor fragmento se parece a la pregunta, ni siquiera llamamos al LLM
    if not fragmentos or distancias[0] > config.UMBRAL_DISTANCIA:
        yield "respuesta", SIN_INFO
        yield "fin", None
        return

    yield "fuentes", _fuentes(metas)

    flujo = ollama.chat(
        model=config.MODELO_LLM,
        messages=_mensajes(pregunta, fragmentos, metas),
        stream=True,
        think=razonar,
        options=config.LLM_OPCIONES,
        keep_alive=config.LLM_KEEP_ALIVE,
    )
    en_think = False  # para el caso de <think> embebido en el contenido
    for trozo in flujo:
        msg = trozo["message"]
        pensamiento = getattr(msg, "thinking", None) or ""
        contenido = msg.content or ""
        if pensamiento:
            yield "razonamiento", pensamiento
        if contenido:
            if "<think>" in contenido:
                en_think = True
                contenido = contenido.replace("<think>", "")
            if "</think>" in contenido:
                antes, _, contenido = contenido.partition("</think>")
                if antes:
                    yield "razonamiento", antes
                en_think = False
            if contenido:
                yield ("razonamiento" if en_think else "respuesta"), contenido
    yield "fin", None


def preguntar(pregunta: str):
    """Devuelve (respuesta, lista_de_fuentes). Versión sin streaming (terminal / Gradio)."""
    respuesta, fuentes = "", []
    for tipo, dato in preguntar_stream(pregunta, razonar=False):
        if tipo == "fuentes":
            fuentes = dato
        elif tipo == "respuesta":
            respuesta += dato
    return _quitar_think(respuesta), fuentes


if __name__ == "__main__":  # prueba rápida en terminal
    print("Asistente del curso. Escribe 'salir' para terminar.\n")
    while True:
        p = input("Pregunta: ").strip()
        if p.lower() in {"salir", "exit", ""}:
            break
        resp, fuentes = preguntar(p)
        print(f"\n{resp}\n")
        if fuentes:
            print("Fuentes:", "; ".join(fuentes), "\n")
