"""FASE 1: lee los documentos, los parte en fragmentos, calcula embeddings y los guarda en ChromaDB.

Uso:  python indexar.py
"""
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


def leer_documento(ruta: Path):
    """Devuelve una lista de (numero_de_pagina, texto)."""
    if ruta.suffix.lower() == ".pdf":
        paginas = []
        for n, pagina in enumerate(PdfReader(str(ruta)).pages, start=1):
            paginas.append((n, pagina.extract_text() or ""))
        return paginas
    # .txt y .md: se toman como una sola "página"
    return [(1, ruta.read_text(encoding="utf-8", errors="ignore"))]


def fragmentar(texto: str, tam: int, traslape: int):
    texto = " ".join(texto.split())  # limpia saltos de línea y espacios repetidos
    paso = max(1, tam - traslape)
    return [texto[i:i + tam] for i in range(0, len(texto), paso) if texto[i:i + tam].strip()]


def main(log=print):
    """Indexa todo. `log` recibe los mensajes de progreso (la GUI le pasa su propia función)."""
    carpeta = Path(config.CARPETA_DOCS)
    archivos = [p for p in sorted(carpeta.iterdir()) if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not archivos:
        log(f"No encontré documentos en '{carpeta}/'. Copia ahí tus PDF, TXT o MD y vuelve a correr.")
        return

    log("Cargando modelo de embeddings (la primera vez se descarga)...")
    modelo = SentenceTransformer(config.MODELO_EMBEDDINGS)

    cliente = chromadb.PersistentClient(path=config.CARPETA_DB)
    try:  # reinicia la colección para no duplicar fragmentos al reindexar
        cliente.delete_collection(config.NOMBRE_COLECCION)
    except Exception:
        pass
    coleccion = cliente.create_collection(config.NOMBRE_COLECCION, metadata={"hnsw:space": "cosine"})

    total = 0
    for archivo in archivos:
        ids, textos, metas = [], [], []
        for n_pag, texto in leer_documento(archivo):
            for j, frag in enumerate(fragmentar(texto, config.TAM_FRAGMENTO, config.TRASLAPE)):
                ids.append(f"{archivo.name}-p{n_pag}-{j}")
                textos.append(frag)
                metas.append({"archivo": archivo.name, "pagina": n_pag})
        if not textos:
            log(f"  - {archivo.name}: sin texto extraíble (¿PDF escaneado?), se omite")
            continue
        embeddings = modelo.encode(textos, show_progress_bar=False).tolist()
        coleccion.add(ids=ids, documents=textos, embeddings=embeddings, metadatas=metas)
        total += len(textos)
        log(f"  - {archivo.name}: {len(textos)} fragmentos")

    log(f"\nListo. {total} fragmentos indexados en '{config.CARPETA_DB}/'.")


if __name__ == "__main__":
    main()
