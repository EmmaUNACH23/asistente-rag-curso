"""Interfaz web con Gradio.  Uso:  python app.py   y abre http://127.0.0.1:7860"""
import gradio as gr

from rag import preguntar


def responder(pregunta):
    if not pregunta.strip():
        return "Escribe una pregunta.", ""
    respuesta, fuentes = preguntar(pregunta)
    texto_fuentes = "\n".join(f"• {f}" for f in fuentes) if fuentes else "—"
    return respuesta, texto_fuentes


with gr.Blocks(title="Asistente del curso") as demo:
    gr.Markdown("# 📚 Asistente de preguntas sobre documentos del curso\n"
                "Responde basándose únicamente en el material cargado (RAG).")
    entrada = gr.Textbox(label="Tu pregunta", placeholder="Ej: ¿Qué es un embedding?", lines=2)
    boton = gr.Button("Preguntar", variant="primary")
    salida = gr.Textbox(label="Respuesta", lines=8)
    fuentes = gr.Textbox(label="Fuentes consultadas", lines=3)

    boton.click(responder, inputs=entrada, outputs=[salida, fuentes])
    entrada.submit(responder, inputs=entrada, outputs=[salida, fuentes])

if __name__ == "__main__":
    demo.launch()
