#!/usr/bin/env python3
"""Interfaz de escritorio (GTK 3) para el asistente RAG.   Uso:  python app_gtk.py

GTK 3 (y no GTK 4) a propósito: es la versión que usan los temas de Cinnamon, así que la
ventana hereda tu tema (p. ej. Graphite-green-Dark) sin configurar nada.

Todo el trabajo pesado (embeddings, búsqueda, LLM) corre en hilos aparte; la interfaz nunca
se congela y la respuesta aparece token a token.
"""
import os
import re
import subprocess
import sys
import threading
import time

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import GLib, Gdk, Gtk, Pango  # noqa: E402

# Las rutas de config.py son relativas: las anclamos a la carpeta del proyecto para que
# funcione igual al lanzarlo desde un acceso directo.
RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)
sys.path.insert(0, RAIZ)

import config  # noqa: E402
import rag  # noqa: E402

CSS = b"""
.entrada-pregunta { padding: 8px; }
.barra-estado { padding: 4px 10px; opacity: 0.75; font-size: 90%; }
"""

RE_INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
RE_ENCABEZADO = re.compile(r"^(#{1,6})\s+(.*)$")
RE_VINETA = re.compile(r"^(\s*)([-*•]|\d+[.)])\s+(.*)$")


class VentanaRag(Gtk.Window):
    def __init__(self):
        super().__init__(title="Asistente del curso")
        self.set_default_size(920, 700)
        self.set_icon_name("system-help")

        self._listo = False          # modelos cargados
        self._ocupado = False        # hay una consulta o indexación en curso
        self._cancelar = threading.Event()

        self._construir_ui()
        self._aplicar_css()
        self.connect("destroy", self._al_cerrar)

        self._mensaje_sistema(
            f"Modelo de lenguaje: {config.MODELO_LLM} (Ollama).  "
            "Cargando modelo de embeddings y base vectorial…")
        threading.Thread(target=self._arranque, daemon=True).start()

    # ------------------------------------------------------------------ interfaz
    def _construir_ui(self):
        cabecera = Gtk.HeaderBar(show_close_button=True, title="Asistente del curso",
                                 subtitle=f"RAG local · {config.MODELO_LLM}")
        self.set_titlebar(cabecera)

        self.btn_limpiar = Gtk.Button.new_from_icon_name("edit-clear-all-symbolic", Gtk.IconSize.BUTTON)
        self.btn_limpiar.set_tooltip_text("Limpiar la conversación")
        self.btn_limpiar.connect("clicked", lambda *_: self.buffer.set_text(""))
        cabecera.pack_start(self.btn_limpiar)

        self.btn_reindexar = Gtk.Button.new_from_icon_name("view-refresh-symbolic", Gtk.IconSize.BUTTON)
        self.btn_reindexar.set_tooltip_text("Reindexar la carpeta documentos/ (hazlo si agregas o cambias archivos)")
        self.btn_reindexar.connect("clicked", self._al_reindexar)
        cabecera.pack_start(self.btn_reindexar)

        self.btn_carpeta = Gtk.Button.new_from_icon_name("folder-open-symbolic", Gtk.IconSize.BUTTON)
        self.btn_carpeta.set_tooltip_text("Abrir la carpeta documentos/")
        self.btn_carpeta.connect("clicked", self._al_abrir_carpeta)
        cabecera.pack_start(self.btn_carpeta)

        caja_razonar = Gtk.Box(spacing=6)
        etiqueta = Gtk.Label(label="Razonar")
        self.sw_razonar = Gtk.Switch(active=config.RAZONAR_POR_DEFECTO, valign=Gtk.Align.CENTER)
        self.sw_razonar.set_tooltip_text(
            "Activado: DeepSeek 'piensa' antes de responder (mejor en preguntas complejas,\n"
            "pero en CPU puede tardar varios minutos). Desactivado: respuesta directa.")
        caja_razonar.pack_start(etiqueta, False, False, 0)
        caja_razonar.pack_start(self.sw_razonar, False, False, 0)
        cabecera.pack_end(caja_razonar)

        raiz = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(raiz)

        # Conversación
        self.vista = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR,
                                  left_margin=16, right_margin=16, top_margin=12, bottom_margin=12,
                                  pixels_below_lines=3)
        self.buffer = self.vista.get_buffer()
        self._crear_tags()
        desplazable = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        desplazable.add(self.vista)
        self.desplazable = desplazable
        raiz.pack_start(desplazable, True, True, 0)

        raiz.pack_start(Gtk.Separator(), False, False, 0)

        # Entrada
        caja_entrada = Gtk.Box(spacing=8)
        caja_entrada.get_style_context().add_class("entrada-pregunta")
        self.entrada = Gtk.Entry(placeholder_text="Escribe tu pregunta y pulsa Enter…", hexpand=True)
        self.entrada.connect("activate", self._al_enviar)
        self.btn_enviar = Gtk.Button(label="Preguntar")
        self.btn_enviar.get_style_context().add_class("suggested-action")
        self.btn_enviar.connect("clicked", self._al_enviar)
        self.btn_detener = Gtk.Button(label="Detener")
        self.btn_detener.get_style_context().add_class("destructive-action")
        self.btn_detener.connect("clicked", lambda *_: self._cancelar.set())
        caja_entrada.pack_start(self.entrada, True, True, 0)
        caja_entrada.pack_start(self.btn_enviar, False, False, 0)
        caja_entrada.pack_start(self.btn_detener, False, False, 0)
        raiz.pack_start(caja_entrada, False, False, 0)

        # Estado
        self.estado = Gtk.Label(label="", xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.estado.get_style_context().add_class("barra-estado")
        raiz.pack_start(self.estado, False, False, 0)

        self._actualizar_sensibilidad()

    def _aplicar_css(self):
        proveedor = Gtk.CssProvider()
        proveedor.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), proveedor, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def _crear_tags(self):
        t = self.buffer.create_tag
        t("usuario", weight=Pango.Weight.BOLD, foreground="#4caf50", pixels_above_lines=14)
        t("asistente", weight=Pango.Weight.BOLD, foreground="#42a5f5", pixels_above_lines=14)
        t("sistema", style=Pango.Style.ITALIC, foreground="#9e9e9e", pixels_above_lines=8)
        t("error", foreground="#ef5350", pixels_above_lines=8)
        t("razon_titulo", weight=Pango.Weight.BOLD, foreground="#9e9e9e", scale=0.9, pixels_above_lines=6)
        t("razon", style=Pango.Style.ITALIC, foreground="#9e9e9e", scale=0.92,
          left_margin=28, right_margin=16)
        t("fuentes", foreground="#9e9e9e", scale=0.88, pixels_above_lines=6)
        t("negrita", weight=Pango.Weight.BOLD)
        t("cursiva", style=Pango.Style.ITALIC)
        t("codigo", family="monospace", background_rgba=Gdk.RGBA(0.5, 0.5, 0.5, 0.22))
        t("bloque_codigo", family="monospace", background_rgba=Gdk.RGBA(0.5, 0.5, 0.5, 0.22),
          left_margin=28, right_margin=16)
        for n, escala in ((1, 1.35), (2, 1.2), (3, 1.1)):
            t(f"h{n}", weight=Pango.Weight.BOLD, scale=escala, pixels_above_lines=8)
        t("vineta", left_margin=36, indent=-16)

    # ------------------------------------------------------------------ utilidades de texto
    def _fin(self):
        return self.buffer.get_end_iter()

    def _insertar(self, texto, *tags):
        if tags:
            self.buffer.insert_with_tags_by_name(self._fin(), texto, *tags)
        else:
            self.buffer.insert(self._fin(), texto)

    def _bajar(self):
        marca = self.buffer.get_mark("fin_scroll")
        if marca is None:
            marca = self.buffer.create_mark("fin_scroll", self._fin(), False)
        else:
            self.buffer.move_mark(marca, self._fin())
        self.vista.scroll_mark_onscreen(marca)

    def _mensaje_sistema(self, texto, tag="sistema"):
        self._insertar(texto + "\n", tag)
        self._bajar()

    def _render_inline(self, texto, *base):
        pos = 0
        for m in RE_INLINE.finditer(texto):
            if m.start() > pos:
                self._insertar(texto[pos:m.start()], *base)
            if m.group(1) is not None:
                self._insertar(m.group(1), "negrita", *base)
            elif m.group(2) is not None:
                self._insertar(m.group(2), "codigo", *base)
            else:
                self._insertar(m.group(3), "cursiva", *base)
            pos = m.end()
        if pos < len(texto):
            self._insertar(texto[pos:], *base)

    def _render_markdown(self, texto):
        """Markdown básico -> TextView (encabezados, listas, negrita, cursiva, código)."""
        en_codigo = False
        for linea in texto.strip("\n").split("\n"):
            if linea.strip().startswith("```"):
                en_codigo = not en_codigo
                continue
            if en_codigo:
                self._insertar(linea + "\n", "bloque_codigo")
                continue
            if (m := RE_ENCABEZADO.match(linea)):
                nivel = min(len(m.group(1)), 3)
                self._render_inline(m.group(2), f"h{nivel}")
            elif (m := RE_VINETA.match(linea)):
                simbolo = "•" if m.group(2) in "-*•" else m.group(2)
                self._insertar(f"{simbolo} ", "vineta")
                self._render_inline(m.group(3), "vineta")
            else:
                self._render_inline(linea)
            self._insertar("\n")

    # ------------------------------------------------------------------ estado de la UI
    def _actualizar_sensibilidad(self):
        libre = self._listo and not self._ocupado
        self.entrada.set_sensitive(libre)
        self.btn_enviar.set_sensitive(libre)
        self.btn_reindexar.set_sensitive(libre)
        self.btn_detener.set_visible(self._ocupado and self._listo)
        self.btn_enviar.set_visible(not (self._ocupado and self._listo))
        if libre:
            self.entrada.grab_focus()

    def _estado(self, texto):
        GLib.idle_add(self.estado.set_text, texto)

    def _ui(self, fn, *args):
        """Ejecuta fn(*args) en el hilo de GTK (devuelve False para que idle_add no repita)."""
        GLib.idle_add(lambda: (fn(*args), False)[1])

    # ------------------------------------------------------------------ arranque
    def _arranque(self):
        try:
            self._estado("Cargando modelo de embeddings y base vectorial…")
            rag.inicializar()
        except Exception as e:  # noqa: BLE001
            self._ui(self._mensaje_sistema,
                     f"No se pudo abrir la base vectorial ({type(e).__name__}: {e}).\n"
                     "Pulsa el botón de reindexar (↻) para crearla a partir de documentos/.", "error")
            self._ui(self._marcar_listo)
            self._estado("Sin base vectorial")
            return

        # Comprueba Ollama y que el modelo esté descargado; si lo está, lo precarga en RAM.
        try:
            import ollama
            instalados = {getattr(m, "model", None) or m.get("model") for m in ollama.list()["models"]}
            if config.MODELO_LLM not in instalados and f"{config.MODELO_LLM}:latest" not in instalados:
                self._ui(self._mensaje_sistema,
                         f"El modelo '{config.MODELO_LLM}' no está descargado. En una terminal ejecuta:\n"
                         f"    ollama pull {config.MODELO_LLM}", "error")
            else:
                self._estado(f"Cargando {config.MODELO_LLM} en memoria (unos segundos)…")
                ollama.generate(model=config.MODELO_LLM, prompt="", keep_alive=config.LLM_KEEP_ALIVE)
        except Exception as e:  # noqa: BLE001
            self._ui(self._mensaje_sistema,
                     "No pude conectar con Ollama. ¿Está en ejecución?  "
                     "(sudo systemctl start ollama)\n"
                     f"Detalle: {type(e).__name__}: {e}", "error")

        self._ui(self._mensaje_sistema, "Listo. Pregunta lo que quieras sobre tus documentos.")
        self._ui(self._marcar_listo)
        self._estado("Listo")

    def _marcar_listo(self):
        self._listo = True
        self._actualizar_sensibilidad()

    # ------------------------------------------------------------------ preguntar
    def _al_enviar(self, *_):
        pregunta = self.entrada.get_text().strip()
        if not pregunta or not self._listo or self._ocupado:
            return
        self.entrada.set_text("")
        self._ocupado = True
        self._cancelar.clear()
        self._actualizar_sensibilidad()

        self._insertar("Tú\n", "usuario")
        self._insertar(pregunta + "\n")
        self._insertar("Asistente\n", "asistente")
        self._bajar()

        self._turno = {"razon_ini": False, "resp_marca": None, "texto": "", "fuentes": []}
        razonar = self.sw_razonar.get_active()
        threading.Thread(target=self._consultar, args=(pregunta, razonar), daemon=True).start()

    def _consultar(self, pregunta, razonar):
        t0 = time.time()
        n_trozos = 0
        try:
            self._estado("Buscando en los documentos…")
            for tipo, dato in rag.preguntar_stream(pregunta, razonar=razonar):
                if self._cancelar.is_set():
                    self._ui(self._mensaje_sistema, "[detenido]")
                    break
                if tipo == "fuentes":
                    self._ui(self._turno.__setitem__, "fuentes", dato)
                    self._estado("DeepSeek está procesando el contexto…")
                elif tipo in ("razonamiento", "respuesta"):
                    n_trozos += 1
                    self._ui(self._pintar_trozo, tipo, dato)
                    seg = max(time.time() - t0, 0.001)
                    accion = "Razonando" if tipo == "razonamiento" else "Respondiendo"
                    self._estado(f"{accion}… {n_trozos / seg:.1f} tok/s aprox.")
        except Exception as e:  # noqa: BLE001
            self._ui(self._mensaje_sistema, self._explicar_error(e), "error")
        finally:
            self._ui(self._terminar_turno, time.time() - t0)

    @staticmethod
    def _explicar_error(e):
        nombre = type(e).__name__
        if "Connect" in nombre or isinstance(e, ConnectionError):
            return ("No pude conectar con Ollama. Inícialo con:  sudo systemctl start ollama\n"
                    f"({nombre}: {e})")
        if getattr(e, "status_code", None) == 404:
            return f"Falta el modelo. Descárgalo con:  ollama pull {config.MODELO_LLM}"
        return f"Error: {nombre}: {e}"

    def _pintar_trozo(self, tipo, texto):
        turno = self._turno
        if tipo == "razonamiento":
            if not turno["razon_ini"]:
                turno["razon_ini"] = True
                self._insertar("💭 Razonamiento\n", "razon_titulo")
            self._insertar(texto, "razon")
        else:
            if turno["resp_marca"] is None:
                texto = texto.lstrip()          # R1 suele abrir con saltos de línea
                if not texto:
                    return
                if turno["razon_ini"]:
                    self._insertar("\n")
                turno["resp_marca"] = self.buffer.create_mark(None, self._fin(), True)
            turno["texto"] += texto
            self._insertar(texto)
        self._bajar()

    def _terminar_turno(self, segundos):
        turno = self._turno
        if turno["resp_marca"] is not None:
            # Reemplaza el texto plano que fue llegando por su versión con formato Markdown
            ini = self.buffer.get_iter_at_mark(turno["resp_marca"])
            self.buffer.delete(ini, self._fin())
            self._render_markdown(turno["texto"])
            self.buffer.delete_mark(turno["resp_marca"])
        elif turno["razon_ini"] and not turno["texto"]:
            self._insertar("\n")
        if turno["fuentes"]:
            self._insertar("Fuentes: " + "; ".join(turno["fuentes"]) + "\n", "fuentes")
        self._bajar()
        self._ocupado = False
        self._actualizar_sensibilidad()
        self.estado.set_text(f"Listo · última respuesta en {segundos:.0f} s")
        return False

    # ------------------------------------------------------------------ reindexar / carpeta
    def _al_reindexar(self, *_):
        if self._ocupado:
            return
        self._ocupado = True
        self._actualizar_sensibilidad()
        self._mensaje_sistema("Reindexando documentos/ …")
        threading.Thread(target=self._reindexar, daemon=True).start()

    def _reindexar(self):
        try:
            import indexar
            self._estado("Reindexando…")
            indexar.main(log=lambda m: self._ui(self._mensaje_sistema, str(m).strip()))
            rag.reiniciar_coleccion()
            rag.inicializar()
            self._estado("Índice actualizado")
        except Exception as e:  # noqa: BLE001
            self._ui(self._mensaje_sistema, f"Error al indexar: {type(e).__name__}: {e}", "error")
            self._estado("Error al indexar")
        finally:
            def fin():
                self._ocupado = False
                self._actualizar_sensibilidad()
            self._ui(fin)

    def _al_abrir_carpeta(self, *_):
        os.makedirs(config.CARPETA_DOCS, exist_ok=True)
        subprocess.Popen(["xdg-open", os.path.abspath(config.CARPETA_DOCS)])

    def _al_cerrar(self, *_):
        self._cancelar.set()
        Gtk.main_quit()


def main():
    ventana = VentanaRag()
    ventana.show_all()
    ventana._actualizar_sensibilidad()   # show_all() vuelve visibles los botones ocultos
    Gtk.main()


if __name__ == "__main__":
    main()
