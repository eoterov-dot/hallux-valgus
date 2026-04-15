#!/usr/bin/env python3
"""
Medición de Ángulos — Hallux Valgus
Herramienta semi-asistida: 6 clics por pie → AHV y AIM 1-2 automáticos
Incluye repositorio de pacientes (RUT) descargable en CSV y Excel.

Uso:
    streamlit run auto_medicion_hv.py

Requiere:
    pip install streamlit streamlit-image-coordinates \
                opencv-python-headless numpy Pillow pandas openpyxl
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from datetime import datetime, date
import io, os, base64

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    COORDS_OK = True
except ImportError:
    COORDS_OK = False

# OCR: intentamos ocrmac (Vision de macOS, sin dependencias externas)
# y pytesseract como alternativa.
try:
    from ocrmac import OCR as _MacOCR
    OCR_ENGINE = "ocrmac"
    OCR_OK     = True
except ImportError:
    try:
        import pytesseract as _tess
        OCR_ENGINE = "tesseract"
        OCR_OK     = True
    except ImportError:
        OCR_ENGINE = None
        OCR_OK     = False

# ══════════════════════════════════════════════════════════════
# OCR — EXTRACCIÓN DE DATOS DESDE LA RADIOGRAFÍA
# ══════════════════════════════════════════════════════════════

def ocr_patient_info(img_pil: Image.Image):
    """
    Extrae nombre y RUT/ID del texto DICOM de la radiografía.
    Usa ocrmac (Vision de macOS) o pytesseract como fallback.
    Devuelve (nombre, rut) o (None, None).
    """
    if not OCR_OK:
        return None, None

    import re
    w, h = img_pil.size

    all_text = ""

    if OCR_ENGINE == "ocrmac":
        # ── Apple Vision (macOS nativo, sin instalaciones extra) ──
        try:
            import tempfile, os
            # Guardar temporalmente para ocrmac
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            img_pil.crop((0, 0, w, int(h * 0.15))).save(tmp_path)
            results = _MacOCR(tmp_path).recognize()
            os.unlink(tmp_path)
            # results = [(text, confidence, bbox), ...]
            all_text = "\n".join(r[0] for r in results)
        except Exception:
            all_text = ""

    else:
        # ── pytesseract (necesita brew install tesseract) ─────────
        import pytesseract as _tess
        gray = img_pil.convert("L")
        crops = [
            gray.crop((0,      0,        w // 2, int(h * 0.15))),
            gray.crop((w // 2, 0,        w,      int(h * 0.15))),
            gray.crop((0,      0,        w,      int(h * 0.15))),
        ]
        for crop in crops:
            arr = np.array(crop)
            if arr.mean() < 128:
                arr = 255 - arr
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            arr   = clahe.apply(arr)
            arr   = cv2.resize(arr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for psm in [6, 4, 11]:
                try:
                    all_text += "\n" + _tess.image_to_string(arr, config=f"--psm {psm} --oem 3")
                except Exception:
                    pass

    nombre, rut = None, None

    for line in all_text.split("\n"):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # ── Nombre: formato DICOM "APELLIDO^NOMBRE" ──────────
        # Tesseract 4 suele leer "^" como "-~", "~", "^~" u otras variantes.
        # Probamos todos los separadores conocidos.
        separadores = ["^", "-~", "^~", "~", " - "]
        sep_usado   = next((s for s in separadores if s in line), None)

        if sep_usado and not nombre:
            parts = line.split(sep_usado, 1)
            ap    = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", parts[0]).strip()
            nom   = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", parts[1]).strip() if len(parts) > 1 else ""
            candidate = f"{ap} {nom}".strip()
            if len(candidate) >= 6 and re.search(r"[A-Z]{3,}", candidate):
                nombre = candidate[:80]

        # ── Fallback: línea larga de solo mayúsculas = nombre ─
        if not nombre:
            solo_mayus = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", line).strip()
            if len(solo_mayus) >= 10 and solo_mayus == solo_mayus.upper():
                palabras = [p for p in solo_mayus.split() if len(p) >= 3]
                if len(palabras) >= 3:
                    nombre = " ".join(palabras)[:80]

        # ── RUT chileno con puntos: 12.345.678-9 ─────────────
        if not rut:
            m = re.search(r"\b(\d{1,2}\.\d{3}\.\d{3}[-][\dkK])\b", line)
            if m:
                rut = m.group(1)

        # ── RUT / ID sin puntos: 12345678-9 ──────────────────
        if not rut:
            m = re.search(r"\b(\d{6,8}[-][\dkK])\b", line)
            if m:
                rut = m.group(1)

        if nombre and rut:
            break

    return nombre, rut


# ══════════════════════════════════════════════════════════════
# CONSTANTES CLÍNICAS
# ══════════════════════════════════════════════════════════════

ANGLE_INFO = {
    "AHV":   {"name": "Ángulo de Hallux Valgus",        "normal": 15, "mild": 20, "moderate": 40, "color": "#f87171"},
    "AIM12": {"name": "Ángulo Intermetatarsiano 1-2",   "normal":  9, "mild": 11, "moderate": 16, "color": "#34d399"},
}

# Definición de los 6 puntos que el médico debe marcar
CLICK_STEPS = [
    {"idx": 0, "bone": "MT1", "color": "#f87171", "desc": "MT1 — punto PROXIMAL del 1er metatarsiano (inicio de la diáfisis)"},
    {"idx": 1, "bone": "MT1", "color": "#f87171", "desc": "MT1 — punto DISTAL del 1er metatarsiano (hacia la cabeza)"},
    {"idx": 2, "bone": "MT2", "color": "#34d399", "desc": "MT2 — punto PROXIMAL del 2do metatarsiano"},
    {"idx": 3, "bone": "MT2", "color": "#34d399", "desc": "MT2 — punto DISTAL del 2do metatarsiano"},
    {"idx": 4, "bone": "PF1", "color": "#60a5fa", "desc": "PF1 — BASE de la falange proximal del hallux (articulación MTF)"},
    {"idx": 5, "bone": "PF1", "color": "#60a5fa", "desc": "PF1 — EXTREMO DISTAL de la falange proximal del hallux"},
]

REPO_COLS = ["RUT", "Nombre", "Fecha", "Lateralidad", "Operado", "Pie",
             "AHV (°)", "AHV Clasificación",
             "AIM 1-2 (°)", "AIM 1-2 Clasificación", "Notas"]

REPO_FILE = "repositorio_hv.csv"


# ══════════════════════════════════════════════════════════════
# CÁLCULO DE ÁNGULOS
# ══════════════════════════════════════════════════════════════

def get_severity(angle, key):
    info = ANGLE_INFO.get(key, {})
    if angle is None or info.get("normal") is None:
        return "N/A", "#8b949e"
    if angle <= info["normal"]:    return "Normal",   "#56d364"
    if angle <= info["mild"]:      return "Leve",     "#f0a030"
    if angle <= info["moderate"]:  return "Moderado", "#f57f5b"
    return "Severo", "#f87171"


def calc_angle(p1, p2, p3, p4):
    """
    Ángulo agudo entre dos líneas:
      Línea 1: p1 → p2
      Línea 2: p3 → p4
    """
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)
    v2 = np.array([p4[0]-p3[0], p4[1]-p3[1]], dtype=float)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1 or m2 < 1:
        return None
    cos_a = float(np.clip(abs(np.dot(v1, v2) / (m1 * m2)), 0.0, 1.0))
    angle = np.degrees(np.arccos(cos_a))
    return round(min(angle, 180.0 - angle), 1)


# ══════════════════════════════════════════════════════════════
# SEPARACIÓN BILATERAL
# ══════════════════════════════════════════════════════════════

def crop_to_content(img, pad=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, th = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(th)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    hi, wi = img.shape[:2]
    return img[max(0,y-pad):min(hi,y+h+pad), max(0,x-pad):min(wi,x+w+pad)]


def detect_and_split_feet(img_bgr, pad=25):
    hi, wi = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    ksz = max(5, wi // 25)
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    fg_c = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    col  = np.sum(fg_c > 0, axis=0).astype(np.float32)
    ksm  = max(3, wi // 20)
    col_s = np.convolve(col, np.ones(ksm) / ksm, mode="same")
    q1, q3 = wi // 4, 3 * wi // 4
    mid = col_s[q1:q3]
    if mid.min() < col_s.max() * 0.20:
        sx   = q1 + int(np.argmin(mid))
        left  = crop_to_content(img_bgr[:, :sx+pad], pad)
        right = crop_to_content(img_bgr[:, max(0,sx-pad):], pad)
        return [("Pie Derecho (D)", left), ("Pie Izquierdo (I)", right)]
    return [("Pie", crop_to_content(img_bgr, pad))]


# ══════════════════════════════════════════════════════════════
# VISUALIZACIÓN: dibujar progreso sobre la imagen
# ══════════════════════════════════════════════════════════════

def draw_progress(foot_pil: Image.Image, points: list) -> Image.Image:
    """
    Dibuja los puntos y líneas acumulados sobre la imagen del pie.
    points = lista de (x, y) en coordenadas de la imagen original.
    """
    img    = foot_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw   = ImageDraw.Draw(overlay)

    bone_colors = {"MT1": (248, 113, 113), "MT2": (52, 211, 153), "PF1": (96, 165, 250)}

    for i, pt in enumerate(points):
        step = CLICK_STEPS[i]
        r, g, b = bone_colors[step["bone"]]
        x, y = int(pt[0]), int(pt[1])

        # Punto
        draw.ellipse([(x-8, y-8), (x+8, y+8)], fill=(r, g, b, 220))
        draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(255, 255, 255, 230))

        # Línea cuando tenemos el segundo punto del mismo hueso
        if i % 2 == 1:
            prev = points[i - 1]
            px, py = int(prev[0]), int(prev[1])
            draw.line([(px, py), (x, y)], fill=(r, g, b, 210), width=4)
            # Etiqueta en el centro de la línea
            mx, my = (px + x) // 2, (py + y) // 2
            draw.rectangle([(mx-22, my-13), (mx+22, my+13)], fill=(0, 0, 0, 160))
            draw.text((mx - 17, my - 10), step["bone"], fill=(r, g, b, 255))

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


# ══════════════════════════════════════════════════════════════
# REPOSITORIO
# ══════════════════════════════════════════════════════════════

def init_repo():
    if "repo" not in st.session_state:
        if os.path.exists(REPO_FILE):
            try:
                st.session_state["repo"] = pd.read_csv(REPO_FILE)
                return
            except Exception:
                pass
        st.session_state["repo"] = pd.DataFrame(columns=REPO_COLS)


def get_repo() -> pd.DataFrame:
    return st.session_state.get("repo", pd.DataFrame(columns=REPO_COLS))


def append_to_repo(row: dict):
    df  = get_repo()
    new = pd.DataFrame([{c: row.get(c, "") for c in REPO_COLS}])
    st.session_state["repo"] = pd.concat([df, new], ignore_index=True)
    try:
        st.session_state["repo"].to_csv(REPO_FILE, index=False)
    except Exception:
        pass


def repo_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hallux Valgus")
        ws = writer.sheets["Hallux Valgus"]
        for col_cells in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col_cells) + 3
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len, 40)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# MÓDULO DE MEDICIÓN POR PIE
# ══════════════════════════════════════════════════════════════

def measure_foot(foot_label: str, foot_bgr: np.ndarray,
                 state_key: str, rut: str, nombre: str,
                 exam_date, lateralidad: str, operado: str, notas: str):
    """
    Interfaz de medición para un pie.
    El médico hace 6 clics en orden: MT1×2, MT2×2, PF1×2.
    La app dibuja las líneas progresivamente y calcula los ángulos al final.
    """
    foot_pil  = Image.fromarray(cv2.cvtColor(foot_bgr, cv2.COLOR_BGR2RGB))
    orig_w, orig_h = foot_pil.size

    # Estado de esta medición
    if state_key not in st.session_state:
        st.session_state[state_key] = {"points": [], "saved": False}
    state  = st.session_state[state_key]
    points = state["points"]
    n      = len(points)

    st.markdown(f"### {foot_label}")
    st.caption(f"Imagen: {orig_w}×{orig_h} px")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        # ── Instrucción del paso actual ──────────────────────
        if n < 6:
            step = CLICK_STEPS[n]
            st.markdown(
                f'<div style="background:#161b22;padding:9px 13px;'
                f'border-radius:7px;border-left:3px solid {step["color"]};'
                f'font-size:0.88rem;color:{step["color"]};margin-bottom:8px">'
                f'👆 Clic {n+1}/6 — {step["desc"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0f2a18;padding:9px 13px;'
                'border-radius:7px;border-left:3px solid #56d364;'
                'font-size:0.88rem;color:#56d364;margin-bottom:8px">'
                '✅ Medición completa</div>',
                unsafe_allow_html=True,
            )

        # ── Imagen con progreso ───────────────────────────────
        img_annotated = draw_progress(foot_pil, points)

        if n < 6:
            # Escalar para visualización (máx 680px de ancho)
            max_w = 680
            if orig_w > max_w:
                scale = max_w / orig_w
                disp_w = max_w
                disp_h = int(orig_h * scale)
            else:
                disp_w, disp_h = orig_w, orig_h

            img_display = img_annotated.resize((disp_w, disp_h), Image.LANCZOS)

            coord = streamlit_image_coordinates(
                img_display,
                key=f"{state_key}_click_{n}",
            )

            if coord is not None:
                # Convertir coordenadas de pantalla → imagen original
                sx = orig_w / disp_w
                sy = orig_h / disp_h
                ix = int(coord["x"] * sx)
                iy = int(coord["y"] * sy)
                state["points"].append((ix, iy))
                st.rerun()
        else:
            # Mostrar imagen final sin widget de clics
            st.image(img_annotated, use_container_width=True)

        # Botón para reiniciar
        if n > 0:
            if st.button("🔄 Repetir medición", key=f"reset_{state_key}"):
                state["points"] = []
                state["saved"]  = False
                st.rerun()

    # ── Panel de resultados ───────────────────────────────────
    with right_col:
        st.markdown("#### Progreso")

        # Barra de progreso
        prog_pct = int(n / 6 * 100)
        st.progress(prog_pct, text=f"{n}/6 puntos marcados")

        # Estado de cada hueso
        for bone_key, color, idx_start in [("MT1","#f87171",0),("MT2","#34d399",2),("PF1","#60a5fa",4)]:
            pts_placed = sum(1 for j in range(idx_start, idx_start+2) if j < n)
            icons = {0: "⬜⬜", 1: "🟡⬜", 2: "✅✅"}
            st.markdown(
                f'<span style="color:{color}">●</span> '
                f'<span style="font-size:0.82rem"><b>{bone_key}</b> {icons[pts_placed]}</span>',
                unsafe_allow_html=True,
            )

        if n == 6:
            st.markdown("---")
            st.markdown("#### Resultados")

            # Extraer líneas
            mt1 = (points[0], points[1])
            mt2 = (points[2], points[3])
            pf1 = (points[4], points[5])

            ahv = calc_angle(mt1[0], mt1[1], pf1[0], pf1[1])
            aim = calc_angle(mt1[0], mt1[1], mt2[0], mt2[1])

            for angle, key in [(ahv, "AHV"), (aim, "AIM12")]:
                info = ANGLE_INFO[key]
                sev, sev_col = get_severity(angle, key)
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;
                            border-left:3px solid {info['color']};
                            border-radius:8px;padding:12px;
                            text-align:center;margin-bottom:10px">
                  <div style="font-size:0.72rem;color:#8b949e">{info['name']}</div>
                  <div style="font-size:2rem;font-weight:700;
                              color:{sev_col};margin:3px 0">{angle}°</div>
                  <div style="font-size:0.82rem;font-weight:600;
                              color:{sev_col}">{sev}</div>
                  <div style="font-size:0.68rem;color:#4b5563">
                    Normal ≤{info['normal']}°</div>
                </div>
                """, unsafe_allow_html=True)

            # Guardar
            st.markdown("---")
            if not rut.strip():
                st.warning("Ingresa el RUT en el panel lateral para guardar.")
            elif state.get("saved"):
                st.success("✅ Guardado en repositorio")
                if st.button("Guardar corrección", key=f"resave_{state_key}"):
                    state["saved"] = False
                    st.rerun()
            else:
                if st.button(f"💾 Guardar — {foot_label}",
                             type="primary",
                             key=f"save_{state_key}",
                             use_container_width=True):
                    append_to_repo({
                        "RUT":                  rut.strip(),
                        "Nombre":               nombre.strip(),
                        "Fecha":                exam_date.strftime("%Y-%m-%d"),
                        "Lateralidad":          lateralidad,
                        "Operado":              operado,
                        "Pie":                  foot_label,
                        "AHV (°)":              ahv,
                        "AHV Clasificación":    get_severity(ahv, "AHV")[0],
                        "AIM 1-2 (°)":          aim,
                        "AIM 1-2 Clasificación":get_severity(aim, "AIM12")[0],
                        "Notas":                notas.strip(),
                    })
                    state["saved"] = True
                    st.success("✅ Guardado")
                    st.rerun()


# ══════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Hallux Valgus — Medición",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] { background: #0d1117; }
      [data-testid="stSidebar"]          { background: #161b22; }
      h1,h2,h3,h4 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

    init_repo()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🦴 Hallux Valgus")
        st.markdown("### Datos del paciente")

        # ── Extracción automática desde la imagen ────────────
        if OCR_OK:
            if st.button("🔍 Extraer datos de la imagen",
                         help="Lee el texto de la radiografía para pre-llenar RUT y nombre",
                         use_container_width=True):
                st.session_state["ocr_trigger"] = True
        else:
            st.caption(
                "💡 Instala `pytesseract` y Tesseract para extraer "
                "datos automáticamente desde la radiografía."
            )

        # Campos del paciente (se pre-llenan si OCR encontró datos)
        rut    = st.text_input("RUT *",
                               value=st.session_state.get("ocr_rut", ""),
                               placeholder="12.345.678-9")
        nombre = st.text_input("Nombre",
                               value=st.session_state.get("ocr_nombre", ""),
                               placeholder="Apellido Nombre")
        fecha        = st.date_input("Fecha examen", value=date.today())
        lateralidad  = st.selectbox("Lateralidad afectada",
                                    ["Bilateral", "Derecho", "Izquierdo"],
                                    help="Lado clínicamente afectado")
        operado      = st.radio("¿Operado?", ["No", "Sí"],
                                horizontal=True,
                                help="¿El paciente tiene cirugía previa de HV?")
        notas        = st.text_area("Notas clínicas", height=80)

        st.markdown("---")
        st.markdown("### 📋 Protocolo")
        st.markdown("""
        **6 clics por pie en este orden:**

        <span style="color:#f87171">●</span> **MT1** — 2 clics en la diáfisis del 1er metatarsiano
        <span style="color:#34d399">●</span> **MT2** — 2 clics en la diáfisis del 2do metatarsiano
        <span style="color:#60a5fa">●</span> **PF1** — 2 clics en la diáfisis de la falange proximal del hallux

        Los clics definen el **eje longitudinal** de cada hueso.
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📐 Rangos normales")
        for k, v in ANGLE_INFO.items():
            st.markdown(f"**{k}** ≤{v['normal']}° Normal · ≤{v['mild']}° Leve · ≤{v['moderate']}° Mod.")

    # ── Tabs ─────────────────────────────────────────────────
    st.markdown("# 🦴 Medición de Ángulos — Hallux Valgus")
    tab_medir, tab_repo = st.tabs(["📏  Medir", "📊  Repositorio de pacientes"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — MEDIR
    # ════════════════════════════════════════════════════════
    with tab_medir:
        if not COORDS_OK:
            st.error("""
            **Instala el paquete de coordenadas:**
            ```
            pip3 install streamlit-image-coordinates
            ```
            Luego cierra y vuelve a abrir `Iniciar_App.command`.
            """)
            return

        uploaded = st.file_uploader(
            "Sube la radiografía AP del pie (unilateral o bilateral)",
            type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        )

        if not uploaded:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;
                        border-radius:12px;padding:60px;text-align:center;
                        margin-top:20px">
              <p style="font-size:3rem">📷</p>
              <p style="font-size:1rem;color:#8b949e">
                Sube una radiografía AP del pie para comenzar
              </p>
              <p style="font-size:0.78rem;color:#4b5563">
                JPG · PNG · TIFF · BMP  |  Unilateral o bilateral
              </p>
            </div>
            """, unsafe_allow_html=True)
            return

        img_pil = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_key = uploaded.name.replace(" ", "_").replace(".", "_")

        # ── OCR automático si el usuario lo pidió ─────────────
        if st.session_state.pop("ocr_trigger", False):
            with st.spinner("Leyendo datos de la imagen..."):
                ocr_nombre, ocr_rut = ocr_patient_info(img_pil)
            if ocr_nombre or ocr_rut:
                if ocr_nombre: st.session_state["ocr_nombre"] = ocr_nombre
                if ocr_rut:    st.session_state["ocr_rut"]    = ocr_rut
                st.success(
                    f"Datos encontrados → "
                    f"{'Nombre: ' + ocr_nombre if ocr_nombre else ''}"
                    f"{'  |  RUT: ' + ocr_rut if ocr_rut else ''}"
                )
                st.rerun()
            else:
                st.warning("No se encontraron datos legibles. Ingrésalos manualmente.")

        with st.spinner("Detectando pies..."):
            feet = detect_and_split_feet(img_bgr)

        n_feet = len(feet)
        st.success(
            f"{'Bilateral' if n_feet == 2 else 'Unilateral'} — "
            f"{n_feet} pie(s) detectado(s)"
        )

        for i, (foot_label_auto, foot_bgr) in enumerate(feet):
            st.markdown("---")

            # ── Selector de lateralidad por pie ───────────────
            lat_key = f"lat_{img_key}_{i}"
            opciones = ["Derecho", "Izquierdo", "No especificado"]

            # Pre-selección inteligente según el orden de detección
            # (izquierda de imagen = índice 0, derecha = índice 1)
            default_idx = 0 if i == 0 else 1

            lc, _ = st.columns([1, 3])
            with lc:
                lado = st.selectbox(
                    f"¿Qué pie es este? (imagen {i+1}/{n_feet})",
                    opciones,
                    index=default_idx,
                    key=lat_key,
                    help="Puedes cambiarlo si la imagen está como espejo",
                )

            foot_label = f"Pie {lado}"
            foot_id    = f"pie_{i}_{img_key}"
            state_key  = f"st_{img_key}_{foot_id}"

            measure_foot(
                foot_label=foot_label,
                foot_bgr=foot_bgr,
                state_key=state_key,
                rut=rut,
                nombre=nombre,
                exam_date=fecha,
                lateralidad=lateralidad,
                operado=operado,
                notas=notas,
            )

    # ════════════════════════════════════════════════════════
    # TAB 2 — REPOSITORIO
    # ════════════════════════════════════════════════════════
    with tab_repo:
        st.markdown("## 📊 Repositorio de pacientes")

        df = get_repo()

        # ── Importar CSV de sesión anterior ──────────────────
        with st.expander("📥 Importar datos de sesión anterior"):
            imp = st.file_uploader("CSV exportado previamente", type=["csv"], key="imp_csv")
            if imp:
                try:
                    df_imp = pd.read_csv(imp)
                    merged = pd.concat([df, df_imp], ignore_index=True).drop_duplicates()
                    st.session_state["repo"] = merged
                    try:
                        merged.to_csv(REPO_FILE, index=False)
                    except Exception:
                        pass
                    st.success(f"✅ Importados {len(df_imp)} registros. Total: {len(merged)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if df.empty:
            st.info("El repositorio está vacío. Realiza mediciones y guárdalas.")
            return

        # ── Filtros ───────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            rut_f = st.text_input("Buscar RUT", key="f_rut")
        with fc2:
            pie_f = st.selectbox("Pie", ["Todos","Pie Derecho (D)","Pie Izquierdo (I)","Pie"], key="f_pie")
        with fc3:
            cls_f = st.selectbox("AHV", ["Todos","Normal","Leve","Moderado","Severo"], key="f_cls")
        with fc4:
            op_f  = st.selectbox("Operado", ["Todos","No","Sí"], key="f_op")

        df_show = df.copy()
        if rut_f:
            df_show = df_show[df_show["RUT"].astype(str).str.contains(rut_f, na=False)]
        if pie_f != "Todos":
            df_show = df_show[df_show["Pie"] == pie_f]
        if cls_f != "Todos":
            df_show = df_show[df_show.get("AHV Clasificación", pd.Series()) == cls_f]
        if op_f != "Todos":
            df_show = df_show[df_show.get("Operado", pd.Series()) == op_f]

        # ── Estadísticas ──────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        ahv_vals  = pd.to_numeric(df_show.get("AHV (°)", pd.Series()),     errors="coerce").dropna()
        aim_vals  = pd.to_numeric(df_show.get("AIM 1-2 (°)", pd.Series()), errors="coerce").dropna()
        n_sev     = (df_show.get("AHV Clasificación", pd.Series()) == "Severo").sum()
        with s1: st.metric("Registros",     len(df_show))
        with s2: st.metric("AHV promedio",  f"{ahv_vals.mean():.1f}°"  if not ahv_vals.empty  else "—")
        with s3: st.metric("AIM 1-2 prom.", f"{aim_vals.mean():.1f}°"  if not aim_vals.empty  else "—")
        with s4: st.metric("AHV Severos",   int(n_sev))

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # ── Exportar ──────────────────────────────────────────
        st.markdown("### ⬇️ Exportar datos")
        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "📄 Descargar CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"hallux_valgus_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            try:
                xlsx = repo_to_excel(df)
                st.download_button(
                    "📊 Descargar Excel",
                    data=xlsx,
                    file_name=f"hallux_valgus_{datetime.now():%Y%m%d_%H%M}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                st.info("Instala openpyxl para exportar Excel: `pip3 install openpyxl`")

        # ── Gestión ───────────────────────────────────────────
        with st.expander("🗑️ Eliminar registros por RUT"):
            del_rut = st.text_input("RUT a eliminar", key="del_rut")
            if del_rut and st.button("Eliminar", type="secondary", key="btn_del"):
                before  = len(df)
                df_new  = df[df["RUT"].astype(str) != del_rut.strip()]
                st.session_state["repo"] = df_new
                try:
                    df_new.to_csv(REPO_FILE, index=False)
                except Exception:
                    pass
                st.warning(f"Eliminados {before - len(df_new)} registros del RUT {del_rut}")
                st.rerun()


if __name__ == "__main__":
    main()
