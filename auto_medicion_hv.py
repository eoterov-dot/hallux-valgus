#!/usr/bin/env python3
"""
Medición Automática de Ángulos — Hallux Valgus
Versión con interfaz web Streamlit

Soporta:
  · Imágenes con un solo pie
  · Imágenes bilaterales (dos pies en la misma placa)
  · Modo lote: procesa múltiples radiografías a la vez

Uso:
    streamlit run auto_medicion_hv.py

Requiere:
    pip install streamlit opencv-python-headless numpy Pillow pandas
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
from datetime import datetime
import io
import zipfile

# ══════════════════════════════════════════════════════════════
# CONSTANTES CLÍNICAS
# ══════════════════════════════════════════════════════════════

ANGLE_INFO = {
    "AHV": {
        "name":     "Ángulo de Hallux Valgus",
        "desc":     "Eje 1er metatarsiano / Eje falange proximal",
        "normal":   15, "mild": 20, "moderate": 40,
        "color_bgr": (100, 120, 248), "color_hex": "#f87171",
    },
    "AIM12": {
        "name":     "Ángulo Intermetatarsiano 1-2",
        "desc":     "Eje 1er metatarsiano / Eje 2do metatarsiano",
        "normal":   9,  "mild": 11, "moderate": 16,
        "color_bgr": (80, 211, 130), "color_hex": "#34d399",
    },
    "AIM25": {
        "name":     "Ángulo Intermetatarsiano 2-5",
        "desc":     "Eje 2do metatarsiano / Eje 5to metatarsiano",
        "normal":   20, "mild": 25, "moderate": 35,
        "color_bgr": (251, 150, 80), "color_hex": "#fb923c",
    },
}

def get_severity(angle, info):
    if info.get("normal") is None:
        return "N/A", "#8b949e"
    if angle <= info["normal"]:    return "Normal",   "#56d364"
    if angle <= info["mild"]:      return "Leve",     "#f0a030"
    if angle <= info["moderate"]:  return "Moderado", "#f57f5b"
    return "Severo", "#f87171"


# ══════════════════════════════════════════════════════════════
# DETECCIÓN Y SEPARACIÓN DE PIES  ← NUEVO
# ══════════════════════════════════════════════════════════════

def crop_to_content(img: np.ndarray, pad: int = 30) -> np.ndarray:
    """Recorta la imagen al contenido no-negro más un margen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return img[y1:y2, x1:x2]


def detect_and_split_feet(img_bgr: np.ndarray, pad: int = 25):
    """
    Analiza si la imagen contiene uno o dos pies y los devuelve recortados.

    Para una placa bilateral (dos pies lado a lado con fondo negro) busca
    el "valle" vertical entre los dos pies mediante proyección de columnas.

    Retorna lista de (etiqueta, img_recortada).
    """
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Máscara del contenido visible (no-negro)
    _, fg_mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)

    # Suavizar para unir regiones cercanas
    close_sz = max(5, w_img // 25)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
    fg_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)

    # Proyección vertical: píxeles iluminados por columna
    col_sum = np.sum(fg_closed > 0, axis=0).astype(np.float32)

    # Suavizar proyección con convolución
    smooth_k = max(3, w_img // 20)
    kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
    col_smooth = np.convolve(col_sum, kernel, mode="same")

    # Buscar el valle sólo en el tercio central (donde estaría la separación)
    q1, q3 = w_img // 4, 3 * w_img // 4
    center_proj = col_smooth[q1:q3]
    valley_val   = center_proj.min()
    max_val      = col_smooth.max()

    # Si el valle es < 20 % del máximo → hay separación clara → dos pies
    two_feet = valley_val < max_val * 0.20

    if two_feet:
        split_x = q1 + int(np.argmin(center_proj))

        left_img  = img_bgr[:, :split_x + pad]
        right_img = img_bgr[:, max(0, split_x - pad):]

        left_crop  = crop_to_content(left_img,  pad)
        right_crop = crop_to_content(right_img, pad)

        # Convención radiológica AP: izquierda de imagen = pie DERECHO del paciente
        return [
            ("Pie Derecho (D)",    left_crop),
            ("Pie Izquierdo (I)",  right_crop),
        ]
    else:
        return [("Pie", crop_to_content(img_bgr, pad))]


# ══════════════════════════════════════════════════════════════
# PIPELINE DE VISIÓN ARTIFICIAL
# ══════════════════════════════════════════════════════════════

def preprocess(gray: np.ndarray) -> np.ndarray:
    norm     = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    return cv2.fastNlMeansDenoising(enhanced, h=10)


def segment(img_proc: np.ndarray) -> np.ndarray:
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    candidates = []
    for inv in [False, True]:
        src = 255 - img_proc if inv else img_proc
        _, otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adapt = cv2.adaptiveThreshold(
            src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
        for mask in [otsu, adapt]:
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  k3)
            candidates.append(m)

    def elongated_count(mask):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        c = 0
        for i in range(1, n):
            a  = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if a < 300: continue
            if max(bw, bh) / (min(bw, bh) + 1) > 1.8:
                c += 1
        return c

    return max(candidates, key=elongated_count)


def pca_axis(pts: np.ndarray):
    mean = pts.mean(axis=0)
    c    = pts - mean
    cov  = (c.T @ c) / len(pts)
    vals, vecs = np.linalg.eigh(cov)
    idx  = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    elong = np.sqrt(vals[0] / (vals[1] + 1e-9))
    return mean, vecs[:, 0], elong, vals


def extract_bones(mask: np.ndarray, min_area_frac=0.001, min_elong=2.5):
    """
    Detecta estructuras óseas elongadas con tres filtros calibrados:

    1. Área ≥ 0.1% imagen  → elimina caracteres de texto sueltos.
    2. Elongación ≥ 2.5    → solo estructuras alargadas tipo hueso.
    3. Orientación suave   → |axis_x| < 0.82 rechaza solo lo que es
       casi perfectamente horizontal (texto de cabecera de placa).
       Acepta huesos hasta ~55° desde la vertical, cubriendo incluso
       falanges muy desviadas en Hallux Valgus severo (40-50°).
    """
    h, w = mask.shape
    min_area = int(w * h * min_area_frac)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    bones = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area: continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        # Umbral bajo (1.2) porque un hueso inclinado 45° tiene bbox cuadrado
        # aunque su elongación real sea alta. La elongación PCA filtra mejor.
        if max(bw, bh) / (min(bw, bh) + 1) < 1.2: continue

        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        if len(pts) < 40: continue

        center, axis, elong, vals = pca_axis(pts)
        if elong < min_elong: continue

        # Rechaza solo estructuras casi puramente horizontales (texto cabecera).
        # Umbral 0.82 = permite ángulos hasta ~55° desde la vertical.
        if abs(axis[0]) >= 0.82:
            continue

        proj   = (pts - center) @ axis
        length = float(proj.max() - proj.min())

        bones.append({
            "center": center, "axis": axis,
            "elongation": elong, "length": length,
            "area": int(area), "label": "",
        })

    # Top 9 por longitud para evitar fragmentos residuales
    bones = sorted(bones, key=lambda b: b["length"], reverse=True)[:9]
    return bones


def classify_bones(bones: list, foot_label: str = ""):
    """
    Clasifica metatarsianos vs falanges usando posición Y (proximal/distal)
    y longitud relativa.  También ordena medial→lateral correctamente según
    si es pie derecho (hallux a la izquierda de la imagen) o izquierdo
    (hallux a la derecha).
    """
    if len(bones) < 2:
        return [], []

    # En AP dorsoplantar: falanges están MÁS ARRIBA (menor y = más distal)
    #                     metatarsianos están MÁS ABAJO (mayor y = más proximal)
    y_vals  = np.array([b["center"][1] for b in bones])
    y_split = np.percentile(y_vals, 45)   # punto de corte proximal/distal

    lengths    = np.array([b["length"] for b in bones])
    len_median = np.median(lengths)

    meta, phal = [], []
    for b in bones:
        long  = b["length"] >= len_median * 0.80
        lower = b["center"][1] >= y_split
        if long and lower:
            meta.append(b)
        elif not long and not lower:
            phal.append(b)
        elif long:
            meta.append(b)   # largo pero superior → base metatarsiano
        else:
            phal.append(b)   # corto inferior → falange de radio menor

    # Si no hay falanges, las más distales de los candidatos las asignamos
    if not phal and meta:
        meta_by_y = sorted(meta, key=lambda b: b["center"][1])
        phal = meta_by_y[:1]
        meta = meta_by_y[1:]

    # Ordenar medial → lateral
    # Pie DERECHO: hallux a la IZQUIERDA de la imagen → orden creciente de x
    # Pie IZQUIERDO: hallux a la DERECHA de la imagen → orden decreciente de x
    reverse = "Izquierdo" in foot_label or "(I)" in foot_label
    meta = sorted(meta, key=lambda b: b["center"][0], reverse=reverse)
    phal = sorted(phal, key=lambda b: b["center"][0], reverse=reverse)

    for i, b in enumerate(meta): b["label"] = f"MT{i+1}"
    for i, b in enumerate(phal): b["label"] = f"PF{i+1}"
    return meta, phal


def angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    d1 = d1 / (np.linalg.norm(d1) + 1e-9)
    d2 = d2 / (np.linalg.norm(d2) + 1e-9)
    cos_a = np.clip(abs(np.dot(d1, d2)), 0.0, 1.0)
    a = np.degrees(np.arccos(cos_a))
    return round(min(a, 180.0 - a), 1)


def confidence_score(b1, b2) -> float:
    return round(min(1.0, (b1["elongation"] + b2["elongation"]) / 16.0), 2)


def measure_angles(meta: list, phal: list) -> dict:
    results = {}
    if len(meta) >= 1 and len(phal) >= 1:
        results["AHV"] = {
            "angle": angle_between(meta[0]["axis"], phal[0]["axis"]),
            "bone1": meta[0], "bone2": phal[0],
            "confidence": confidence_score(meta[0], phal[0]),
        }
    if len(meta) >= 2:
        results["AIM12"] = {
            "angle": angle_between(meta[0]["axis"], meta[1]["axis"]),
            "bone1": meta[0], "bone2": meta[1],
            "confidence": confidence_score(meta[0], meta[1]),
        }
    if len(meta) >= 5:
        results["AIM25"] = {
            "angle": angle_between(meta[1]["axis"], meta[4]["axis"]),
            "bone1": meta[1], "bone2": meta[4],
            "confidence": confidence_score(meta[1], meta[4]),
        }
    return results


# ══════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════

def draw_overlay(img_bgr: np.ndarray, meta: list, phal: list,
                 measurements: dict, label: str = "") -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    def draw_axis(bone, color, thickness=1):
        c  = bone["center"].astype(int)
        hl = int(bone["length"] / 2)
        p1 = tuple(np.clip(c - bone["axis"] * hl, 0, [w-1, h-1]).astype(int))
        p2 = tuple(np.clip(c + bone["axis"] * hl, 0, [w-1, h-1]).astype(int))
        cv2.line(out, p1, p2, color, thickness)
        cv2.circle(out, tuple(c), 4, color, -1)
        cv2.putText(out, bone.get("label", ""), (c[0]+5, c[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    for b in meta: draw_axis(b, (160, 160, 160), 1)
    for b in phal: draw_axis(b, (110, 110, 110), 1)

    y_off = 20
    for key, m in measurements.items():
        info = ANGLE_INFO[key]
        col  = info["color_bgr"]
        sev, _ = get_severity(m["angle"], info)
        conf_pct = int(m["confidence"] * 100)

        draw_axis(m["bone1"], col, 2)
        draw_axis(m["bone2"], col, 2)

        txt = f"{key}: {m['angle']}  {sev}  (conf.{conf_pct}%)"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (7, y_off-th-3), (11+tw, y_off+4), (12, 12, 18), -1)
        cv2.rectangle(out, (7, y_off-th-3), (11+tw, y_off+4), col, 1)
        cv2.putText(out, txt, (9, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
        y_off += 28

    # Etiqueta del pie (D / I)
    if label:
        cv2.putText(out, label, (9, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1, cv2.LINE_AA)

    note = "RESULTADO AUTOMATICO - REQUIERE VERIFICACION CLINICA"
    (nw, nh), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(out, (7, h-28), (11+nw, h-8), (12, 12, 18), -1)
    cv2.putText(out, note, (9, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 200), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ══════════════════════════════════════════════════════════════

def analyze_single(img_bgr: np.ndarray, label: str = "Pie",
                   min_elongation: float = 2.5) -> dict:
    """
    Analiza un único pie recortado.

    Recorte al ANTEPIÉ (70 % superior):
      Los ángulos AHV y AIM solo necesitan metatarsianos y falanges.
      El calcáneo, tarsales y tobillo están en el tercio inferior y
      generaban falsos positivos que confundían al clasificador.
      Analizamos solo el 70 % superior y dibujamos sobre la imagen completa.
    """
    h, w = img_bgr.shape[:2]

    # ── Limitar al antepié ────────────────────────────────────
    forefoot_h  = int(h * 0.70)
    forefoot    = img_bgr[:forefoot_h, :]

    gray  = cv2.cvtColor(forefoot, cv2.COLOR_BGR2GRAY)
    proc  = preprocess(gray)
    mask  = segment(proc)
    bones = extract_bones(mask, min_elong=min_elongation)
    meta, phal = classify_bones(bones, foot_label=label)
    meas  = measure_angles(meta, phal)

    # Dibujar sobre el recorte del antepié y reincrustar en la imagen completa
    forefoot_ann = draw_overlay(forefoot, meta, phal, meas, label)
    img_r = img_bgr.copy()
    img_r[:forefoot_h, :] = forefoot_ann

    return {
        "label":        label,
        "measurements": meas,
        "meta":         meta,
        "phal":         phal,
        "img_result":   img_r,
        "bones_total":  len(bones),
    }


def analyze(img_bgr: np.ndarray, min_elongation: float = 2.0) -> list:
    """
    Pipeline completo.
    Detecta automáticamente si hay uno o dos pies y analiza cada uno.
    Devuelve lista de resultados (uno por pie).
    """
    feet = detect_and_split_feet(img_bgr)
    return [analyze_single(foot_img, lbl, min_elongation)
            for lbl, foot_img in feet]


# ══════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════

def render_foot_results(res: dict):
    """Muestra resultados de un pie individual."""
    meas     = res["measurements"]
    n_bones  = res["bones_total"]
    label    = res["label"]

    st.markdown(f"#### {label}")
    st.caption(f"Estructuras detectadas: {n_bones} "
               f"(MT: {len(res['meta'])} · PF: {len(res['phal'])})")

    if not meas:
        st.warning("No se calcularon ángulos para este pie. "
                   "Ajusta la elongación mínima o verifica la imagen.")
        return

    cols = st.columns(len(meas))
    for i, (key, m) in enumerate(meas.items()):
        info = ANGLE_INFO[key]
        sev, sev_col = get_severity(m["angle"], info)
        conf_pct = int(m["confidence"] * 100)
        with cols[i]:
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid {info['color_hex']}44;
                        border-left:3px solid {info['color_hex']};
                        border-radius:8px; padding:14px; text-align:center;">
              <div style="color:#8b949e; font-size:0.72rem;">{info['name']}</div>
              <div style="font-size:2rem; font-weight:700; color:{sev_col};
                          margin:4px 0">{m['angle']}°</div>
              <div style="font-size:0.82rem; font-weight:600;
                          color:{sev_col}">{sev}</div>
              <div style="font-size:0.67rem; color:#4b5563; margin-top:4px">
                Confianza: {conf_pct}%
              </div>
              <div style="font-size:0.67rem; color:#4b5563">
                Normal ≤{info['normal']}°
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Tabla
    rows = []
    for key, m in meas.items():
        info = ANGLE_INFO[key]
        sev, _ = get_severity(m["angle"], info)
        rows.append({
            "Ángulo":        key,
            "Nombre":        info["name"],
            "Valor (°)":     m["angle"],
            "Clasificación": sev,
            "Confianza (%)": int(m["confidence"] * 100),
            "Normal ≤":      f"{info['normal']}°",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Auto-Medición Hallux Valgus",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] { background: #0d1117; }
      [data-testid="stSidebar"]          { background: #161b22; }
      h1,h2,h3,h4 { color: #58a6ff; }
      .warn-box {
          background:#1c1a10; border:1px solid #6e5203;
          border-radius:8px; padding:12px 16px;
          color:#d4aa40; font-size:0.82rem; margin-top:12px;
      }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros")
        st.markdown("---")
        min_elong = st.slider(
            "Elongación mínima de hueso",
            min_value=1.5, max_value=6.0, value=2.5, step=0.1,
            help="2.5 recomendado. Sube si detecta ruido; baja si no encuentra huesos.",
        )
        st.markdown("---")
        st.markdown("### 📋 Rangos normales")
        for k, v in ANGLE_INFO.items():
            st.markdown(
                f"**{k}** — Normal ≤{v['normal']}° · "
                f"Leve ≤{v['mild']}° · Mod. ≤{v['moderate']}°"
            )
        st.markdown("---")
        st.markdown("""
        ### ℹ️ Acerca de esta herramienta
        Usa **visión artificial (OpenCV + PCA)** para detectar
        automáticamente los ejes óseos del pie.

        **Soporta:**
        - Radiografías de un solo pie
        - Radiografías **bilaterales** (dos pies en la misma placa)
        - Modo lote (múltiples imágenes a la vez)

        **Optimizado para:** radiografías AP dorsoplantares estándar.
        """)

    # ── Header ───────────────────────────────────────────────
    st.markdown("# 🦴 Medición Automática — Hallux Valgus")
    st.markdown(
        "Detección automática de ejes óseos · "
        "Soporta placas unilaterales y **bilaterales**"
    )
    st.markdown("---")

    tab1, tab2 = st.tabs([
        "📷 Análisis individual",
        "📁 Análisis por lote (múltiples imágenes)",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — IMAGEN INDIVIDUAL
    # ════════════════════════════════════════════════════════
    with tab1:
        uploaded = st.file_uploader(
            "Sube una radiografía del pie (unilateral o bilateral)",
            type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
            key="single",
        )

        if not uploaded:
            st.markdown("""
            <div style="background:#161b22; border:2px dashed #30363d;
                        border-radius:12px; padding:50px; text-align:center;
                        color:#4b5563; margin-top:20px">
              <p style="font-size:3rem">📷</p>
              <p style="font-size:1.05rem; color:#8b949e">
                Sube una radiografía para comenzar el análisis automático
              </p>
              <p style="font-size:0.78rem">
                JPG · PNG · TIFF · BMP · Unilateral o bilateral
              </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            img_pil = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Detectando pies y analizando…"):
                results = analyze(img_bgr, min_elongation=min_elong)

            n_feet = len(results)
            st.success(
                f"✅ {'Bilateral' if n_feet == 2 else 'Unilateral'} detectada — "
                f"analizando **{n_feet} pie{'s' if n_feet > 1 else ''}**"
            )

            # ── Imágenes ──────────────────────────────────────
            img_cols = st.columns(n_feet)
            for i, res in enumerate(results):
                with img_cols[i]:
                    st.subheader(res["label"])
                    st.image(
                        cv2.cvtColor(res["img_result"], cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

            st.markdown("---")

            # ── Resultados por pie ────────────────────────────
            if n_feet == 2:
                res_cols = st.columns(2)
                for i, res in enumerate(results):
                    with res_cols[i]:
                        render_foot_results(res)
            else:
                render_foot_results(results[0])

            # ── Descargas ─────────────────────────────────────
            st.markdown("---")
            st.subheader("⬇️ Descargar resultados")

            dl_cols = st.columns(n_feet + 1)

            # Una imagen anotada por pie
            for i, res in enumerate(results):
                with dl_cols[i]:
                    jpg = cv2.imencode(
                        ".jpg", res["img_result"],
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )[1].tobytes()
                    safe_label = res["label"].replace(" ", "_").replace("/","")
                    st.download_button(
                        f"Imagen — {res['label']}",
                        data=jpg,
                        file_name=f"{uploaded.name.rsplit('.',1)[0]}_{safe_label}.jpg",
                        mime="image/jpeg",
                    )

            # JSON con todos los resultados
            with dl_cols[-1]:
                report = {
                    "archivo":  uploaded.name,
                    "fecha":    datetime.now().isoformat(),
                    "tipo":     "bilateral" if n_feet == 2 else "unilateral",
                    "pies": [
                        {
                            "pie": res["label"],
                            "huesos_detectados": res["bones_total"],
                            "mediciones": {
                                k: {
                                    "nombre":        ANGLE_INFO[k]["name"],
                                    "angulo_grados": m["angle"],
                                    "severidad":     get_severity(m["angle"], ANGLE_INFO[k])[0],
                                    "confianza_pct": int(m["confidence"] * 100),
                                    "rango_normal":  f"≤{ANGLE_INFO[k]['normal']}°",
                                }
                                for k, m in res["measurements"].items()
                            },
                        }
                        for res in results
                    ],
                }
                st.download_button(
                    "Reporte JSON (ambos pies)",
                    data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{uploaded.name.rsplit('.',1)[0]}_reporte.json",
                    mime="application/json",
                )

            st.markdown("""
            <div class="warn-box">
              ⚠️ <strong>Verificación obligatoria:</strong>
              Los valores son estimaciones automáticas. Revisar visualmente
              los ejes detectados antes de registrar en el estudio.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — LOTE
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            "Sube varias radiografías a la vez (unilaterales y/o bilaterales). "
            "El sistema las analiza todas y genera un resumen en CSV + ZIP."
        )

        batch_files = st.file_uploader(
            "Selecciona múltiples radiografías",
            type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
            accept_multiple_files=True,
            key="batch",
        )

        if not batch_files:
            st.info("Selecciona una o más radiografías para el análisis por lote.")
        else:
            if st.button(f"🚀 Analizar {len(batch_files)} imagen(es)", type="primary"):
                all_rows   = []
                zip_images = []
                progress   = st.progress(0, text="Procesando…")
                status     = st.empty()

                for idx, f in enumerate(batch_files):
                    status.markdown(f"⏳ Procesando **{f.name}**…")
                    img_bgr = cv2.cvtColor(
                        np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
                    )
                    results = analyze(img_bgr, min_elongation=min_elong)

                    for res in results:
                        row = {
                            "Archivo": f.name,
                            "Pie":     res["label"],
                            "Huesos":  res["bones_total"],
                        }
                        for k in ANGLE_INFO:
                            if k in res["measurements"]:
                                m    = res["measurements"][k]
                                info = ANGLE_INFO[k]
                                row[f"{k} (°)"]      = m["angle"]
                                row[f"{k} Clasif."]  = get_severity(m["angle"], info)[0]
                                row[f"{k} Conf.(%)"] = int(m["confidence"] * 100)
                            else:
                                row[f"{k} (°)"]      = "—"
                                row[f"{k} Clasif."]  = "—"
                                row[f"{k} Conf.(%)"] = "—"
                        all_rows.append(row)

                        # Guardar imagen anotada
                        safe = res["label"].replace(" ", "_").replace("/","")
                        jpg  = cv2.imencode(
                            ".jpg", res["img_result"],
                            [cv2.IMWRITE_JPEG_QUALITY, 92]
                        )[1].tobytes()
                        zip_images.append(
                            (f"{f.name.rsplit('.',1)[0]}_{safe}.jpg", jpg)
                        )

                    progress.progress(
                        (idx + 1) / len(batch_files),
                        text=f"{idx+1}/{len(batch_files)} procesadas"
                    )

                status.success(
                    f"✅ Completado — {len(batch_files)} imágenes · "
                    f"{len(all_rows)} pies analizados"
                )
                progress.empty()

                df = pd.DataFrame(all_rows)
                st.subheader("📊 Resumen del lote")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # ZIP
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("resumen_hallux_valgus.csv",
                                df.to_csv(index=False).encode("utf-8"))
                    for fname, data in zip_images:
                        zf.writestr(fname, data)
                zip_buf.seek(0)

                st.download_button(
                    "⬇️ Descargar ZIP (imágenes anotadas + CSV)",
                    data=zip_buf.getvalue(),
                    file_name=f"hallux_valgus_{datetime.now():%Y%m%d_%H%M}.zip",
                    mime="application/zip",
                )

                st.markdown("""
                <div class="warn-box">
                  ⚠️ Revisar cada imagen anotada antes de incluir los valores en el estudio.
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
