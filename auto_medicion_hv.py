#!/usr/bin/env python3
"""
Medición Automática de Ángulos — Hallux Valgus
Versión con interfaz web Streamlit

Uso:
    streamlit run auto_medicion_hv.py

Requiere:
    pip install streamlit opencv-python-headless numpy Pillow scikit-learn pandas
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
        "desc":     "Eje 1er metatarsiano / Eje falange proximal del 1er dedo",
        "normal":   15,
        "mild":     20,
        "moderate": 40,
        "color_bgr":(100, 120, 248),
        "color_hex":"#f87171",
    },
    "AIM12": {
        "name":     "Ángulo Intermetatarsiano 1-2",
        "desc":     "Eje 1er metatarsiano / Eje 2do metatarsiano",
        "normal":   9,
        "mild":     11,
        "moderate": 16,
        "color_bgr":(80, 211, 130),
        "color_hex":"#34d399",
    },
    "AIM25": {
        "name":     "Ángulo Intermetatarsiano 2-5",
        "desc":     "Eje 2do metatarsiano / Eje 5to metatarsiano",
        "normal":   20,
        "mild":     25,
        "moderate": 35,
        "color_bgr":(251, 150, 80),
        "color_hex":"#fb923c",
    },
}

def get_severity(angle, info):
    if info.get("normal") is None:
        return "N/A", "#8b949e"
    if angle <= info["normal"]:   return "Normal",   "#56d364"
    if angle <= info["mild"]:     return "Leve",     "#f0a030"
    if angle <= info["moderate"]: return "Moderado", "#f57f5b"
    return "Severo", "#f87171"


# ══════════════════════════════════════════════════════════════
# PIPELINE DE VISIÓN ARTIFICIAL
# ══════════════════════════════════════════════════════════════

def preprocess(gray: np.ndarray) -> np.ndarray:
    """Normaliza y mejora el contraste de la radiografía."""
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised


def segment(img_proc: np.ndarray) -> np.ndarray:
    """
    Prueba múltiples estrategias de umbralización y elige la que
    produce más estructuras elongadas (huesos).
    """
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    candidates = []
    for inv in [False, True]:
        src = 255 - img_proc if inv else img_proc
        _, otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adapt = cv2.adaptiveThreshold(src, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
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


def pca_axis(points_xy: np.ndarray):
    """PCA sobre un conjunto de puntos (x,y). Devuelve (centro, eje_principal, elongación)."""
    mean = points_xy.mean(axis=0)
    c = points_xy - mean
    cov = (c.T @ c) / len(points_xy)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    elongation = np.sqrt(vals[0] / (vals[1] + 1e-9))
    return mean, vecs[:, 0], elongation, vals


def extract_bones(mask: np.ndarray, min_area_frac=0.0005, min_elong=2.0):
    """
    Extrae componentes óseos: usa PCA para calcular el eje principal
    y filtra por elongación.
    """
    h, w = mask.shape
    min_area = int(w * h * min_area_frac)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    bones = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) / (min(bw, bh) + 1) < 1.5:
            continue

        # Coordenadas de los píxeles del componente
        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        if len(pts) < 20:
            continue

        center, axis, elong, vals = pca_axis(pts)
        if elong < min_elong:
            continue

        proj = (pts - center) @ axis
        length = proj.max() - proj.min()

        bones.append({
            "center":     center,
            "axis":       axis,
            "elongation": elong,
            "length":     float(length),
            "area":       int(area),
            "bbox":       (stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP],
                           bw, bh),
        })

    return bones


def classify_bones(bones: list):
    """
    Clasifica metatarsianos vs falanges por longitud relativa,
    y los ordena de medial (1ro) a lateral (5to) por posición X.
    """
    if len(bones) < 2:
        return [], []

    lengths = np.array([b["length"] for b in bones])
    thresh = np.percentile(lengths, 35)   # los más largos = metatarsianos

    meta = sorted([b for b in bones if b["length"] >= thresh],
                  key=lambda b: b["center"][0])
    phal = sorted([b for b in bones if b["length"] <  thresh],
                  key=lambda b: b["center"][0])

    for i, b in enumerate(meta): b["label"] = f"MT{i+1}"
    for i, b in enumerate(phal): b["label"] = f"PF{i+1}"

    return meta, phal


def angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    """Ángulo agudo (0–90°) entre dos vectores dirección."""
    d1 = d1 / (np.linalg.norm(d1) + 1e-9)
    d2 = d2 / (np.linalg.norm(d2) + 1e-9)
    cos_a = np.clip(abs(np.dot(d1, d2)), 0.0, 1.0)
    a = np.degrees(np.arccos(cos_a))
    return round(min(a, 180.0 - a), 1)


def confidence_score(b1, b2) -> float:
    """Heurística de confianza basada en elongación y longitud de los huesos."""
    e = (b1["elongation"] + b2["elongation"]) / 2
    return round(min(1.0, e / 8.0), 2)


def measure_angles(meta: list, phal: list) -> dict:
    """
    Calcula AHV, AIM 1-2 y AIM 2-5 cuando hay suficientes huesos.
    Devuelve dict con angle, bone1, bone2, confidence.
    """
    results = {}

    if len(meta) >= 1 and len(phal) >= 1:
        a = angle_between(meta[0]["axis"], phal[0]["axis"])
        results["AHV"] = {"angle": a, "bone1": meta[0], "bone2": phal[0],
                          "confidence": confidence_score(meta[0], phal[0])}

    if len(meta) >= 2:
        a = angle_between(meta[0]["axis"], meta[1]["axis"])
        results["AIM12"] = {"angle": a, "bone1": meta[0], "bone2": meta[1],
                            "confidence": confidence_score(meta[0], meta[1])}

    if len(meta) >= 5:
        a = angle_between(meta[1]["axis"], meta[4]["axis"])
        results["AIM25"] = {"angle": a, "bone1": meta[1], "bone2": meta[4],
                            "confidence": confidence_score(meta[1], meta[4])}

    return results


# ══════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════

def draw_overlay(img_bgr: np.ndarray, meta: list, phal: list,
                 measurements: dict) -> np.ndarray:
    """Dibuja ejes óseos y etiquetas de ángulo sobre la imagen."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    def draw_axis(bone, color, thickness=1):
        c  = bone["center"].astype(int)
        hl = int(bone["length"] / 2)
        p1 = tuple((c - bone["axis"] * hl).astype(int))
        p2 = tuple((c + bone["axis"] * hl).astype(int))
        cv2.line(out, p1, p2, color, thickness)
        cv2.circle(out, tuple(c), 4, color, -1)
        cv2.putText(out, bone.get("label", ""), (c[0]+5, c[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Todos los huesos en gris
    for b in meta: draw_axis(b, (160, 160, 160), 1)
    for b in phal: draw_axis(b, (110, 110, 110), 1)

    # Mediciones resaltadas en color
    y_off = 18
    for key, m in measurements.items():
        info = ANGLE_INFO[key]
        col  = info["color_bgr"]
        sev, _ = get_severity(m["angle"], info)
        conf_pct = int(m["confidence"] * 100)

        draw_axis(m["bone1"], col, 2)
        draw_axis(m["bone2"], col, 2)

        label = f"{key}: {m['angle']}  {sev}  (conf.{conf_pct}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (7, y_off-th-3), (11+tw, y_off+4), (12, 12, 18), -1)
        cv2.rectangle(out, (7, y_off-th-3), (11+tw, y_off+4), col, 1)
        cv2.putText(out, label, (9, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
        y_off += 28

    # Pie de imagen
    note = "RESULTADO AUTOMATICO  -  REQUIERE VERIFICACION CLINICA"
    (nw, nh), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(out, (7, h-25), (11+nw, h-4), (12, 12, 18), -1)
    cv2.putText(out, note, (9, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 200), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ══════════════════════════════════════════════════════════════

def analyze(img_bgr: np.ndarray, min_elongation: float = 2.0) -> dict:
    """
    Pipeline completo.
    Devuelve: {measurements, meta, phal, img_result, bones_total}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    proc = preprocess(gray)
    mask = segment(proc)
    bones = extract_bones(mask, min_elong=min_elongation)
    meta, phal = classify_bones(bones)
    meas = measure_angles(meta, phal)
    img_result = draw_overlay(img_bgr, meta, phal, meas)
    return {
        "measurements": meas,
        "meta":         meta,
        "phal":         phal,
        "img_result":   img_result,
        "bones_total":  len(bones),
    }


# ══════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Auto-Medición Hallux Valgus",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS personalizado ──────────────────────────────────────
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] { background: #0d1117; }
      [data-testid="stSidebar"]          { background: #161b22; }
      h1,h2,h3  { color: #58a6ff; }
      .metric-card {
          background: #161b22;
          border: 1px solid #30363d;
          border-radius: 10px;
          padding: 18px 12px;
          text-align: center;
      }
      .m-val  { font-size:2.2rem; font-weight:700; color:#fff; }
      .m-name { font-size:0.73rem; color:#8b949e; margin-bottom:6px; }
      .m-sev  { font-size:0.85rem; font-weight:600; margin-top:4px; }
      .m-conf { font-size:0.68rem; color:#4b5563; margin-top:4px; }
      .m-norm { font-size:0.68rem; color:#4b5563; }
      .warn-box {
          background:#1c1a10; border:1px solid #6e5203;
          border-radius:8px; padding:12px 16px;
          color:#d4aa40; font-size:0.82rem; margin-top:14px;
      }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros")
        st.markdown("---")
        min_elong = st.slider(
            "Elongación mínima de hueso",
            min_value=1.5, max_value=5.0, value=2.0, step=0.1,
            help="Aumenta si detecta ruido; reduce si no encuentra huesos."
        )
        st.markdown("---")
        st.markdown("### 📋 Rangos normales")
        for k, v in ANGLE_INFO.items():
            st.markdown(
                f"**{k}** — Normal ≤{v['normal']}° · Leve ≤{v['mild']}° · Mod. ≤{v['moderate']}°"
            )
        st.markdown("---")
        st.markdown("""
        ### ℹ️ Acerca de esta herramienta
        Usa **visión artificial (OpenCV + PCA)** para detectar
        automáticamente los ejes de los huesos del pie y calcular
        los ángulos relevantes para el estudio de Hallux Valgus.

        **Limitaciones:**
        - Optimizado para radiografías **AP dorsoplantares** estándar
        - La confianza baja con imágenes oblicuas o baja calidad
        - Siempre verificar visualmente el resultado
        """)

    # ── Header ────────────────────────────────────────────────
    st.markdown("# 🦴 Medición Automática — Hallux Valgus")
    st.markdown("Detección automática de ejes óseos y cálculo de ángulos mediante visión artificial")
    st.markdown("---")

    # ── Tabs: una imagen / múltiples imágenes ─────────────────
    tab1, tab2 = st.tabs(["📷 Análisis individual", "📁 Análisis por lote (múltiples imágenes)"])

    # ═══════════════════════════════════════════════
    # TAB 1 — IMAGEN INDIVIDUAL
    # ═══════════════════════════════════════════════
    with tab1:
        uploaded = st.file_uploader(
            "Sube una radiografía del pie (vista dorsoplantar AP)",
            type=["jpg","jpeg","png","tiff","tif","bmp"],
            key="single",
        )

        if not uploaded:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;
                        border-radius:12px;padding:50px;text-align:center;
                        color:#4b5563;margin-top:20px">
              <p style="font-size:3rem">📷</p>
              <p style="font-size:1.05rem;color:#8b949e">
                Sube una radiografía para comenzar el análisis automático
              </p>
              <p style="font-size:0.78rem">JPG · PNG · TIFF · BMP</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            img_pil  = Image.open(uploaded).convert("RGB")
            img_np   = np.array(img_pil)
            img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Analizando radiografía…"):
                result = analyze(img_bgr, min_elongation=min_elong)

            meas       = result["measurements"]
            img_result = result["img_result"]
            n_bones    = result["bones_total"]

            # Columnas: original | anotada
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Radiografía original")
                st.image(img_pil, use_container_width=True)
                st.caption(f"{img_pil.width}×{img_pil.height} px — "
                           f"{uploaded.name}")
            with col2:
                st.subheader("Análisis automático")
                st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
                st.caption(f"Estructuras elongadas detectadas: {n_bones} "
                           f"(metatarsianos: {len(result['meta'])} · "
                           f"falanges: {len(result['phal'])})")

            # ── Resultados ──────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Ángulos medidos")

            if not meas:
                st.warning(
                    "⚠️ No se pudieron calcular ángulos. "
                    "Verifica que la imagen sea una radiografía AP del antepié "
                    "y ajusta el parámetro de elongación en el panel lateral."
                )
            else:
                cols = st.columns(len(meas))
                for i, (key, m) in enumerate(meas.items()):
                    info = ANGLE_INFO[key]
                    sev, sev_col = get_severity(m["angle"], info)
                    conf_pct = int(m["confidence"] * 100)
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card"
                             style="border-color:{info['color_hex']}33">
                          <div class="m-name">{info['name']}</div>
                          <div class="m-val" style="color:{sev_col}">
                            {m['angle']}°
                          </div>
                          <div class="m-sev" style="color:{sev_col}">{sev}</div>
                          <div class="m-conf">Confianza: {conf_pct}%</div>
                          <div class="m-norm">Normal ≤{info['normal']}°</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Tabla resumen
                st.markdown("#### Detalle")
                rows = []
                for key, m in meas.items():
                    info = ANGLE_INFO[key]
                    sev, _ = get_severity(m["angle"], info)
                    rows.append({
                        "Ángulo":        key,
                        "Nombre":        info["name"],
                        "Valor (°)":     m["angle"],
                        "Clasificación": sev,
                        "Confianza (%)": int(m["confidence"]*100),
                        "Normal ≤":      f"{info['normal']}°",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

            # ── Descargas ───────────────────────────────────────
            st.markdown("---")
            dc1, dc2 = st.columns(2)

            with dc1:
                img_bytes = cv2.imencode(
                    ".jpg", img_result,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )[1].tobytes()
                st.download_button(
                    "⬇️ Descargar imagen anotada",
                    data=img_bytes,
                    file_name=uploaded.name.rsplit(".",1)[0] + "_medido.jpg",
                    mime="image/jpeg",
                )

            with dc2:
                report = {
                    "archivo":    uploaded.name,
                    "fecha":      datetime.now().isoformat(),
                    "dimensiones":f"{img_pil.width}x{img_pil.height}",
                    "huesos_detectados": n_bones,
                    "mediciones": {
                        k: {
                            "nombre":       ANGLE_INFO[k]["name"],
                            "angulo_grados": m["angle"],
                            "severidad":    get_severity(m["angle"],
                                                         ANGLE_INFO[k])[0],
                            "confianza_pct":int(m["confidence"]*100),
                            "rango_normal": f"≤{ANGLE_INFO[k]['normal']}°",
                        }
                        for k, m in meas.items()
                    },
                }
                st.download_button(
                    "⬇️ Descargar reporte JSON",
                    data=json.dumps(report, ensure_ascii=False,
                                    indent=2).encode("utf-8"),
                    file_name=uploaded.name.rsplit(".",1)[0] + "_reporte.json",
                    mime="application/json",
                )

            st.markdown("""
            <div class="warn-box">
              ⚠️ <strong>Uso clínico e investigación:</strong>
              Los valores son estimaciones automáticas basadas en visión artificial.
              La precisión depende de la calidad y posicionamiento de la radiografía.
              <strong>Verificar y corregir manualmente</strong> antes de registrar en el estudio.
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # TAB 2 — LOTE DE IMÁGENES
    # ═══════════════════════════════════════════════
    with tab2:
        st.markdown(
            "Sube varias radiografías a la vez. "
            "El sistema las analizará todas y generará un resumen descargable."
        )

        batch_files = st.file_uploader(
            "Selecciona múltiples radiografías",
            type=["jpg","jpeg","png","tiff","tif","bmp"],
            accept_multiple_files=True,
            key="batch",
        )

        if not batch_files:
            st.info("No hay imágenes cargadas. Selecciona una o más radiografías.")
        else:
            if st.button(f"🚀 Analizar {len(batch_files)} imagen(es)", type="primary"):
                batch_results = []
                progress = st.progress(0, text="Procesando…")
                status_area = st.empty()

                for idx, f in enumerate(batch_files):
                    status_area.markdown(f"⏳ Procesando **{f.name}**…")
                    img_pil  = Image.open(f).convert("RGB")
                    img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    res      = analyze(img_bgr, min_elongation=min_elong)
                    meas     = res["measurements"]

                    row = {"Archivo": f.name, "Huesos detectados": res["bones_total"]}
                    for k in ANGLE_INFO:
                        if k in meas:
                            info = ANGLE_INFO[k]
                            row[f"{k} (°)"]       = meas[k]["angle"]
                            row[f"{k} Clasif."]   = get_severity(meas[k]["angle"], info)[0]
                            row[f"{k} Conf.(%)"]  = int(meas[k]["confidence"]*100)
                        else:
                            row[f"{k} (°)"]      = "—"
                            row[f"{k} Clasif."]  = "—"
                            row[f"{k} Conf.(%)"] = "—"

                    batch_results.append({
                        "row":        row,
                        "img_result": res["img_result"],
                        "filename":   f.name,
                    })
                    progress.progress((idx+1)/len(batch_files),
                                      text=f"{idx+1}/{len(batch_files)} procesadas")

                status_area.success(
                    f"✅ Análisis completado — {len(batch_results)} imágenes")
                progress.empty()

                # Tabla resumen
                st.subheader("📊 Resumen del lote")
                df = pd.DataFrame([r["row"] for r in batch_results])
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Generar ZIP con todas las imágenes anotadas + CSV
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # CSV
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    zf.writestr("resumen_hallux_valgus.csv", csv_bytes)

                    # Imágenes anotadas
                    for r in batch_results:
                        jpg = cv2.imencode(
                            ".jpg", r["img_result"],
                            [cv2.IMWRITE_JPEG_QUALITY, 92]
                        )[1].tobytes()
                        stem = r["filename"].rsplit(".", 1)[0]
                        zf.writestr(f"{stem}_medido.jpg", jpg)

                zip_buf.seek(0)
                st.download_button(
                    "⬇️ Descargar ZIP (imágenes anotadas + CSV)",
                    data=zip_buf.getvalue(),
                    file_name=f"hallux_valgus_{datetime.now():%Y%m%d_%H%M}.zip",
                    mime="application/zip",
                )

                st.markdown("""
                <div class="warn-box">
                  ⚠️ <strong>Verificación obligatoria:</strong>
                  Revisa cada imagen anotada antes de incluir los valores en el estudio.
                  Las medidas automáticas son un punto de partida, no un resultado definitivo.
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
