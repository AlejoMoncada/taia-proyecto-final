"""Build the defense deck as HTML and print it to PDF with a headless Chromium browser.

Usage:
    uv run python scripts/build_slides.py            # writes docs/slides/diapositivas.html and the PDF
    uv run python scripts/build_slides.py --html-only

Inputs: real PTB-XL traces in assets/video/traces.json (normalized samples, no raw data)
and the checked-in evidence in docs/evidence/. Visual identity: Universidad de La Sabana.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
TRACES = ROOT / "assets" / "video" / "traces.json"
HTML = ROOT / "docs" / "slides" / "diapositivas.html"
PDF = ROOT / "output" / "pdf" / "diapositivas.pdf"
FS = 100  # Hz
BROWSERS = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
]


# ---------------------------------------------------------------- datos

def load_results() -> dict[str, dict[str, str]]:
    with (EVIDENCE / "table_test.csv").open(encoding="utf-8", newline="") as stream:
        return {row["Modelo"]: row for row in csv.DictReader(stream)}


def mean(cell: str) -> float:
    return float(cell.split("±")[0].strip())


def es(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


# ---------------------------------------------------------------- trazados

def trace_svg(samples: list[float], seconds: float, width: int, height: int,
              label: str | None = None, marks: dict[str, int] | None = None, stroke: float = 1.6) -> str:
    """Real ECG trace on millimetre paper: 0,2 s per major square, like an electrocardiograph."""
    n = min(len(samples), int(seconds * FS))
    data = samples[:n]
    px_s = width / seconds
    major = px_s * 0.2
    lo, hi = min(data), max(data)
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad

    def y(v: float) -> float:
        return height - (v - lo) / (hi - lo) * height

    points = " ".join(f"{i * width / (n - 1):.1f},{y(v):.1f}" for i, v in enumerate(data))
    grid = []
    minor = major / 5
    k = 0
    x = 0.0
    while x <= width + 0.1:
        grid.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" class="{"gM" if k % 5 == 0 else "gm"}"/>')
        x += minor
        k += 1
    k = 0
    yy = float(height)
    while yy >= -0.1:
        grid.append(f'<line x1="0" y1="{yy:.1f}" x2="{width}" y2="{yy:.1f}" class="{"gM" if k % 5 == 0 else "gm"}"/>')
        yy -= minor
        k += 1
    extra = ""
    if marks:
        for name, idx in marks.items():
            mx = idx * width / (n - 1)
            extra += (f'<line x1="{mx:.1f}" y1="14" x2="{mx:.1f}" y2="{height}" class="mark"/>'
                      f'<text x="{mx:.1f}" y="11" class="markt">{escape(name)}</text>')
    tag = (f'<rect x="0" y="0" width="{12 + 9 * len(label)}" height="22" class="leadbg"/>'
           f'<text x="5" y="16" class="lead">{escape(label)}</text>') if label else ""
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="Trazado de ECG {escape(label or "")}">{"".join(grid)}{extra}'
            f'<polyline points="{points}" class="ecg" style="stroke-width:{stroke}"/>{tag}</svg>')


# ---------------------------------------------------------------- gráficas propias

def f1_bars(results: dict[str, dict[str, str]]) -> str:
    rows = [("Random Forest", "random_forest", False), ("XGBoost", "xgboost", False),
            ("XGBoost ajustado", "xgboost_tuned", False), ("CNN 1D", "cnn1d", True),
            ("CNN + LSTM", "cnn_lstm", True)]
    out = []
    for name, key, net in rows:
        f1 = mean(results[key]["F1 macro"])
        auc = mean(results[key]["ROC-AUC macro"])
        sd = results[key]["F1 macro"].split("±")[1].strip().replace(".", ",")
        width = (f1 - 0.5) / 0.3 * 100
        out.append(f'<div class="bar-row{" net" if net else ""}"><span class="bar-name">{name}</span>'
                   f'<span class="bar-track"><span class="bar" style="width:{width:.1f}%"></span></span>'
                   f'<span class="bar-val">{es(f1)}<small> ± {sd}</small></span>'
                   f'<span class="bar-auc">{es(auc)}</span></div>')
    return ('<div class="bars"><div class="bar-row head"><span></span><span class="axis">'
            '<span>0,50</span><span>0,60</span><span>0,70</span><span>0,80</span></span>'
            '<span>F1 macro</span><span>ROC-AUC</span></div>' + "".join(out) + "</div>")


def dumbbell(results: dict[str, dict[str, str]]) -> str:
    classes = [("NORM", "Normal"), ("MI", "Infarto"), ("STTC", "Cambios ST/T"),
               ("CD", "Conducción"), ("HYP", "Hipertrofia")]
    w, h, left, right = 760, 330, 170, 40
    lo, hi = 0.55, 0.90

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (w - left - right)

    parts = []
    for t in (0.6, 0.7, 0.8, 0.9):
        parts.append(f'<line x1="{x(t):.0f}" y1="10" x2="{x(t):.0f}" y2="{h - 30}" class="axisl"/>'
                     f'<text x="{x(t):.0f}" y="{h - 8}" class="axist">{es(t, 1)}</text>')
    for i, (code, name) in enumerate(classes):
        yy = 34 + i * 58
        a = mean(results["xgboost"][f"F1 {code}"])
        b = mean(results["cnn_lstm"][f"F1 {code}"])
        flat = abs(b - a) < 0.01
        parts.append(f'<text x="0" y="{yy + 6}" class="cls{" flat" if flat else ""}">{name}</text>'
                     f'<line x1="{x(a):.1f}" y1="{yy}" x2="{x(b):.1f}" y2="{yy}" class="link"/>'
                     f'<circle cx="{x(a):.1f}" cy="{yy}" r="9" class="dot-base"/>'
                     f'<circle cx="{x(b):.1f}" cy="{yy}" r="9" class="dot-net"/>'
                     f'<text x="{x(max(a, b)) + 16:.1f}" y="{yy + 6}" class="delta{" flat" if flat else ""}">'
                     f'{"sin cambio" if flat else "+" + es(b - a)}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="dumbbell">{"".join(parts)}</svg>'


def cost_bars(results: dict[str, dict[str, str]]) -> str:
    rows = [("XGBoost", "xgboost", "CPU"), ("CNN 1D", "cnn1d", "GPU"), ("CNN + LSTM", "cnn_lstm", "GPU")]
    top = max(mean(results[k]["Entrenamiento (s)"]) for _, k, _ in rows)
    out = []
    for name, key, hw in rows:
        train = mean(results[key]["Entrenamiento (s)"])
        search = results[key]["Busqueda (s)"]
        search = "sin búsqueda" if search.strip() == "-" else f"búsqueda {thousands(float(search))} s"
        infer = es(mean(results[key]["Inferencia (ms/ECG)"]))
        out.append(f'<div class="cost-row{" net" if hw == "GPU" else ""}"><span class="bar-name">{name}'
                   f'<small>{hw}</small></span><span class="bar-track"><span class="bar" '
                   f'style="width:{train / top * 100:.1f}%"></span></span><span class="bar-val">{train:.0f} s</span>'
                   f'<span class="cost-note">{search} · {infer} ms por ECG</span></div>')
    return '<div class="bars cost">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- diapositivas

def slide(body: str, cls: str = "", n: int = 0, total: int = 0) -> str:
    foot = ('<footer><img src="../../assets/brand/logo_unisabana.png" alt="Universidad de La Sabana">'
            f'<span>TAIA · Grupo 9 · ECG multietiqueta con PTB-XL</span><span class="num">{n} / {total}</span></footer>')
    return f'<section class="slide {cls}">{body}{foot}</section>'


def build_html() -> str:
    t = json.loads(TRACES.read_text(encoding="utf-8"))
    r = load_results()
    sup, leads, beat = t["superclasses"], t["leads"], t["beat"]
    f1_xgb, f1_lstm = mean(r["xgboost"]["F1 macro"]), mean(r["cnn_lstm"]["F1 macro"])
    ece = {k: es(mean(r[k]["ECE"])) for k in ("xgboost", "cnn1d", "cnn_lstm")}
    ratio = mean(r["cnn_lstm"]["Entrenamiento (s)"]) / mean(r["xgboost"]["Entrenamiento (s)"])

    names = {"NORM": "Normal", "MI": "Infarto de miocardio", "STTC": "Cambios del ST/T",
             "CD": "Trastorno de conducción", "HYP": "Hipertrofia"}
    five = "".join(
        f'<div class="dx"><div class="dx-label"><b>{c}</b><span>{names[c]}</span></div>'
        f'{trace_svg(sup[c]["samples"], 4, 760, 78, stroke=1.5)}<span class="dx-id">ECG {sup[c]["ecg_id"]}</span></div>'
        for c in names)
    order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    twelve = "".join(f'<div class="lead-cell">{trace_svg(leads[k], 5, 560, 70, label=k, stroke=1.3)}</div>'
                     for k in order)
    marks = {"P": beat["marks"]["p"], "QRS": beat["marks"]["r"], "T": beat["marks"]["t"]}

    slides = [
        ("cover", f'''
<img class="cover-logo" src="../../assets/brand/logo_unisabana.png" alt="Universidad de La Sabana">
<h1>¿Una red que lee la señal cruda supera a 210 características hechas a mano?</h1>
<p class="lede">Clasificación multietiqueta de ECG con PTB-XL</p>
<div class="cover-trace">{trace_svg(sup["NORM"]["samples"], 10, 1280, 170, stroke=2)}</div>
<p class="meta">Grupo 9 · TAIA · Maestría en Inteligencia Artificial · Universidad de La Sabana · Octubre de 2026</p>
<p class="caption cover-cap">ECG real de PTB-XL: registro {sup["NORM"]["ecg_id"]}, derivación II, 10 segundos. Uso académico, sin aplicación clínica.</p>'''),
        ("", f'''
<h2>Un ECG son 12 vistas del corazón, registradas durante 10 segundos</h2>
<div class="twelve">{twelve}</div>
<p class="caption">Registro {t["reference_ecg_id"]} de PTB-XL, diagnóstico normal. Se muestran 5 de los 10 s. A 100 Hz son 1.000 muestras por derivación: 12 × 1.000 números es todo lo que recibe la red.</p>'''),
        ("", f'''
<h2>Cinco diagnósticos que pueden aparecer en el mismo paciente</h2>
<div class="five">{five}</div>
<div class="facts"><p><b>21.388</b> ECG con diagnóstico de cardiólogo</p><p><b>24,1 %</b> tiene más de una etiqueta</p><p>Por eso es <b>multietiqueta</b>: cinco salidas independientes, no una sola clase</p></div>'''),
        ("", '''
<h2>El conjunto de prueba no tomó ninguna decisión</h2>
<div class="split">
  <div class="part train" style="flex:17084"><b>Entrenamiento</b><span>17.084 ECG · folds 1 a 8</span><p>Ajusta los pesos, la imputación y el escalado</p></div>
  <div class="part val" style="flex:3400"><b>Validación</b><span>2.146 ECG · fold 9</span><p>Elige hiperparámetros, mejor época y umbral por clase</p></div>
  <div class="part test" style="flex:3400"><b>Prueba</b><span>2.158 ECG · fold 10</span><p>Solo se mira al final, una vez</p></div>
</div>
<ul class="rules">
  <li>Ningún paciente aparece en dos particiones: usamos <code>strat_fold</code> de PTB-XL.</li>
  <li>Semillas fijas: 123 para la búsqueda; 42, 43 y 44 para los modelos finales.</li>
  <li>Mismo presupuesto por familia: 8 configuraciones al azar, elegidas por ROC-AUC macro en validación.</li>
</ul>'''),
        ("", f'''
<h2>Dos maneras de leer el mismo latido</h2>
<div class="beat">{trace_svg(beat["samples"], beat["n_samples"] / FS, 560, 280, marks=marks, stroke=2.6)}</div>
<div class="two">
  <div><h3>Línea base: lo que medimos nosotros</h3><p><b>210 características</b> por registro: amplitudes, espectro, ritmo RR, edad y sexo.</p><p>Random Forest y XGBoost.</p></div>
  <div><h3>Redes: lo que aprende el modelo</h3><p>La <b>señal completa</b>, sin reglas previas.</p><p>CNN 1D residual (363.781 parámetros) y CNN + LSTM bidireccional (464.901).</p></div>
</div>
<p class="caption">Latido real del registro {beat["ecg_id"]}: onda P, complejo QRS y onda T.</p>'''),
        ("", f'''
<h2>Leer la señal cruda sube el F1 macro de {es(f1_xgb, 2)} a {es(f1_lstm, 2)}</h2>
{f1_bars(r)}
<p class="caption">Prueba: 2.158 ECG. Media ± desviación entre tres semillas. Esa desviación mide la variación del entrenamiento, no es un intervalo de confianza. CNN 1D y CNN + LSTM quedan dentro de esa variación: no afirmamos que una supere a la otra.</p>'''),
        ("", f'''
<h2>La ganancia está en infarto y conducción; la hipertrofia no mejora</h2>
<div class="dumb-wrap">{dumbbell(r)}
<div class="legend"><p><i class="dot-base"></i>XGBoost</p><p><i class="dot-net"></i>CNN + LSTM</p><p class="note">F1 por clase en prueba, media de tres semillas.</p><p class="note">HYP es la clase menos frecuente y se queda cerca de 0,61 con todos los modelos. No probamos si el límite es el desbalance, el ruido de etiqueta o la señal.</p></div></div>'''),
        ("", '''
<h2>Cada red se queda con su mejor época de validación</h2>
<div class="figure"><img src="../evidence/curves.png" alt="Curvas de pérdida y AUC en entrenamiento y validación"></div>
<p class="caption side">Máximo 60 épocas o 5 minutos por corrida, paciencia de 8 épocas y se restauran los pesos de la mejor época según AUC en validación. Dos configuraciones de CNN 1D corrieron lento y se cortaron en la época 4; lo reportamos como limitación.</p>'''),
        ("", f'''
<h2>Las redes calibran mejor, pero no las recalibramos</h2>
<div class="figure wide"><img src="../evidence/reliability_test.png" alt="Diagramas de confiabilidad en prueba"></div>
<div class="facts three"><p>ECE <b>{ece["xgboost"]}</b> XGBoost</p><p>ECE <b>{ece["cnn1d"]}</b> CNN 1D</p><p>ECE <b>{ece["cnn_lstm"]}</b> CNN + LSTM</p></div>
<p class="caption">Diagramas de la semilla 42. No aplicamos escalado de temperatura; si se hace, debe ajustarse solo en validación. Calibrar bien aquí no garantiza probabilidades confiables en otro hospital.</p>'''),
        ("", f'''
<h2>La mejora cuesta hasta {ratio:.0f} veces más tiempo de entrenamiento</h2>
{cost_bars(r)}
<p class="caption">Entrenamiento final medio por semilla. XGBoost corre en CPU (Ryzen 5 3600) y las redes en GPU (GTX 1660 SUPER), así que igualamos el número de configuraciones, no el cómputo total. Lo reportamos tal cual.</p>'''),
        ("close", '''
<h2>Lo que podemos afirmar, y lo que no</h2>
<div class="two verdict">
  <div><h3>Afirmamos</h3><ul><li>En esta partición, las redes superan a la línea base en F1, AUC y calibración.</li><li>La mejora se concentra en infarto y trastornos de conducción.</li><li>Ninguna decisión se tomó mirando el conjunto de prueba.</li></ul></div>
  <div><h3>No afirmamos</h3><ul><li>Que la CNN + LSTM sea mejor que la CNN 1D.</li><li>Que el modelo sirva en otro hospital o con pacientes reales.</li><li>Que funcione igual por sexo, edad o equipo: no lo medimos.</li></ul></div>
</div>
<p class="next"><b>Siguiente paso:</b> bootstrap pareado por paciente, escalado de temperatura ajustado en validación y análisis por subgrupos.</p>
<p class="caption">Wagner et al. (2020), Scientific Data, doi:10.1038/s41597-020-0495-6 · Strodthoff et al. (2021), IEEE JBHI, doi:10.1109/JBHI.2020.3022989 · Código: github.com/AlejoMoncada/taia-proyecto-final</p>'''),
    ]
    total = len(slides)
    body = "".join(slide(b, c, i + 1, total) for i, (c, b) in enumerate(slides))
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>ECG multietiqueta con PTB-XL · TAIA Grupo 9</title>
<style>{CSS}</style></head><body>{body}</body></html>
"""


CSS = """
@font-face{font-family:"Cabinet Grotesk";src:url("../../assets/fonts/CabinetGrotesk-Regular.ttf");font-weight:400}
@font-face{font-family:"Cabinet Grotesk";src:url("../../assets/fonts/CabinetGrotesk-Bold.ttf");font-weight:700}
@font-face{font-family:"Cabinet Grotesk";src:url("../../assets/fonts/CabinetGrotesk-Extrabold.ttf");font-weight:800}
:root{--azul:#001c64;--contraste:#193f9e;--claro:#9dc4ff;--cielo:#e0edff;--amarillo:#fffae0;
--tinta:#282828;--gris:#6d6d6d;--grisclaro:#f4f4f4;--medio:#aaaaaa}
@page{size:1280px 720px;margin:0}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#fff}
body{font-family:"Cabinet Grotesk",Arial,sans-serif;color:var(--tinta);font-variant-numeric:tabular-nums;
-webkit-print-color-adjust:exact;print-color-adjust:exact}
::selection{background:var(--cielo);color:var(--azul)}
.slide{position:relative;width:1280px;height:720px;overflow:hidden;padding:52px 72px 0;page-break-after:always;break-after:page}
.slide::before{content:"";position:absolute;inset:0 0 auto 0;height:8px;background:var(--azul)}
h1{font-weight:800;font-size:54px;line-height:1.06;letter-spacing:-.025em;color:var(--azul);max-width:900px;text-wrap:balance}
h2{font-weight:800;font-size:38px;line-height:1.1;letter-spacing:-.02em;color:var(--azul);max-width:1080px;text-wrap:balance;margin-bottom:26px}
h3{font-weight:700;font-size:22px;color:var(--contraste);margin-bottom:10px}
p,li{font-size:21px;line-height:1.4}
b{font-weight:700;color:var(--azul)}
code{font-family:inherit;font-weight:700;color:var(--contraste)}
.caption{font-size:16px;line-height:1.45;color:var(--gris);max-width:1080px;margin-top:16px}
footer{position:absolute;left:72px;right:72px;bottom:20px;display:flex;align-items:center;gap:18px;
border-top:1px solid var(--cielo);padding-top:10px;font-size:13px;color:var(--contraste)}
footer img{height:26px}
footer .num{margin-left:auto;font-weight:700}
svg .gm{stroke:#193f9e;stroke-opacity:.07;stroke-width:.6}
svg .gM{stroke:#193f9e;stroke-opacity:.17;stroke-width:.8}
svg .ecg{fill:none;stroke:var(--azul);stroke-linejoin:round;stroke-linecap:round}
svg .leadbg{fill:#fff;fill-opacity:.9}
svg .lead{font:700 14px "Cabinet Grotesk",Arial;fill:var(--contraste)}
svg .mark{stroke:var(--contraste);stroke-dasharray:3 4;stroke-width:1}
svg .markt{font:700 14px "Cabinet Grotesk",Arial;fill:var(--contraste);text-anchor:middle}
/* portada */
.cover{padding-top:64px}
.cover-logo{position:absolute;top:44px;right:72px;height:64px}
.cover h1{margin-top:58px}
.lede{font-size:26px;color:var(--contraste);margin-top:18px}
.cover-trace{position:absolute;left:0;right:0;top:372px;height:170px;background:#fff}
.cover-trace svg{display:block}
.meta{position:absolute;left:72px;bottom:96px;font-size:18px;font-weight:700;color:var(--azul)}
.cover-cap{position:absolute;left:72px;bottom:58px;margin:0}
/* 12 derivaciones */
.twelve{display:grid;grid-template-columns:repeat(2,560px);grid-auto-flow:column;grid-template-rows:repeat(6,70px);gap:6px 16px}
.lead-cell svg{display:block}
/* cinco diagnosticos */
.five{display:flex;flex-direction:column;gap:8px;width:1136px}
.dx{display:grid;grid-template-columns:250px 760px 1fr;align-items:center;gap:18px}
.dx-label b{display:block;font-size:20px}
.dx-label span{font-size:17px;color:var(--tinta)}
.dx-id{font-size:13px;color:var(--gris)}
.dx svg{display:block}
.facts{display:flex;gap:44px;margin-top:20px}
.facts p{font-size:19px;max-width:360px}
.facts b{font-size:28px;font-weight:800}
/* particion */
.split{display:flex;gap:8px;margin-top:8px;height:210px}
.part{padding:20px 22px;min-width:0;display:flex;flex-direction:column;gap:6px}
.part b{font-size:24px;font-weight:800}
.part span{font-size:16px;font-weight:700}
.part p{font-size:18px;line-height:1.35;margin-top:auto}
.train{background:var(--azul);color:#fff}.train b{color:#fff}.train span{color:var(--claro)}
.val{background:var(--cielo)}.val span{color:var(--contraste)}
.test{background:var(--amarillo);outline:2px solid #e9dc9a;outline-offset:-2px}.test span{color:#6b5d14}
.rules{list-style:none;margin-top:30px;display:flex;flex-direction:column;gap:12px}
.rules li{padding-left:22px;position:relative}
.rules li::before{content:"";position:absolute;left:0;top:13px;width:9px;height:9px;background:var(--contraste)}
/* latido */
.beat{position:absolute;right:72px;top:150px}
.beat svg{display:block}
.two{display:grid;grid-template-columns:1fr 1fr;gap:48px;width:500px}
.beat ~ .two{grid-template-columns:1fr;gap:30px;margin-top:20px}
.beat ~ .two p{font-size:21px}
.beat ~ .two h3{font-size:24px}
.beat ~ .two > div:first-child{padding-bottom:22px;border-bottom:1px solid var(--cielo)}
.beat ~ .caption{position:absolute;right:72px;top:446px;width:560px}
.two p{font-size:19px;margin-bottom:6px}
/* barras */
.bars{display:flex;flex-direction:column;gap:14px;margin-top:10px}
.bar-row,.cost-row{display:grid;grid-template-columns:220px 560px 190px 120px;align-items:center;gap:20px}
.bar-row.head{font-size:14px;font-weight:700;color:var(--gris)}
.bar-row .axis{display:flex;justify-content:space-between}
.bar-name{font-size:20px;font-weight:700;color:var(--gris)}
.bar-name small{display:block;font-size:13px;font-weight:400}
.bar-track{height:34px;background:var(--grisclaro);position:relative}
.bar{position:absolute;inset:0 auto 0 0;background:var(--medio)}
.net .bar{background:var(--contraste)}
.net .bar-name{color:var(--azul)}
.bar-val{font-size:26px;font-weight:800;color:var(--gris)}
.net .bar-val{color:var(--azul)}
.bar-val small{font-size:14px;font-weight:400}
.bar-auc{font-size:20px;color:var(--gris)}
.net .bar-auc{color:var(--azul);font-weight:700}
.cost-row{grid-template-columns:200px 480px 110px 1fr}
.cost{gap:26px;margin-top:20px}
.cost .bar-track{height:44px}
.cost-note{font-size:16px;color:var(--gris);line-height:1.35}
/* dumbbell */
.dumb-wrap{display:flex;gap:40px;align-items:flex-start}
.dumbbell .axisl{stroke:var(--cielo);stroke-width:1.2}
.dumbbell .axist{font:400 14px "Cabinet Grotesk",Arial;fill:var(--gris);text-anchor:middle}
.dumbbell .cls{font:700 21px "Cabinet Grotesk",Arial;fill:var(--azul)}
.dumbbell .cls.flat{fill:#b4231a}
.dumbbell .link{stroke:var(--claro);stroke-width:6}
.dumbbell .dot-base{fill:var(--medio)}
.dumbbell .dot-net{fill:var(--contraste)}
.dumbbell .delta{font:800 20px "Cabinet Grotesk",Arial;fill:var(--contraste)}
.dumbbell .delta.flat{fill:#b4231a}
.legend{width:320px;display:flex;flex-direction:column;gap:10px;padding-top:14px}
.legend p{font-size:19px;display:flex;align-items:center;gap:10px}
.legend i{display:inline-block;width:16px;height:16px;border-radius:50%}
.legend i.dot-base{background:var(--medio)}.legend i.dot-net{background:var(--contraste)}
.legend .note{font-size:16px;color:var(--gris);display:block;margin-top:8px}
/* figuras */
.figure{display:flex;justify-content:flex-start}
.figure img{max-height:500px;max-width:720px;display:block}
.figure.wide img{max-width:1136px;max-height:260px}
.caption.side{position:absolute;right:72px;top:150px;width:380px;font-size:18px}
.facts.three{margin-top:22px}
/* cierre */
.verdict{width:auto;grid-template-columns:1fr 1fr;gap:56px}
.verdict ul{list-style:none;display:flex;flex-direction:column;gap:12px}
.verdict li{font-size:20px;padding-left:22px;position:relative}
.verdict li::before{content:"";position:absolute;left:0;top:12px;width:9px;height:9px;background:var(--contraste)}
.verdict > div:last-child li::before{background:var(--medio)}
.verdict > div:last-child h3{color:var(--gris)}
.next{margin-top:34px;background:var(--amarillo);padding:16px 20px;font-size:20px}
.close .caption{font-size:13px;margin-top:22px}
"""


# ---------------------------------------------------------------- pdf

def print_pdf(html: Path, pdf: Path) -> None:
    browser = next((b for b in BROWSERS if b.exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        raise SystemExit("No se encontró Edge ni Chrome para imprimir el PDF.")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=5000", f"--print-to-pdf={pdf}", html.resolve().as_uri()],
                   check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(build_html(), encoding="utf-8", newline="\n")
    print(f"HTML: {HTML.relative_to(ROOT)}")
    if not args.html_only:
        print_pdf(HTML, PDF)
        print(f"PDF: {PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
