# Proyecto TAIA - Grupo 9 (Maestria IA, U. de La Sabana)

Lee SEGUIMIENTO.md: tiene la rubrica, las indicaciones del profe y el estado.
Objetivo: montar el proyecto de ML (tema propuesto: clasificacion de ECG con PTB-XL, 5 superclases) de forma reproducible.
- Usa uv (pyproject.toml), PyTorch, semillas fijas.
- Estructura: configs/ (YAML), src/, notebooks/, tests/, outputs/, README.md, .gitignore (datos y secretos fuera).
- Split por paciente (usar strat_fold de PTB-XL: 1-8 train, 9 val, 10 test). Ajustar escaladores solo con train.
- Linea base clasica (features + XGBoost/RF) vs CNN 1D (y opcional CNN+LSTM), mismo presupuesto.
- Metricas: F1 macro y por clase, ROC-AUC, matriz de confusion, calibracion, tiempo y costo.
- Ve marcando avances en SEGUIMIENTO.md. No hagas push a GitHub ni publiques nada sin que Alejandro lo pida.
- Textos (README, informe) en espanol natural, sin sonar a IA.

## Forma de trabajo
- Opus orquesta y decide: plan, decisiones de diseno experimental, revision de resultados y lo que se reporta.
- Delegar todo lo posible a subagentes con model haiku: descargas, scripts, tests, graficas, borradores de redaccion y revisiones mecanicas.
- Cada delegacion lleva instrucciones concretas (archivos, criterio de aceptacion, comando de verificacion); Opus revisa lo que devuelve antes de aceptarlo.
