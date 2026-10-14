# Seguimiento - Proyecto final TAIA (Unidad 4, Grupo 9)

Tema propuesto: clasificacion de ECG con PTB-XL (PhysioNet, CC BY 4.0), 5 superclases. Alternativa: somnolencia con MediaPipe + LSTM.
Cierre en Moodle: dom 18 oct 2026 23:59. El profe pedira ampliar a dom 25 oct (verificar en Moodle).

## Entregas (mismo responsable de la vez pasada sube)
1. Diseno experimental (formativa, id 302398): plan PDF max 6 pag. Problema/alcance, dataset (procedencia, licencia, limites), particion sin fuga, metricas justificadas, linea base con resultados, riesgos de sesgo/privacidad, arquitectura+cronograma+roles+reproducibilidad.
2. Ejecucion y validacion (evaluativa /5, id 302405): repo + PDF max 6 pag. Red avanzada justificada, hiperparametros fundamentados, curvas train/val, mejora vs linea base en tablas, informe con evidencia. No elegir modelo con test; semillas y mismas particiones.
3. Sustentacion (evaluativa /5, id 302406): informe 8-12 pag, repo reproducible (semillas, dependencias, pesos), 8-12 diapositivas, defensa sincronica (ausente = 0), ficha de modelo y datos, declarar uso de IA.

## Indicaciones del profe (clase 13 oct)
- Repo publico con link; los tres informes pueden ir en un solo PDF dentro del repo.
- Estructura: config (YAML/JSON), README, requirements, tests, outputs; secretos en .gitignore.
- Split por paciente/entidad; scaler/PCA ajustados solo con train.
- Comparacion justa (mismo split y presupuesto, igualar tiempo mejor que epochs); baseline clasica; reportar costo computacional.
- Desbalance: F1, matriz de confusion, ROC-AUC, calibracion post-entrenamiento.
- En Teams el grupo hablo (13 oct) de un video de "como estamos entrenando".

## Estado
- [ ] Tema elegido con el grupo (se avanzo con PTB-XL; falta confirmarlo con el grupo)
- [ ] Roles repartidos
- [x] Esqueleto del repo (14 oct): uv + pyproject/uv.lock, configs/default.yaml, src/ecg, tests, notebooks, outputs, README, .gitignore
- [x] Descarga PTB-XL + split por paciente (14 oct): `uv run python -m ecg.data`. Mirror S3 de PhysioNet con verificacion SHA256 (physionet.org iba a ~50 KB/s). Folds 1-8/9/10 y chequeo de que ningun paciente se repite.
  Resultado: 21.388 ECG con superclase; train 17.084 / val 2.146 / test 2.158, prevalencias parejas.
- [ ] Plan (formativa)
- [x] Linea base (14 oct): `uv run python -m ecg.baseline`. 210 features (estadisticos, bandas Welch, FC/RR, edad, sexo) + RF y XGBoost, umbrales por clase elegidos en val.
  Test: XGBoost F1 macro 0,690 / AUC 0,890 / ECE 0,048 (53 s); RF F1 0,650 / AUC 0,866 (23 s). Tabla en README, detalle en outputs/baseline/.
  Pendiente: correr con 2-3 semillas para reportar varianza.
- [ ] Modelo avanzado + experimentos
- [ ] Informe unico PDF
- [ ] Diapositivas
- [ ] Ensayo de sustentacion
