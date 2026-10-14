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
- [ ] Tema elegido con el grupo
- [ ] Roles repartidos
- [ ] Plan (formativa)
- [ ] Linea base
- [ ] Modelo avanzado + experimentos
- [ ] Informe unico PDF
- [ ] Diapositivas
- [ ] Ensayo de sustentacion
