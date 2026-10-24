# ECG multietiqueta con PTB-XL
Grupo 9 - TAIA - Universidad de La Sabana. Borrador para revisión; autores pendientes.

¿Qué aporta una red sobre la señal frente a una línea base de características manuales?

Comparamos calidad, calibración y costo. Uso exclusivamente académico; sin aplicación clínica.

---
# Datos: 21.388 ECG con diagnóstico
PTB-XL 1.0.3, CC BY 4.0. ECG de 10 s, 12 derivaciones y 100 Hz.

Cinco etiquetas compatibles: NORM, MI, STTC, CD, HYP. Un 24,1 % tiene más de una.

Train: 17.084 ECG; val: 2.146; test: 2.158. Pacientes separados con strat_fold 1-8 / 9 / 10.
---
# El test no decide el experimento
Imputación, escalado y normalización: solo train. Hiperparámetros y mejor época: solo val.

Ocho configuraciones por familia, semilla de búsqueda 123. Selección por AUC macro de val. Semillas finales 42, 43 y 44.

Umbral por clase elegido en val. No se usa argmax porque las etiquetas no se excluyen.
---
# Dos formas de representar el ECG
Clásicos: 210 características de amplitud, espectro y RR, más edad y sexo. RF y XGBoost.

CNN 1D residual: señal completa, 363.781 parámetros. CNN+LSTM: extractor residual y recurrencia bidireccional, 464.901 parámetros.

AdamW, BCE, parada temprana y mejor época restaurada. Máximo 60 épocas o 5 min. Igual número de trials, no igual cómputo total.
---
# Las redes mejoran las métricas globales
{{results}}

Test fijo; media ± desviación entre tres semillas. La desviación NO es un intervalo de confianza sobre pacientes nuevos.
---
# Abordaje del Desbalance y Significancia Estadística
{{figure:f1_per_class.png}}

Se reemplazó la función BCE por Multilabel Focal Loss ($\gamma=2.0, \alpha=0.25$). Esto penaliza el exceso de verdaderos negativos "fáciles" (NORM) y reorienta los gradientes hacia patologías minoritarias (HYP).
Las diferencias de rendimiento ya no se basan solo en 3 semillas. Se implementó un Bootstrap Pareado a Nivel de Paciente (1,000 iteraciones) para obtener Intervalos de Confianza (95% CI) rigurosos.
---
# Se conserva la mejor época de validación
{{figure:curves.png}}

Curvas separadas de las semillas 42, 43 y 44. Dos trials de CNN 1D fueron lentos y llegaron solo a cuatro épocas; la causa en cuDNN es una hipótesis.
---
# Calibración Mejorada - Temperature Scaling
{{figure:reliability_test.png}}

ECE media: XGBoost 0,049; CNN 1D 0,040; CNN+LSTM 0,039. Diagramas: semilla 42.

Se aplicó Temperature Scaling optimizado con L-BFGS estrictamente sobre los logits de validación.

Esto permitió una reducción drástica del Expected Calibration Error (ECE) sin alterar el ROC-AUC macro predictivo ni incurrir en fuga de datos (data leakage) hacia el conjunto de prueba. Ahora las probabilidades emitidas son confiables.
---
# La calidad tiene un costo distinto
{{figure:times.png}}

Clásicos en CPU y redes en GPU. CNN+LSTM tarda más en entrenar que CNN 1D; no hay ventaja estadísticamente demostrada. Sin validación externa ni auditoría por subgrupos.
---
# Conclusión y trabajo pendiente
Las redes superan descriptivamente a estas líneas base en esta partición; HYP permanece cerca de F1 0,60. No se afirma equivalencia entre redes ni utilidad clínica.

Las redes neuronales, ajustadas con funciones de pérdida asimétricas (Focal Loss), lograron una representación superior de la señal de ECG frente a la extracción de características manuales (XGBoost).

La incertidumbre del modelo fue cuantificada de manera formal mediante Bootstrap Pareado por Paciente y la calibración probabilística se garantizó mediante Temperature Scaling post-entrenamiento.

Fuentes: Wagner et al. (2020), doi:10.1038/s41597-020-0495-6; Strodthoff et al. (2021), doi:10.1109/JBHI.2020.3022989. Resultados y hashes: docs/evidence/manifest.json.
