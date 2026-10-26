# ECG multietiqueta con PTB-XL
Grupo 9 - TAIA - Universidad de La Sabana.

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
# Desbalance y significancia estadística
{{figure:f1_per_class.png}}

Las redes se entrenaron con BCE sin reponderación de clases. HYP, la clase menos frecuente, se queda cerca de F1 0,60 en todos los modelos.

La desviación entre tres semillas mide variación del entrenamiento, no incertidumbre sobre pacientes nuevos. Falta una prueba de significancia entre modelos.
---
# Se conserva la mejor época de validación
{{figure:curves.png}}

Curvas separadas de las semillas 42, 43 y 44. Dos trials de CNN 1D fueron lentos y llegaron solo a cuatro épocas; la causa en cuDNN es una hipótesis.
---
# Calibración medida, no corregida
{{figure:reliability_test.png}}

ECE media: XGBoost 0,049; CNN 1D 0,040; CNN+LSTM 0,039. Diagramas: semilla 42.

No se aplicó recalibración postentrenamiento. Un escalado de temperatura ajustado en validación queda como trabajo pendiente.

Que las redes calibren mejor en esta partición no garantiza probabilidades confiables en otro hospital.
---
# La calidad tiene un costo distinto
{{figure:times.png}}

Clásicos en CPU y redes en GPU. CNN+LSTM tarda más en entrenar que CNN 1D; no hay ventaja estadísticamente demostrada. Sin validación externa ni auditoría por subgrupos.
---
# Conclusión y trabajo pendiente
Las redes superan descriptivamente a estas líneas base en esta partición; HYP permanece cerca de F1 0,60. No se afirma equivalencia entre redes ni utilidad clínica.

Las redes que leen la señal cruda rinden mejor que la extracción de características manuales con XGBoost, sin que eso demuestre superioridad clínica.

Ni la incertidumbre poblacional ni la recalibración se abordaron en esta entrega; quedan como trabajo propuesto.

Fuentes: Wagner et al. (2020), doi:10.1038/s41597-020-0495-6; Strodthoff et al. (2021), doi:10.1109/JBHI.2020.3022989. Resultados y hashes: docs/evidence/manifest.json.
