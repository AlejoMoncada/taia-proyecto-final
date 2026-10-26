# Clasificación multietiqueta de ECG con PTB-XL
## Resumen y alcance
Proyecto final TAIA, Grupo 9. Maestría en Inteligencia Artificial, Universidad de La Sabana. Octubre de 2026.

Comparamos modelos clásicos y redes profundas para identificar cinco superclases diagnósticas en ECG de 12 derivaciones: NORM, MI, STTC, CD e HYP. Un registro puede tener varias etiquetas; no son cinco categorías excluyentes. El objetivo es medir calidad predictiva, calibración y costo con una partición por paciente, no construir un dispositivo médico.

La evidencia guardada muestra F1 macro de 0,652 para Random Forest, 0,692 para XGBoost por defecto, 0,690 para XGBoost ajustado, 0,744 para CNN 1D y 0,750 para CNN+LSTM. Son medias sobre tres semillas con el mismo test. Las redes muestran mejoras descriptivas frente a las características manuales, especialmente en MI y CD; HYP sigue siendo difícil.

El alcance de esa conclusión es limitado. No se igualó el tiempo total de cómputo ni la información de entrada: los modelos clásicos usan características y metadatos; las redes reciben la señal. No hay validación externa ni prueba estadística que permita afirmar superioridad o equivalencia entre las dos redes. La desviación entre semillas no es un intervalo de confianza sobre pacientes nuevos.

Este informe reúne diseño, ejecución y discusión. Los resultados son los ya guardados; no se reentrenó ni se tomó una decisión nueva usando test para preparar la entrega.
---
## Datos y unidad de evaluación
Usamos PTB-XL 1.0.3, publicado en PhysioNet bajo CC BY 4.0. El conjunto original contiene 21.799 ECG de 10 segundos y 12 derivaciones. Trabajamos a 100 Hz: cada registro contiene 1.000 muestras por derivación. La descarga usa el mirror S3 de PhysioNet y compara los archivos con los SHA256 oficiales. Los datos y pesos se conservan fuera de git.

Las superclases se obtienen de los códigos SCP diagnósticos mediante scp_statements.csv. Se descartan los registros sin ninguna de las cinco superclases, por lo que quedan 21.388 ECG. No se elimina la coexistencia de diagnósticos: 5.144 registros tienen más de una superclase.

| Partición | Folds | ECG | Pacientes |
|---|---|---|---|
| Entrenamiento | 1-8 | 17.084 | 14.823 |
| Validación | 9 | 2.146 | 1.917 |
| Prueba | 10 | 2.158 | 1.877 |

La unidad de entrada y de cálculo de las métricas es el ECG; la unidad que se mantiene separada entre particiones es el paciente. Un paciente puede tener varios registros. El programa comprueba que ningún patient_id aparezca en dos particiones.

Las etiquetas proceden de informes codificados, no de un diagnóstico clínico confirmado. En el CSV original, el 26,3 % de los registros no figura como validado por un humano. La edad 300 es una codificación de anonimización para mayores de 89 años, no una edad real. La concentración por sitio y dispositivo limita la generalización. Más detalle: ficha_datos.md y referencia [1].
---
## Protocolo experimental y métricas
El entrenamiento usa los folds 1-8 de strat_fold, validación el 9 y prueba el 10: 17.084, 2.146 y 2.158 ECG. El imputador y el escalador de los modelos clásicos se ajustan solo con train; la media y desviación por derivación de las redes también salen solo de train. Validación y test reciben esas mismas transformaciones.

La búsqueda evalúa ocho configuraciones aleatorias por familia para XGBoost, CNN 1D y CNN+LSTM, con semilla 123. La configuración se elige por ROC-AUC macro en validación. Cada configuración seleccionada se entrena con las semillas 42, 43 y 44. Random Forest es una referencia sin búsqueda. No se usa el test para elegir arquitectura, hiperparámetros, época o umbral.

Cada clase tiene un umbral elegido en validación al maximizar F1 en una rejilla de 0,05 a 0,95, con paso 0,01. Las predicciones multietiqueta usan esos cinco umbrales; no se aplica argmax. Los umbrales permanecen iguales al evaluar test.

- F1 macro da el mismo peso a cada superclase, incluso a HYP, menos frecuente. F1 por clase muestra dónde se concentra la mejora.
- ROC-AUC macro evalúa el ordenamiento sin fijar un umbral. Es el criterio de selección, distinto del F1 usado para describir resultados.
- Cada matriz de confusión es binaria por clase: TN, FP, FN y TP. Una matriz multiclase de cinco categorías sería incorrecta.
- Brier promedia el error cuadrático de las probabilidades por clase. ECE promedia el desajuste por clase usando diez intervalos de igual ancho. Menor es mejor; ECE depende de la discretización.
- Se informa tiempo de entrenamiento final, búsqueda e inferencia por ECG por separado. Las medias y desviaciones describen tres semillas, no tres particiones independientes.
---
## Modelos e hiperparámetros
La línea base extrae 210 características: estadísticos y percentiles de amplitud por derivación, cruces por cero, potencia relativa de Welch en 0,5-5, 5-15 y 15-40 Hz, frecuencia cardiaca y variabilidad RR estimadas de la derivación II, edad y sexo. La imputación evita perder registros por metadatos ausentes.

Random Forest usa 500 árboles, profundidad sin límite, al menos dos muestras por hoja y balanceo por submuestra. XGBoost por defecto usa 600 árboles, profundidad 6, tasa 0,05 y fracciones de filas y columnas de 0,8, uno contra el resto por clase. La búsqueda seleccionó profundidad 8, 600 árboles, tasa 0,05, ambas fracciones 0,8 y min_child_weight 1.

La CNN 1D combina bloques residuales y max pooling. La configuración elegida tiene canales [64, 64, 128, 128], kernel 5 y dropout 0,1: 363.781 parámetros. La cabeza concatena promedio y máximo globales y produce cinco logits.

CNN+LSTM usa canales [64, 64, 128], kernel 5, dropout 0,1 y una LSTM bidireccional con 128 unidades ocultas: 464.901 parámetros. Las convoluciones extraen patrones locales y la recurrencia permite integrar la secuencia resultante. Esa motivación no prueba que la LSTM aporte una mejora: eso debe contrastarse con los resultados.

Ambas redes usan BCE con logits, AdamW, tasa 0,001, decaimiento de pesos 0,001 y lotes de 128. ReduceLROnPlateau reduce la tasa a la mitad tras tres épocas sin mejora; la parada temprana tiene paciencia 8. El máximo es 60 épocas o cinco minutos por corrida y se restauran los pesos de la mejor época según AUC de validación.

El presupuesto iguala cantidad de configuraciones, NO tiempo total: CPU Ryzen 5 3600 y 32 GB de RAM para clásicos; GPU GTX 1660 SUPER de 6 GB para redes. Esta diferencia se reporta como limitación, no como igualdad de costo.
---
## Resultados agregados en prueba
La tabla usa el fold 10, con 2.158 ECG, y las semillas 42, 43 y 44. El símbolo ± es la desviación estándar entre semillas, también para Brier y ECE. Búsqueda es el tiempo acumulado de los ocho trials, separado del entrenamiento final medio.

{{results}}

{{figure:f1_per_class.png}}

Las ganancias descriptivas frente a XGBoost por defecto son 0,052 de F1 para CNN 1D y 0,058 para CNN+LSTM. En HYP no hay una mejora clara: los valores medios siguen cerca de 0,60. Ajustar XGBoost eleva el AUC de 0,891 a 0,894, pero no mejora F1 ni calibración. Optimizar AUC no garantiza mejorar todas las métricas.

El desbalance de PTB-XL, con NORM como la clase más frecuente y HYP como la menos frecuente, explica en parte el techo de HYP, que ronda F1 0,60 en todos los modelos. Las redes de esta corrida se entrenaron con `BCEWithLogitsLoss`, sin reponderación de clases ni pérdidas asimétricas, y no se aplicó recalibración postentrenamiento. Atacar esos dos frentes con una pérdida focal y con un escalado de temperatura ajustado en validación son las dos intervenciones que el grupo identificó como siguientes pasos; ninguna está aplicada en los resultados de arriba y por eso no se reportan cifras de ellas.
---
## Aprendizaje y selección en validación
{{figure:curves.png}}

Las curvas permiten comparar pérdida de entrenamiento y validación y comprobar la parada temprana. Se muestran por separado las semillas 42, 43 y 44, no una curva promedio. La evaluación final restaura la mejor época, que puede ser anterior al último punto dibujado.

En la búsqueda guardada, el mejor AUC macro de validación fue aproximadamente 0,903 para XGBoost, 0,933 para CNN 1D y 0,935 para CNN+LSTM. Se consultaron ocho configuraciones por familia, con el mismo criterio y semilla de búsqueda. Los archivos trials.csv permiten revisar los candidatos y sus tiempos sin mirar test.

Dos trials de CNN 1D con canales [32, 32, 64, 64] y kernel 7 tardaron cerca de 90 segundos por época; el límite temporal los detuvo en la época 4. Se sospecha una elección lenta de algoritmo con cuDNN determinista, pero no se verificó esa causa. No se repitieron ni se excluyeron de los tiempos de búsqueda. La limitación afecta la comparabilidad de los candidatos, aunque todos tengan el mismo tope temporal.
---
## Errores por clase
{{figure:confusion_cnn1d_seed42.png}}

Las matrices muestran la CNN 1D, semilla 42, en test. Cada panel evalúa una etiqueta por separado y conserva tanto verdaderos negativos como positivos. No se deben sumar los soportes de las cinco clases para estimar el número de ECG: hay registros multietiqueta.

Con CNN+LSTM, el F1 medio es 0,858 en NORM, 0,753 en MI, 0,768 en STTC, 0,758 en CD y 0,612 en HYP. Para CNN 1D es 0,851, 0,748, 0,765, 0,760 y 0,598, respectivamente. CNN+LSTM no gana en todas las clases: CD queda ligeramente mejor con CNN 1D.

HYP tiene menor prevalencia y errores importantes. El experimento no identifica una causa única: el desbalance, el ruido de etiqueta y las limitaciones de señal o representación pueden contribuir. No se probó una intervención específica que permita atribuirles causalidad.

La diferencia de F1 macro entre redes es 0,006. Esa cercanía descriptiva no demuestra que sean equivalentes ni que una sea estadísticamente superior. Una comparación posterior debería usar un bootstrap pareado por paciente sobre predicciones fijas, manteniendo juntos los registros de cada paciente y sin cambiar hiperparámetros con el resultado.
---
## Calibración y costo computacional
{{figure:reliability_test.png}}

Los diagramas corresponden a la semilla 42; la tabla agregada resume las tres semillas. CNN 1D y CNN+LSTM tienen ECE media de 0,040 y 0,039 frente a 0,049 de XGBoost por defecto. La variación entre semillas de las redes es 0,010 y 0,008. Estos valores no garantizan probabilidades confiables en otro hospital.

{{figure:times.png}}

La búsqueda tomó unos 319 s para XGBoost, 1.454 s para CNN 1D y 1.302 s para CNN+LSTM. El entrenamiento final medio tomó 92, 137 y 170 s, respectivamente. La inferencia media fue 0,024, 0,250 y 0,189 ms por ECG; los tiempos dependen del hardware y del procesamiento por lotes y no representan latencia de un sistema clínico completo.

No se aplicó recalibración postentrenamiento. Un escalado de temperatura futuro debe ajustarse exclusivamente con validación y congelarse antes de evaluar test; no corresponde elegirlo por la ECE obtenida en test. Se reporta calibración medida, no una recalibración que no se hizo.
---
## Discusión, riesgos y límites
Las redes muestran mejores métricas globales que las líneas base en esta partición. El resultado compara pipelines completos: señal cruda frente a 210 características con edad y sexo. No aísla solo el efecto de la arquitectura y no justifica reemplazar una lectura clínica.

No hay evaluación de desempeño por sexo, edad, dispositivo o sitio ni validación externa. Conocer la composición demográfica no equivale a demostrar equidad. En una ampliación se deben informar tamaño y positivos por clase de cada subgrupo; un AUC sin positivos o sin negativos debe registrarse como no estimable, nunca como cero.

Las semillas cuantifican variación del entrenamiento, no incertidumbre poblacional. Los pacientes se mantienen separados entre particiones, pero pueden tener varios ECG dentro de test; una inferencia estadística que remuestree registros como independientes subestimaría esa dependencia. No se realizó todavía esa prueba.

También limitan el estudio el origen y antigüedad de PTB-XL, el ruido de etiquetas, la codificación especial de edades, la búsqueda pequeña, los dos trials lentos y la desigualdad de cómputo. No se puede extrapolar a 500 Hz ni a otros equipos o poblaciones. La LSTM agrega parámetros y tiempo sin una ventaja concluyente frente a CNN 1D.

Se usaron datos públicos desidentificados. No se intentó reidentificar ni cruzar pacientes con otras fuentes. La licencia exige atribuir el dataset; los datos, metadatos individuales y pesos no se incluyen en este paquete documental. Los errores en HYP y la falta de evaluación externa excluyen cualquier uso diagnóstico o terapéutico.

Conclusión: el experimento respalda describir una mejora de las redes sobre estas líneas base, con los límites anteriores. Las ampliaciones propuestas son subgrupos, bootstrap pareado por paciente y recalibración en validación; ninguna se presenta como realizada. Fuentes: [1] Wagner et al., Scientific Data 7, 154 (2020), doi:10.1038/s41597-020-0495-6; [2] Strodthoff et al., IEEE JBHI 25(5), 1519-1528 (2021), doi:10.1109/JBHI.2020.3022989.
---
## Reproducción y cierre de la entrega
El código usa uv, Python 3.11 y dependencias fijadas en uv.lock; las corridas guardadas registran PyTorch 2.14.1+cu126. La configuración experimental está en configs/default.yaml. La evidencia de este paquete procede de la revisión fdff6cf y de los archivos locales de las corridas: evidence/manifest.json conserva sus hashes. Regenerar los PDF no vuelve a entrenar ni evalúa test otra vez.

```bash
uv sync
uv run python -m ecg.data
uv run python -m ecg.baseline --seeds 42 43 44
uv run python -m ecg.tune_xgb
uv run python -m ecg.deep search --model cnn1d
uv run python -m ecg.deep final --model cnn1d
uv run python -m ecg.deep search --model cnn_lstm
uv run python -m ecg.deep final --model cnn_lstm
uv run python -m ecg.compare
uv run pytest -q
```

Los pesos permanecen locales por la política del proyecto. La reproducción requiere volver a entrenar o recibirlos mediante un canal autorizado; el repositorio por sí solo no los distribuye. El repositorio de entrega está en GitHub (rama `entrega`), con acceso restringido al equipo mientras no se autorice otra cosa.

Declaración de uso de IA: el equipo usó asistentes de IA como apoyo de implementación y redacción, no como autor de las decisiones experimentales. Claude (Anthropic) escribió y revisó buena parte de `src/ecg/` y `tests/`, y redactó la primera versión de este informe y de las diapositivas. La locución del video de sustentación se generó con texto a voz (Gemini TTS, Google AI Studio) sobre un guion aprobado por el grupo, y el video se renderizó con HyperFrames. El tema, el diseño experimental, la partición por paciente, la elección de modelos e hiperparámetros, la lectura de los resultados y la validación final son del grupo; ninguna cifra se aceptó sin contrastarla con los artefactos de las corridas en `outputs/`. El contenido final es responsabilidad de los autores.

Referencias: [1] Wagner et al. (2020), doi:10.1038/s41597-020-0495-6; dataset PTB-XL 1.0.3, https://physionet.org/content/ptb-xl/1.0.3/. [2] Strodthoff et al. (2021), doi:10.1109/JBHI.2020.3022989. Evidencia interna: ficha_datos.md, ficha_modelo.md, evidence/ y configs/default.yaml.
