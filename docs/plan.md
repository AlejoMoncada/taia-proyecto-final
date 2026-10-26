# Plan de diseño experimental: clasificación multietiqueta de ECG con PTB-XL

Proyecto final TAIA — Grupo 9 — Maestría en IA, Universidad de La Sabana

14 de octubre de 2026. Grupo 9.

## 1. Problema y alcance

Queremos clasificar electrocardiogramas (ECG) de 12 derivaciones en cinco superclases diagnósticas: NORM (normal), MI (infarto de miocardio), STTC (alteraciones del segmento ST y de la onda T), CD (trastornos de la conducción) e HYP (hipertrofia). Es un problema multietiqueta, porque un mismo registro puede tener dos o más superclases a la vez.

Elegimos este tema porque PTB-XL es un dataset público, grande y con un protocolo de particiones definido, lo que permite comparar modelos de forma seria. Un clasificador automático de estas superclases podría ordenar casos para revisión, pero eso no es lo que evaluamos aquí.

Quedan fuera de alcance tres cosas. No proponemos uso clínico ni diagnóstico, y el modelo no es un dispositivo médico. Trabajamos solo con la versión de 100 Hz, aunque PTB-XL también trae 500 Hz. Y predecimos las cinco superclases, no los 71 códigos SCP originales.

## 2. Datos

PTB-XL 1.0.3 está publicado en PhysioNet bajo licencia CC BY 4.0 y lo describen Wagner et al. (2020). Tiene 21.799 registros de 10 s, con 12 derivaciones, de 18.869 pacientes. Usamos la versión de 100 Hz (`records100`), descargada del mirror S3 de PhysioNet. Cada archivo se verifica con su SHA256 oficial. Los datos no se suben al repositorio.

Tras descartar los registros sin superclase diagnóstica quedan 21.388 ECG. La tabla muestra cada partición con su prevalencia por superclase, en porcentaje de ECG de la partición.

| Partición | Folds | ECG | Pacientes | NORM | MI | STTC | CD | HYP |
|-----------|-------|----:|----------:|-----:|---:|-----:|---:|----:|
| train | 1–8 | 17.084 | 14.823 | 44,5 | 25,6 | 24,5 | 22,9 | 12,4 |
| val   | 9   | 2.146  | 1.917     | 44,5 | 25,2 | 24,6 | 23,1 | 12,5 |
| test  | 10  | 2.158  | 1.877     | 44,6 | 25,5 | 24,1 | 23,0 | 12,1 |

Las prevalencias son parejas entre particiones, como esperábamos con la estratificación de los autores. Un 24,1 % de los ECG (5.144 de 21.388) tiene más de una superclase, así que la multietiqueta no es un caso raro.

Los límites del dataset que medimos en el CSV son estos. La columna `validated_by_human` marca como revisados por un humano 16.056 de los 21.799 registros (73,7 %). El 26,3 % restante no fue validado así, lo que puede introducir ruido en las etiquetas. La edad viene codificada como 300 en 293 registros, que corresponden a pacientes mayores de 89 años. Hay 51 valores distintos de `site`, y los tres más frecuentes concentran el 93,2 % de los registros. Hay 11 valores distintos de `device`, y los tres más frecuentes suman el 64,9 %. Las fechas de registro en el CSV van de 1984 a 2001.

Según los autores, los registros provienen de un solo país, Alemania, y se tomaron con equipos Schiller. Las etiquetas salen de los informes de cardiología codificados en SCP-ECG, no de un diagnóstico clínico confirmado.

## 3. Partición sin fuga

Usamos la columna `strat_fold`. Los folds 1 a 8 son entrenamiento, el 9 es validación y el 10 es prueba. Según los autores, todos los registros de un mismo paciente caen en el mismo fold. Aun así lo comprobamos en código: `check_no_patient_leakage` falla si un `patient_id` aparece en dos particiones.

El imputador (mediana) y el escalador se ajustan solo con entrenamiento y luego se aplican a validación y prueba. Los umbrales de decisión por clase se eligen en validación, maximizando F1. El conjunto de prueba se usa una sola vez, al final, para el reporte. Nunca lo usamos para elegir modelo ni hiperparámetros.

## 4. Métricas y por qué las usamos

| Métrica | Para qué la usamos |
|---------|--------------------|
| F1 macro | Promedia el F1 de las cinco clases con el mismo peso. HYP pesa lo mismo que NORM, aunque HYP sea el 12 % del conjunto. |
| F1 por clase | Muestra dónde falla el modelo. Un F1 macro razonable puede esconder una clase mala. |
| ROC-AUC macro | No depende del umbral, así que compara modelos sin mezclar la decisión de corte. |
| Matriz de confusión por clase | Muestra falsos positivos y falsos negativos con el umbral final. |
| Brier y ECE | Miden la calidad de las probabilidades. ECE usa 10 bins de igual ancho. Importa si la probabilidad se usa para priorizar casos. |
| Diagrama de confiabilidad | Muestra visualmente la calibración por clase. |
| Tiempo y costo | Tiempo de entrenamiento, tiempo de búsqueda e inferencia por ECG, con el hardware usado. |

## 5. Línea base con resultados

La línea base usa 210 features hechas a mano. Para cada una de las 12 derivaciones calculamos 9 estadísticos de amplitud y cruces por cero, 5 percentiles y la potencia relativa en tres bandas (0,5–5, 5–15 y 15–40 Hz). A partir de la derivación II estimamos la frecuencia cardiaca y la variabilidad RR. Agregamos edad y sexo. Entrenamos un Random Forest y un XGBoost, uno contra el resto por clase.

Los resultados en prueba son la media ± desviación sobre las semillas 42, 43 y 44. Las particiones son las mismas en las tres corridas.

| Modelo | F1 macro | ROC-AUC macro | Brier | ECE | Entrenamiento | Inferencia (ms/ECG) |
|--------|---------:|--------------:|------:|----:|--------------:|--------------------:|
| Random Forest | 0,652 ± 0,002 | 0,866 ± 0,001 | 0,119 ± 0,000 | 0,073 ± 0,000 | 18,5 ± 1,6 s | 0,118 |
| XGBoost       | 0,692 ± 0,002 | 0,891 ± 0,001 | 0,104 ± 0,001 | 0,049 ± 0,001 | 54,5 ± 1,8 s | 0,021 |

Con XGBoost, el F1 por clase en prueba es NORM 0,810, MI 0,646, STTC 0,712, CD 0,679 e HYP 0,612. HYP es la clase más difícil y también la menos frecuente.

XGBoost supera a Random Forest en todas las métricas de la tabla, y la diferencia es mucho mayor que la variación entre semillas. La variación de ±0,002 en F1 solo refleja la semilla del modelo. Como las particiones son fijas, no medimos cuánto cambiaría el resultado con otro reparto de pacientes. Con 2.158 ECG en prueba, esa variación podría ser mayor, y la estimaremos más adelante.

## 6. Modelos propuestos y comparación justa

Además de la línea base, proponemos dos redes en 1D que leen las 12 derivaciones a 100 Hz (1.000 muestras por derivación), normalizadas por derivación con la media y la desviación de entrenamiento. La CNN 1D apila bloques residuales (dos convoluciones con normalización por lotes y un atajo) seguidos de max pooling, que reducen la señal a la mitad en cada bloque. Al final se concatenan el promedio y el máximo globales y una capa lineal da un logit por clase. Los valores de partida son canales [64, 64, 128, 128], kernel 7, dropout 0,2, tasa de aprendizaje 0,001 y decaimiento de pesos 0,0001.

La CNN+LSTM usa tres bloques del mismo tipo como extractor y encima una LSTM bidireccional que recorre la secuencia resultante, con el fin de capturar la relación entre latidos que las convoluciones ven solo localmente. Ambas redes se entrenan con BCE por clase, AdamW y reducción de la tasa de aprendizaje cuando el ROC-AUC de validación se estanca.

Las redes se entrenan con lotes de 128, hasta 60 épocas, con parada temprana si el ROC-AUC macro en validación no mejora durante 8 épocas, y se conservan los pesos de la mejor época. Cada entrenamiento tiene un límite de 5 minutos de tiempo de pared.

La comparación entre familias sigue estas reglas:

| Aspecto | Regla |
|---------|-------|
| Particiones | Folds oficiales 1–8 / 9 / 10 para todas las familias |
| Búsqueda | 8 configuraciones aleatorias por familia (XGBoost, CNN 1D y CNN+LSTM), semilla 123 |
| Criterio de selección | ROC-AUC macro en validación |
| Semillas finales | 42, 43 y 44 con la mejor configuración de cada familia |
| Presupuesto | Mismo número de configuraciones por familia; cada entrenamiento de una red tiene un tope de 5 minutos de tiempo de pared |
| Prueba | No participa en la búsqueda ni en la selección |

No igualamos el tiempo total entre familias: XGBoost necesita mucho menos tiempo que las redes para entrenar, y esa diferencia de costo es parte de lo que reportamos. Todos los tiempos de entrenamiento, búsqueda e inferencia quedan registrados. Las corridas se hacen en un AMD Ryzen 5 3600 con 32 GB de RAM. Las redes usan una GPU NVIDIA GTX 1660 SUPER de 6 GB.

Primeros resultados en prueba (media ± desviación sobre tres semillas): XGBoost ajustado logra F1 macro 0,690 ± 0,004 y ROC-AUC macro 0,894 ± 0,001. La CNN 1D logra 0,744 ± 0,005 y 0,927 ± 0,001. La CNN+LSTM logra 0,750 ± 0,003 y 0,928 ± 0,001. La búsqueda eligió kernel 5 y dropout 0,1 en ambas redes. El detalle está en `docs/ficha_modelo.md`, y el análisis completo queda para la entrega 2.

## 7. Riesgos de sesgo y privacidad

El sesgo de origen es el riesgo principal. Los datos vienen de un solo país según los autores, con pocos sitios y pocos modelos de equipo que concentran casi todos los registros. Un modelo puede aprender el equipo o el sitio en lugar de la fisiología. La distribución por edad y sexo tampoco la hemos auditado todavía.

Otros riesgos son el ruido de etiqueta, por el 26,3 % de registros no validados por un humano, y el desbalance de clases, con HYP en 12,4 % frente a NORM en 44,5 %.

En privacidad, los datos están desidentificados y son públicos bajo CC BY 4.0. No recolectamos datos personales nuevos, y `data/` está en `.gitignore`. El modelo no es un dispositivo médico y no debe usarse para decidir tratamientos.

Las mitigaciones hechas hasta ahora son las métricas por clase y las matrices de confusión por clase de la línea base. Quedan planeados, pero no hechos, un reporte de desempeño por sexo y grupo de edad, y un análisis por sitio y por equipo.

Con base en los primeros resultados experimentales, que mostraron debilidades en la calibración probabilística (ECE) y en la clase sub-representada HYP, se identificaron tres intervenciones para una etapa posterior. Ninguna está aplicada en los resultados que se reportan:

 * En la función de pérdida: reemplazar Binary Cross Entropy (BCE) por Focal Loss, para equilibrar la penalización asimétrica de clases mayoritarias.

 * En la estimación de confianza: ajustar Temperature Scaling en validación, para que las probabilidades emitidas sean interpretables en un futuro sistema de triaje médico.

 * En la validación estadística: pasar de la desviación estándar entre semillas a un Bootstrap Pareado a Nivel de Paciente. Esto respeta la dependencia entre registros del mismo paciente y permite reportar intervalos de confianza del 95 %.

## 8. Arquitectura del código y reproducibilidad

El código está en `src/ecg/`: `data.py` (descarga, etiquetas y partición), `features.py` (extracción de features), `baseline.py` (línea base), `metrics.py` (métricas y calibración), `search.py` y `tune_xgb.py` (búsqueda), `models.py` y `deep.py` (redes y su entrenamiento). La configuración vive en `configs/default.yaml`. Las pruebas están en `tests/` y los resultados en `outputs/`.

Las dependencias están fijadas con uv (`pyproject.toml` y `uv.lock`). La semilla base es 42 y se fija en numpy, random y PyTorch. Los comandos son estos:

```bash
uv sync
uv run python -m ecg.data
uv run python -m ecg.baseline --seeds 42 43 44
uv run python -m ecg.tune_xgb
uv run python -m ecg.deep search --model cnn1d
uv run python -m ecg.deep final --model cnn1d
uv run pytest
```

Cada corrida deja en `outputs/baseline/seeds/` un `runs.csv` con todas las corridas, un `summary_seeds.csv` con medias y desviaciones, y por modelo un `metrics.json`, la matriz de confusión y el diagrama de calibración.

## 9. Cronograma y roles

### Cronograma

Pendiente: definir con el grupo.

| Semana/Fecha | Actividad | Responsable |
|--------------|-----------|-------------|
|              |           |             |

### Roles

Pendiente: definir con el grupo.

## 10. Uso de IA

El equipo usó asistentes de IA como apoyo de implementación y redacción, no como autor de las decisiones experimentales.

- **Código y pruebas.** Claude (Anthropic) escribió y revisó buena parte de `src/ecg/` y `tests/`, con subagentes del mismo asistente para tareas mecánicas: descarga y verificación del dataset, gráficas y revisiones.
- **Redacción y documentos.** Las primeras versiones de `README.md`, `docs/plan.md`, `docs/informe.md` y `docs/diapositivas.md` se redactaron con ese asistente y después los revisó el grupo. El generador `scripts/build_delivery.py` es asistido.
- **Video de sustentación.** La locución se generó con texto a voz (Gemini TTS, Google AI Studio) sobre un guion aprobado por el grupo, y el video se renderizó con HyperFrames.
- **Decisiones humanas.** El tema, el diseño de la comparación, la partición por paciente, la elección de modelos e hiperparámetros, la lectura de los resultados y la validación final son del grupo. Ninguna cifra del informe se aceptó sin contrastarla con los artefactos de las corridas en `outputs/`.
- **Límites.** Las herramientas de IA pueden introducir errores de forma o de razonamiento; los que queden en este documento son responsabilidad del grupo. No se usó IA para generar datos, etiquetas ni resultados.

## 11. Referencias

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. Ch., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., y Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.

Strodthoff, N., Wagner, P., Schaeffter, T., y Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*, 25(5), 1519–1528.

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., y Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154.
