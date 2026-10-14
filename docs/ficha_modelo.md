# Ficha de modelo: clasificación de ECG con PTB-XL

Ficha al estilo de Mitchell et al. (2019), *Model Cards for Model Reporting*, adaptada a nuestro proyecto. Las cifras salen de los resultados de prueba del proyecto y de `configs/default.yaml`. Lo que no hemos medido aparece como pendiente.

## Detalles del modelo

Lo desarrolla el Grupo 9 de la Maestría en Inteligencia Artificial de la Universidad de La Sabana, para el curso TAIA. Fecha de la ficha: 2026. Los datos son PTB-XL 1.0.3 a 100 Hz (ver `docs/ficha_datos.md`). La versión del código se identifica por el commit de Git con el que se generaron los resultados; el hash final está pendiente. Licencia del código: pendiente del grupo.

Comparamos cinco modelos: tres clásicos sobre características hechas a mano y dos redes sobre la señal.

- **Random Forest.** Sobre 210 características hechas a mano. 500 árboles, profundidad sin límite, mínimo 2 muestras por hoja y pesos balanceados por submuestra. No pasó por búsqueda de hiperparámetros.
- **XGBoost por defecto.** Un clasificador por clase (uno contra el resto) sobre las mismas 210 características. 600 árboles, profundidad 6, tasa de aprendizaje 0,05, `subsample` y `colsample_bytree` de 0,8.
- **XGBoost ajustado.** Mismo modelo con la configuración que ganó la búsqueda en validación: profundidad 8, tasa de aprendizaje 0,05, 600 árboles, `subsample` y `colsample_bytree` de 0,8, `min_child_weight` 1.
- **CNN 1D.** Lee las 12 derivaciones en bruto. Bloques residuales (`ResBlock1d`) con canales [64, 64, 128, 128], kernel 5 y dropout 0,1, cada uno seguido de max pooling 2. La cabeza concatena la media y el máximo globales en el tiempo y da un logit por clase. Tiene 363.781 parámetros entrenables.
- **CNN+LSTM.** Mismo tipo de bloques residuales con canales [64, 64, 128], kernel 5 y dropout 0,1, seguidos de una LSTM bidireccional de 128 unidades ocultas. La cabeza usa la media y el máximo sobre la secuencia. Tiene 464.901 parámetros entrenables.

Las redes reciben 1.000 muestras por derivación (10 s a 100 Hz). Cada derivación se normaliza con la media y la desviación típica de entrenamiento (`lead_stats` en `src/ecg/deep.py`). Entrenamos con AdamW, tasa de aprendizaje 0,001, decaimiento de pesos 0,001 y pérdida BCE con logits. La tasa de aprendizaje se reduce a la mitad (factor 0,5, paciencia 3) cuando el ROC-AUC macro de validación se estanca. Usamos lotes de 128, hasta 60 épocas, parada temprana con paciencia 8 sobre el mismo AUC y un tope de 5 minutos de tiempo de pared por entrenamiento. Al terminar restauramos los pesos de la mejor época.

El hardware es un AMD Ryzen 5 3600 con 32 GB de RAM. Las redes usan una GPU NVIDIA GTX 1660 SUPER de 6 GB. Random Forest y XGBoost corren en CPU.

## Uso previsto

El modelo es un ejercicio académico. Sirve para comparar una línea base clásica con dos redes 1D sobre el mismo conjunto de datos y con el mismo presupuesto de búsqueda, dentro del curso.

## Usos fuera de alcance

No debe usarse para diagnóstico, para decidir tratamientos ni para evaluar a pacientes reales. No es un dispositivo médico y no sustituye la lectura de un cardiólogo. Tampoco debe usarse para ordenar casos reales de revisión: las probabilidades no están recalibradas y no hay validación externa.

Los resultados aplican a un solo centro de Alemania, según los autores de PTB-XL, con grabaciones de 1984 a 2001 y a 100 Hz. No sabemos cómo se comporta en otros hospitales, otros países, otros equipos o la frecuencia de 500 Hz.

## Factores

Los factores que nos interesan son sexo, edad, dispositivo y sitio. No hemos evaluado el desempeño por subgrupo: esa parte está pendiente. Solo conocemos la composición del conjunto (52,1 % de hombres, edad concentrada entre 50 y 72 años, tres modelos de equipo con el 64,9 % de los registros y tres sitios con el 93,2 %), que está en `docs/ficha_datos.md`.

## Métricas

- **F1 macro y F1 por clase.** Son la métrica principal, porque HYP pesa lo mismo que NORM aunque sea la clase menos frecuente. El umbral de cada clase se elige en validación, maximizando el F1 sobre una rejilla de 0,05 a 0,95 con paso 0,01. El conjunto de prueba se usa una sola vez, al final.
- **ROC-AUC macro.** No depende del umbral, así que compara los modelos sin mezclar la decisión de corte.
- **Brier.** Error cuadrático medio de las probabilidades. Es el promedio de las cinco clases.
- **ECE.** Error de calibración esperado con 10 bins de igual ancho, calculado por clase y promediado sobre las cinco. Importa si la probabilidad se usa para priorizar.
- **Tiempo.** Tiempo de entrenamiento final, tiempo de búsqueda e inferencia en milisegundos por ECG. Lo reportamos porque las familias no usan el mismo hardware y no igualamos el cómputo total.

## Datos de entrenamiento y evaluación

Usamos PTB-XL 1.0.3 a 100 Hz con los folds oficiales `strat_fold`. Entrenamiento son los folds 1 a 8 (17.084 ECG), validación el fold 9 (2.146) y prueba el fold 10 (2.158). Excluimos los registros sin superclase diagnóstica. `check_no_patient_leakage` verifica que ningún paciente aparezca en dos particiones. La mediana para imputar y el escalador de la línea base se ajustan solo con entrenamiento. Las estadísticas de normalización de las redes también salen solo de entrenamiento. Más detalle en `docs/ficha_datos.md`.

## Resultados

Test, media ± desviación típica sobre las semillas 42, 43 y 44 cuando la hay.

| Modelo | F1 macro | ROC-AUC macro | Brier | ECE | Entrenamiento (s) | Inferencia (ms/ECG) |
|---|---:|---:|---:|---:|---:|---:|
| RF (210 features, sin búsqueda) | 0,652 ± 0,002 | 0,866 ± 0,001 | 0,119 | 0,073 | 19 | 0,118 |
| XGBoost por defecto | 0,692 ± 0,002 | 0,891 ± 0,001 | 0,104 | 0,049 | 55 | 0,021 |
| XGBoost ajustado | 0,690 ± 0,004 | 0,894 ± 0,001 | 0,107 | 0,068 | 92 | 0,024 |
| CNN 1D | 0,744 ± 0,005 | 0,927 ± 0,001 | 0,085 | 0,040 ± 0,010 | 137 | 0,25 (GPU) |
| CNN+LSTM | 0,750 ± 0,003 | 0,928 ± 0,001 | 0,084 | 0,039 ± 0,008 | 170 | 0,19 (GPU) |

F1 por clase en prueba:

- CNN+LSTM: NORM 0,858, MI 0,753, STTC 0,768, CD 0,758, HYP 0,612 (± 0,015).
- CNN 1D: NORM 0,851, MI 0,748, STTC 0,765, CD 0,760, HYP 0,598 (± 0,026).
- XGBoost ajustado: NORM 0,816, MI 0,642, STTC 0,717, CD 0,678, HYP 0,597.

HYP es la clase más difícil para todos los modelos. Las redes ganan poco en HYP frente a XGBoost y mucho en MI y CD. La diferencia entre CNN 1D y CNN+LSTM (0,006 en F1 macro) es parecida a la variación entre semillas. No hicimos una prueba estadística entre modelos.

Las redes superan a los modelos clásicos en las cuatro métricas de calidad. Hay que leer esa comparación con cuidado: las redes ven la señal completa y los clásicos ven 210 características hechas a mano. El tiempo de entrenamiento de las redes es entre 1,5 y 1,9 veces el de XGBoost ajustado, y lo hacen en GPU.

## Ajuste de hiperparámetros

Cada familia tiene el mismo presupuesto: 8 configuraciones aleatorias con semilla 123, elegidas en `configs/default.yaml`. La selección usa el ROC-AUC macro en validación y el conjunto de prueba no participa. Después entrenamos la mejor configuración de cada familia con las semillas 42, 43 y 44.

Los mejores resultados en validación fueron XGBoost 0,903 (trial 7), CNN 1D 0,933 (trial 6) y CNN+LSTM 0,935 (trial 2). Los tiempos de búsqueda fueron 319 s para XGBoost, unos 1.450 s para CNN 1D y unos 1.300 s para CNN+LSTM. En las redes, el AUC de validación de los trials quedó entre 0,920 y 0,935, así que la arquitectura importa poco dentro del espacio de búsqueda.

Hay un problema conocido. Dos trials de CNN 1D (canales [32, 32, 64, 64], kernel 7) tardaron unos 90 s por época, frente a unos 4 s del resto. Sospechamos que cuDNN, con su modo determinista, elige un algoritmo lento para esa forma. El tope de 5 minutos los detuvo en la época 4, con AUC de validación de 0,924 y 0,920 y todavía subiendo. El mismo ancho con kernel 5 llegó a 0,931, por debajo del 0,933 de la configuración elegida. Es poco probable que hubieran ganado, pero no se evaluaron en igualdad de condiciones.

## Consideraciones éticas

Los datos son públicos, están desidentificados y se distribuyen con licencia CC BY 4.0. No intentamos reidentificar a nadie ni cruzar el conjunto con otras fuentes.

Las etiquetas vienen de informes de cardiología codificados en SCP-ECG, no de un diagnóstico confirmado. El 26,3 % de los registros no tiene validación humana, así que parte de las etiquetas puede tener errores. El sesgo de origen es el riesgo principal: un solo centro, pocos equipos y la mitad de los registros con edades entre 50 y 72 años. Un modelo puede aprender el equipo o el sitio en lugar de la fisiología.

Con un F1 de 0,61 en HYP, el modelo falla de forma importante en esa clase. Un uso real tendría que medir los falsos negativos por clase con población local, y eso no está hecho.

## Limitaciones y recomendaciones

- No hay evaluación por sexo, edad, dispositivo ni sitio. Hasta hacerla, no conviene sacar conclusiones sobre equidad.
- No hay prueba estadística entre modelos. La diferencia entre CNN 1D y CNN+LSTM está dentro de la variación entre semillas.
- La partición es fija. No estimamos cuánto cambiaría el resultado con otro reparto de pacientes. Con 2.158 ECG en prueba, esa variación puede ser mayor que la de las semillas.
- Las probabilidades no están recalibradas. Recomendamos probar escalado de temperatura en validación antes de usar las probabilidades. La ECE de las redes varía bastante entre semillas.
- El cómputo no está igualado: las redes usan GPU y los clasificadores clásicos CPU. Reportamos el número de configuraciones y el criterio de selección, no el costo total.
- Los dos trials de CNN 1D afectados por el problema de velocidad no se evaluaron en igualdad de condiciones. Si se repite la búsqueda, conviene fijar un algoritmo de convolución que no sea lento para esas formas, o documentar el cambio.
- No hay validación externa ni resultados a 500 Hz.

## Reproducción

```bash
uv sync
uv run python -m ecg.data
uv run python -m ecg.tune_xgb
uv run python -m ecg.deep search --model cnn1d
uv run python -m ecg.deep final --model cnn1d
uv run python -m ecg.deep search --model cnn_lstm
uv run python -m ecg.deep final --model cnn_lstm
uv run python -m ecg.compare
```

`uv run python -m ecg.compare` arma la tabla y las figuras de comparación en `outputs/comparison/`. Las figuras por semilla salen en `outputs/deep/final/<modelo>/seed_<semilla>/`. Pendiente: registrar el hash del commit final.
