# Clasificación de ECG con PTB-XL — Proyecto final TAIA (Grupo 9)

Maestría en IA, Universidad de La Sabana.

Clasificamos electrocardiogramas de 12 derivaciones en las 5 superclases diagnósticas de PTB-XL
(NORM, MI, STTC, CD, HYP). Es un problema **multietiqueta**: un mismo ECG puede tener, por ejemplo,
infarto (MI) y alteraciones del ST (STTC) a la vez.

## Datos

- **PTB-XL 1.0.3** (PhysioNet), 21.799 ECG de 18.869 pacientes, 10 s por registro. Licencia CC BY 4.0.
- Usamos la versión a 100 Hz (`records100`, ~0,5 GB). El script baja los CSV y esos registros desde el
  mirror oficial de PhysioNet en AWS (`physionet-open`), verifica cada archivo con `SHA256SUMS.txt` y se
  puede reanudar si se corta. El zip de physionet.org pesa 1,8 GB y con nuestra conexión bajaba a ~50 KB/s.
- Los datos no se suben al repo (`data/` está en `.gitignore`).

## Partición sin fuga

PTB-XL trae la columna `strat_fold` (10 folds estratificados) y garantiza que todos los registros de un
paciente caen en el mismo fold. Seguimos la recomendación de los autores:

| Partición | Folds | Uso |
|-----------|-------|-----|
| train | 1–8 | ajuste de modelos, imputador y escalador |
| val   | 9   | selección de hiperparámetros y umbrales |
| test  | 10  | solo para el reporte final |

El código verifica explícitamente que ningún `patient_id` aparezca en dos particiones (`ecg.data.check_no_patient_leakage`).

## Cómo correrlo

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # instala dependencias (versiones fijadas en uv.lock)
uv run python -m ecg.data                 # descarga PTB-XL, hace el split y cachea las señales
uv run python -m ecg.baseline             # línea base clásica (features + RF / XGBoost)
uv run pytest                             # pruebas
```

Toda la configuración (semilla, folds, hiperparámetros) está en `configs/default.yaml`.
Para probar otra configuración: `uv run python -m ecg.baseline --config configs/otra.yaml`.

## Estructura

```
configs/      configuración en YAML
src/ecg/      código: datos, features, métricas, modelos
tests/        pruebas (split sin fuga, features, métricas)
notebooks/    exploración
outputs/      resultados de cada corrida (métricas, figuras, modelos)
```

## Línea base clásica

Por cada una de las 12 derivaciones se extraen estadísticos de amplitud (media, desviación, percentiles,
asimetría, curtosis, RMS), cruces por cero y potencia en tres bandas de frecuencia (Welch). De la
derivación II se estima la frecuencia cardiaca y la variabilidad RR. Se agregan edad y sexo.
Con eso se entrena un Random Forest y un XGBoost (uno contra el resto por clase).

El umbral de decisión de cada clase se escoge en validación maximizando F1; test no se toca hasta el final.

Resultados en test (fold 10, n = 2.158, semilla 42, CPU AMD de 12 hilos):

| Modelo | F1 macro | ROC-AUC macro | Brier | ECE | Entrenamiento | Inferencia |
|--------|---------:|--------------:|------:|----:|--------------:|-----------:|
| Random Forest | 0,650 | 0,866 | 0,120 | 0,073 | 23 s | 0,12 ms/ECG |
| XGBoost       | 0,690 | 0,890 | 0,105 | 0,048 | 53 s | 0,02 ms/ECG |

F1 por clase con XGBoost: NORM 0,81 · MI 0,64 · STTC 0,71 · CD 0,68 · HYP 0,60. HYP, la clase menos
frecuente (12 %), es la más difícil. La extracción de las 210 features tarda unos 42 s para los 21.388 ECG.

## Métricas

F1 macro y por clase, ROC-AUC macro y por clase, matriz de confusión por clase, calibración (Brier, ECE y
diagrama de confiabilidad), tiempo de entrenamiento e inferencia. Los resultados quedan en `outputs/<modelo>/`.

## Reproducibilidad

- Semilla fija (`seed` en el YAML) para numpy, random y PyTorch.
- Mismas particiones para todos los modelos (folds oficiales).
- Dependencias fijadas en `uv.lock`.

## Referencias

- Wagner, P. et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7, 154.
- Strodthoff, N. et al. (2021). *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*. IEEE JBHI, 25(5).
