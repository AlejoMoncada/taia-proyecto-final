# Ficha de datos: PTB-XL 1.0.3

Ficha al estilo de Gebru et al. (2021), *Datasheets for Datasets*, adaptada a nuestro proyecto. Las cifras salen de `ptbxl_database.csv`, `scp_statements.csv` y `data/processed/split.csv`. Lo que no podemos comprobar con esos archivos aparece como lo dicen los autores.

## Resumen

PTB-XL es un conjunto público de electrocardiogramas (ECG) de 12 derivaciones y 10 segundos por registro. Lo usamos para clasificar cada ECG en las cinco superclases diagnósticas NORM, MI, STTC, CD y HYP. Es un problema multietiqueta: un mismo registro puede tener más de una superclase. Trabajamos con la versión a 100 Hz.

| Dato | Valor |
|---|---|
| Registros | 21.799 |
| Pacientes | 18.869 |
| Registros con superclase diagnóstica | 21.388 (de 18.617 pacientes) |
| Licencia | CC BY 4.0 |

## Motivación y creadores

PTB-XL lo publicaron Wagner et al. (2020) en *Scientific Data*, 7, 154. Lo creó la Physikalisch-Technische Bundesanstalt (PTB). Según los autores, el propósito es ofrecer un conjunto grande y abierto para entrenar y comparar métodos de análisis automático de ECG. PhysioNet lo distribuye; nosotros usamos la versión 1.0.3.

## Composición

Cada instancia es un registro de ECG con sus metadatos: edad, sexo, fecha de grabación, sitio, dispositivo, indicador de validación humana y los códigos SCP-ECG asignados.

El archivo `scp_statements.csv` contiene 71 sentencias SCP-ECG. De ellas, 44 son diagnósticas y se agrupan en las cinco superclases. Un registro sin ninguna superclase diagnóstica se descarta, como indica el código de `ecg.data` (práctica del benchmark). Son 411 registros, el 1,9 %.

Población, sobre los 21.799 registros:

- **Pacientes.** 16.758 pacientes tienen un solo registro y 2.111 tienen varios, con un máximo de 10. Hay 1,16 registros por paciente.
- **Sexo.** Usamos la codificación del conjunto, que según la documentación de PhysioNet es 0 = hombre y 1 = mujer. Hay 11.354 registros de hombres (52,1 %) y 10.445 de mujeres (47,9 %).
- **Edad.** Sin los valores codificados, la media es 59,5 años (desviación típica 16,8), la mediana 61 y el rango intercuartílico 50–72. Las edades van de 2 a 89. Los pacientes mayores de 89 aparecen con el valor 300: son 293 registros (1,3 %).
- **Fechas de grabación.** Van del 9 de noviembre de 1984 al 11 de junio de 2001.
- **Sitios.** Hay 51 códigos. Los tres más frecuentes (0, 1 y 2) reúnen el 93,2 % de los registros.
- **Dispositivos.** Hay 11 modelos. Los tres más frecuentes (CS100 3, CS-12 y AT-6 C 5.5) reúnen el 64,9 %.
- **Validación humana.** El 73,7 % de los registros (16.056) está validado por una persona.

Superclases sobre los 21.388 registros etiquetados:

| Superclase | Registros | Prevalencia |
|---|---:|---:|
| NORM | 9.514 | 44,5 % |
| MI | 5.469 | 25,6 % |
| STTC | 5.235 | 24,5 % |
| CD | 4.898 | 22,9 % |
| HYP | 2.649 | 12,4 % |

Hay 16.244 registros con una sola superclase, 4.068 con dos, 919 con tres y 157 con cuatro. En total, 5.144 registros (24,1 %) tienen más de una.

## Proceso de recolección

Lo que sigue lo dicen los autores; no lo podemos verificar con los archivos. Según ellos, los registros provienen de un único centro clínico en Alemania. Las sentencias SCP-ECG se asignaron con un programa de interpretación, y una parte de los registros fue revisada por personal médico. El CSV solo permite ver si hubo validación humana (campo `validated_by_human`), no quién revisó ni cómo.

## Preprocesamiento que hacemos nosotros

- **Frecuencia.** Usamos `records100`, 10 s a 100 Hz. Las señales se guardan en caché como `float32` en `data/processed/signals_100hz.npy`.
- **Etiquetas.** Las sentencias diagnósticas se mapean a sus superclases con `scp_statements.csv`. Cada superclase es una columna binaria (`ecg.data.load_metadata`).
- **Edad.** En `features.py`, toda edad mayor de 120 (el código 300) pasa a ser valor perdido. La línea base la imputa con la mediana de train, dentro del pipeline (`SimpleImputer`).
- **Sexo.** Lo usamos como número (0 o 1).
- **Línea base.** 210 características por registro: estadísticos de amplitud de cada derivación, cruces por cero, potencia de Welch en tres bandas y frecuencia cardiaca y variabilidad RR de la derivación II. Se agregan edad y sexo. El escalado (`StandardScaler`) se ajusta solo con train.
- **Redes profundas.** Está previsto normalizar cada derivación con medias y desviaciones de train. Ese paso todavía no está implementado en `src/ecg`.

## Particiones

Usamos los folds oficiales `strat_fold`: train = 1–8, val = 9 y test = 10. Todos los registros de un paciente caen en el mismo fold, y `ecg.data.check_no_patient_leakage` lo verifica. Excluimos 334, 37 y 40 registros sin superclase de train, val y test.

| Partición | Registros | Pacientes | NORM | MI | STTC | CD | HYP | Más de una superclase | Mujeres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 17.084 | 14.823 | 44,5 % | 25,6 % | 24,5 % | 22,9 % | 12,4 % | 24,2 % | 48,0 % |
| val | 2.146 | 1.917 | 44,5 % | 25,2 % | 24,6 % | 23,1 % | 12,5 % | 23,7 % | 48,1 % |
| test | 2.158 | 1.877 | 44,6 % | 25,5 % | 24,1 % | 23,0 % | 12,1 % | 23,5 % | 48,6 % |

Las prevalencias de las tres particiones son muy parecidas, así que el split no cambia la mezcla de clases. Los pacientes de las tres particiones suman 18.617 sin repetirse.

## Usos previstos y no previstos

Previsto: benchmark académico de clasificación multietiqueta, para comparar una línea base clásica con una CNN 1D y una CNN+LSTM dentro del curso, con un presupuesto de ajuste comparable.

No previsto: decisiones clínicas, diagnóstico de pacientes reales o sustituir la lectura de un cardiólogo. Ningún modelo que entrenemos aquí está validado clínicamente.

## Sesgos y limitaciones

- **Origen.** Un solo centro según los autores. La población, los equipos y las prácticas clínicas no representan otros hospitales ni otros países.
- **Época.** Las grabaciones son de 1984 a 2001. Los equipos y los protocolos actuales pueden ser distintos.
- **Dispositivos.** Tres modelos cubren el 64,9 % de los registros. Como el equipo puede estar asociado al sitio y este a la etiqueta, un modelo podría aprender el equipo en lugar del ECG.
- **Sexo y edad.** Hay 52,1 % de hombres y la edad se concentra entre 50 y 72 años. Los extremos tienen pocos casos.
- **Ruido de etiqueta.** El 26,3 % de los registros no tiene validación humana. Las etiquetas pueden tener errores, y no tenemos forma de medirlos con los archivos.
- **Desbalance.** HYP es la clase menos frecuente (12,4 %) y NORM la más frecuente (44,5 %).
- **Pacientes repetidos.** 2.111 pacientes tienen más de un registro. Un split por registro mezclaría pacientes entre train y test, por eso partimos por paciente.

## Privacidad

Según los autores, el conjunto está desidentificado. No incluye nombres ni documentos, pero sí tiene edad, sexo, fecha de grabación y un identificador numérico de paciente. Nuestra práctica: no intentamos reidentificar a nadie, no cruzamos el conjunto con otras fuentes y no generamos datos nuevos sobre personas. Los datos quedan en `data/`, que está en `.gitignore`.

## Distribución y licencia

PTB-XL 1.0.3 se distribuye bajo CC BY 4.0. Al compartir el conjunto o resultados derivados hay que dar crédito, indicar la licencia y citar a Wagner et al. (2020). No redistribuimos los datos: quien los necesite los descarga desde PhysioNet.

## Mantenimiento

La descarga la hace `uv run python -m ecg.data`. Baja desde el mirror de PhysioNet en AWS (`physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`) los CSV, la licencia y los registros de 100 Hz. Cada archivo se verifica contra `SHA256SUMS.txt`, y si se corta, la descarga se reanuda. Fijamos la versión 1.0.3. Si PhysioNet publica una versión nueva, esta ficha hay que revisarla antes de cambiar los datos.
