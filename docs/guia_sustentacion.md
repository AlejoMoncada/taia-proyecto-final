# Guía para ensayar la sustentación
Objetivo: contar el experimento en diez minutos y poder defender cada decisión. Es una propuesta de ensayo; el grupo debe confirmar el tiempo y repartir las intervenciones. Todos deben estar presentes según la rúbrica.

## Recorrido por las diez diapositivas
1. Problema y alcance (40 s): explicar por qué el ECG es multietiqueta y aclarar el uso académico.
2. Datos (60 s): procedencia, licencia, 100 Hz, exclusiones y separación por paciente.
3. Protocolo (70 s): distinguir train, val y test; explicar selección por AUC y umbrales por F1 en val.
4. Modelos (70 s): describir características manuales, bloques residuales y función de la LSTM sin leer código.
5. Resultados (70 s): comparar medias y aclarar qué mide la desviación entre semillas.
6. Clases (55 s): explicar MI/CD e HYP, sin atribuir una causa que no se investigó.
7. Curvas (55 s): ubicar sobreajuste, mejor época y el problema de los trials lentos.
8. Calibración (55 s): una probabilidad alta no asegura corrección; ECE depende de sus bins.
9. Costo y límites (65 s): CPU/GPU y cómputo desigual; no confundir rapidez con efectividad clínica.
10. Cierre (60 s): conclusión acotada, pendientes y declaración real del uso de IA. Total: 600 s.

## Preguntas que cada integrante debe poder responder
- ¿Por qué no usar accuracy o una matriz de confusión de cinco clases? Las etiquetas coexisten y las clases tienen prevalencias distintas; se usan F1 macro y matrices binarias por clase.
- ¿Dónde podría haber fuga? Pacientes repetidos entre folds, escaladores ajustados con todos los datos o decisiones basadas en test. Mostrar strat_fold, chequeo de pacientes y ajuste solo en train.
- ¿La LSTM ganó? Tiene una media algo mayor, pero no hay prueba estadística de superioridad ni equivalencia. Tres semillas no son tres muestras independientes de pacientes.
- ¿Fue justo el presupuesto? Se igualaron trials y criterio de selección, no tiempo total ni hardware. Reconocer esa limitación y los dos trials lentos.
- ¿Se recalibró? No. Se midieron Brier/ECE; cualquier ajuste posterior debe entrenarse en val, no elegirse por test.
- ¿Hay sesgo? No se midió desempeño por subgrupos. La composición de datos no demuestra equidad; tampoco hay validación externa.
- ¿Podría usarse con pacientes? No: no hay validación clínica, externa ni evaluación de seguridad; HYP conserva errores importantes.

## Antes de subir
- Confirmar autores, tema, roles, cronograma, declaración de IA y fecha vigente en Moodle. La prórroga no está verificada.
- Leer las fichas, abrir los cuatro PDF, comprobar páginas y contrastar las cifras con evidence/table_test.csv.
- Ensayar una vez con cronómetro y otra con preguntas cruzadas. No inventar contribuciones individuales ni resultados pendientes.
- Publicar o subir solo con autorización de Alejandro y por el responsable acordado; preparar pesos aparte solo por un canal autorizado.
