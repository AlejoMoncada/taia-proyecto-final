# Guía para ensayar la sustentación
Objetivo: contar el experimento en diez minutos y poder defender cada decisión. Es una propuesta de ensayo; el grupo debe confirmar el tiempo y repartir las intervenciones. Todos deben estar presentes según la rúbrica.

## Recorrido por las once diapositivas
Cada título es la idea que hay que dejar; el resto de la diapositiva es la evidencia. No leer el texto: contar el título y señalar la evidencia.
1. Pregunta (35 s): mostrar el ECG real de la portada y plantear la pregunta. Aclarar el uso académico.
2. Doce derivaciones (40 s): qué es una derivación, 10 s a 100 Hz, 12 × 1.000 números de entrada.
3. Cinco diagnósticos (50 s): nombrar las superclases sobre los trazados reales; 24,1 % tiene más de una etiqueta.
4. Partición (60 s): train ajusta, val elige, test solo se mira al final; pacientes separados y semillas fijas.
5. Dos lecturas (60 s): 210 características medidas a mano frente a la señal completa; P, QRS y T en el latido real.
6. Resultado (60 s): F1 macro de 0,69 a 0,75; qué mide la desviación entre semillas y por qué no declaramos ganador entre redes.
7. Por clase (55 s): la mejora está en infarto y conducción; hipertrofia no cambia, sin atribuir una causa.
8. Curvas (50 s): mejor época restaurada según AUC en val; los dos trials lentos.
9. Calibración (50 s): ECE de las tres familias; no se recalibró y qué haría falta.
10. Costo (50 s): hasta tres veces más entrenamiento; CPU frente a GPU, cómputo no igualado.
11. Cierre (50 s): lo que afirmamos, lo que no, y el siguiente paso. Total: unos 560 s.

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
