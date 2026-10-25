# SPEC — Locución ECG (TAIA Grupo 9)

GOAL:    El espectador entiende en 85 s qué hicimos, cómo lo comparamos y qué resultado obtuvimos, y termina confiando en que es reproducible.
OUTPUT:  HyperFrames, 1920x1080, 30 fps. Narración en español (voz Achird, en assets/).
TIME:    85.5 s x 30 fps = 2565 frames. La voz es el reloj: cada bloque de voz arranca con su escena.

LOOK:    El medio es papel de electrocardiografía:
         - papel: #F7ECE8 plano con retícula milimetrada roja (cuadrados chicos rgba(193,39,45,0.13), líneas fuertes cada 5 rgba(193,39,45,0.28)). La retícula es textura, nunca se anima.
         - tinta: #1C1714 para todo lo que se lee.
         - acento: #C1272D, el rojo del trazo. Racionado: el trazo y una cifra por escena.
         - display: Archivo Black. Cuerpo: Outfit 600. Sin serif.
         - profundidad: sin sombras de UI; la tinta "se imprime" (trazo dibujado o wipe, no fade flotante). Grano 0%, viñeta 0%.
OBJECTS: - El trazo de ECG: polilínea P-QRS-T real dibujada en rojo, no una onda genérica.
         - Las 12 derivaciones: parrilla de mini-trazos etiquetados (I..aVF, V1..V6).
         - Los estudios: puntos de tinta que se agrupan en 5 clases (NORM, MI, STTC, CD, HYP).
         - El diagrama de dos familias: una raíz y dos ramas que crecen parte por parte.
         - Los pacientes: fichas agrupadas; barreras que dividen folds 1-8 / 9 / 10.
         - Las barras de resultado: F1 macro XGBoost 0,69 vs CNN 1D 0,74 (test, media de 3 semillas).
MOTION:  - Entradas por impresión: trazos que se dibujan (dashoffset 0.4-4.6 s), cifras que aterrizan (scale 1.2->1, power4.out), texto que entra por wipe de clip-path.
         - Salidas: corte seco entre escenas; nada se desvanece en fila.
         - Cámara: empuje lento de settle por escena (scale 1.05->1, ease none).
         - Nada flota, nada respira; el papel no se mueve.
SHOTS:   S1 0.00-11.11  El trazo se dibuja solo; a los 6.5 s la palabra sale y entra la parrilla de 12 derivaciones.
         S2 11.11-25.90 La pregunta aterriza; los puntos se agrupan en 5 clases; la cifra 21.837 estudios (PTB XL) cae al final.
         S3 25.90-44.41 Diagrama que crece: raíz "misma señal" -> rama clásica (features, RF, XGBoost) -> rama de redes (señal cruda, CNN 1D, CNN+LSTM).
         S4 44.41-56.32 Fichas de pacientes; bajan las barreras de folds; una ficha intenta cruzar y rebota; sello "A CIEGAS" sobre test.
         S5 56.32-70.47 Barras de F1 macro con cifras 0,69 y 0,74; panel por clase: MI y CD suben, HYP plana en ~0,60.
         S6 70.47-85.50 El trazo vuelve y se aquietó; end card: "Reproducible", código/semillas/configuraciones, TAIA · Grupo 9. Se sostiene 2.5 s.
         Sonido: voz sola (bloque_1..6.wav en el root). Sin música por ahora: sin fuente de catálogo disponible; se puede agregar después.
BANNED:  void oscuro con elementos flotando; glow, neón, partículas; gradientes violeta/azul; glass cards; papel crema con serif + etiquetas mono + chips píldora + numeración 01/02; todo centrado; todo desvaneciéndose a la vez; texto moviéndose mientras se lee; tres tarjetas en fila; grilla de KPIs; logo esquinero.
QA:      frame 2 (papel + retícula, nada negro), frame 200 (trazo S1 dibujado), frame 480 (parrilla de 12), frame 780 (5 clases agrupadas), frame 1200 (diagrama de ramas completo), frame 1650 (barrera de folds + sello), frame 2100 (barras con cifras), frame 2500 (end card quieto).
