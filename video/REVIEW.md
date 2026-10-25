# REVIEW — Locución ECG (28-10... primera pasada)

## Llamadas creativas
- Medio elegido: papel de electrocardiografía (retícula roja milimetrada + tinta + acento rojo del trazo). Alternativas no construidas: "resultado primero" (arrancar con las barras y retroceder al dato) y "tipos cinéticos" sobre campos de color.
- La voz es el reloj: cada escena arranca con su bloque de audio (bloque_1..6.wav); los silencios de 0,7 s entre bloques quedan como respiración con el póster en pantalla.
- Sin subtítulos (la voz ya está en español; la pantalla no repite la frase textual) y sin música: no hay fuente de catálogo disponible en esta sesión. Sumar después si Alejandro quiere.
- S2: 120 puntos representativos, no 21.837; la cifra real la lleva el rótulo.
- S5: se omiten CNN+LSTM y RF en las barras (el guion compara XGBoost vs CNN 1D); HYP muestra "≈ 0,60" tal como dice la voz.

## Medido vs mirado
- `check`: 0 errores, 0 warnings en lint, runtime, layout, motion, craft; contraste 25/25 AA.
- `snapshot --review`: 12 beats + 5 costuras + hoja de teléfono, revisados en imagen. Corregido: puntos de S2 con patrón de diagonales (ahora anillos orgánicos), sublínea de S3 cruzada por las curvas (movida al pie), etiqueta del eje de S5 chocando con "0,69" (movida abajo), clips de S2 y sello de S4 que morían 0,1-0,2 s antes del corte (extendidos al fin de escena).
- El audio se colocó por tiempos medidos de los WAV; el render final nadie lo escuchó todavía. Verificar volumen de voz vs música si se agrega cama.

## Lo más débil
- S1 arranca con ~0,6 s de papel vacío antes del trazo (y lo mismo, acotado, en cada corte). Es respiración deliberada con la voz, pero si molesta, adelantar el dibujo a 0,3 s.
- El sello "A CIEGAS" de S4 podría entrar con un golpe seco de sonido cuando haya efectos.

## Warnings mantenidos a propósito
- Ninguno: el check final quedó sin warnings.

## Próximos pasos sugeridos
- Aprobar en Studio (preview en fondo corriendo) → `npm run render`.
- Decidir si se agrega música (catálogo HeyGen requiere cuenta) y subtítulos.
