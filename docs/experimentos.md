# Bitácora de experimentos

Configuración base común (`base`): DCGAN 64×64, z de 100 dimensiones, Adam (lr 2e-4, β = 0.5/0.999) para G y D, batch 128, 200 épocas, semilla 23242, 5,000 imágenes Voxel Art. La rejilla de ruido fijo usa los mismos 16 z (`artifacts/fixed_z.pt`) en todas las corridas, y las métricas usan los mismos 2,000 z (semilla 777).

Cada experimento cambia **una sola cosa** respecto a `base`.

## Observaciones de `base` (BCE no saturante, sin normalización espectral)

- Genera personajes reconocibles desde la época ~40. FID = 151.8 y razón de diversidad = 0.966.
- D domina: D(x) ≈ 0.95 y D(G(z)) ≈ 0.05 durante casi todo el entrenamiento, y la pérdida de D se mantiene cerca de 0. El PDF lo señala como alarma.
- Aparecen picos periódicos en la pérdida de G (hasta ~30): D "gana" de golpe y G se recupera.
- En la rejilla fija, los colores de un mismo z cambian entre épocas (por ejemplo, muchas filas con pelo morado en la época 200). G salta entre modos en lugar de estabilizarse.

## Experimento 1: función de pérdida (`hinge`)

**Cambio:** BCE no saturante → pérdida hinge (D: `relu(1 − D(x)) + relu(1 + D(G(z)))`; G: `−D(G(z))`).

**Hipótesis (escrita antes de entrenar):** Creemos que la pérdida hinge **reducirá los picos de la pérdida de G y mejorará el FID** porque D deja de recibir gradiente cuando ya clasifica una muestra con margen 1, así que no puede volverse arbitrariamente confiado como con BCE, y G recibe gradientes más estables. Lo consideraremos útil si el FID baja respecto a `base` sin que caiga la razón de diversidad, y si las curvas muestran menos picos.

**Resultado:** hipótesis **parcialmente confirmada**.
- Las curvas son más suaves: desaparece el patrón de picos periódicos de `base`, y D(x) y D(G(z)) oscilan menos.
- La diversidad sube un poco (0.992 frente a 0.966), y en la rejilla fija se ven más rasgos distintos (lentes, más colores de pelo).
- El FID **no** mejora: 161.9 frente a 151.8. Con una sola semilla y 2,000 muestras no sabemos si 10 puntos de diferencia son ruido; habría que repetir con varias semillas.
- D sigue ganando con claridad (D(x) ≈ 0.92, D(G(z)) ≈ 0.10), así que hinge no resuelve la dominancia de D, solo la suaviza.

## Experimento 2: estabilización (`sn`)

**Cambio:** BCE igual que `base`, más normalización espectral en todas las convoluciones de D. No se cambia nada más.

**Hipótesis (escrita antes de entrenar):** Creemos que la normalización espectral **evitará que D sature (D(x) cerca de 1) y reducirá la oscilación entre modos** porque limita la constante de Lipschitz de D: sus salidas no pueden cambiar bruscamente ante cambios pequeños de la imagen, y G recibe una señal más suave. Lo consideraremos útil si D(x) y D(G(z)) quedan más cerca entre sí, los picos disminuyen y el FID baja sin perder diversidad.

**Resultado:** hipótesis **rechazada**. El entrenamiento colapsó.
- Hasta la época ~130 funciona: la rejilla muestra personajes variados (lentes, pelos de colores), comparables a `base`.
- Alrededor de la iteración 5,800 (época ~145), D gana por completo: D(x) → 1, D(G(z)) → 0 y la pérdida de D queda en 0 desde ahí hasta el final. La pérdida de G sube sin parar (de 5 a 9).
- **Colapso de modos:** en la época 170 los 16 z fijos producen el mismo personaje, y en la época 200 solo ruido. FID = 401.8 y razón de diversidad = 0.093.
- Los picos periódicos no desaparecieron; siguen igual que en `base` hasta el colapso.

**Interpretación (hipótesis, no verificada):** la normalización espectral limita la norma de los pesos de cada convolución, pero en nuestro D cada convolución va seguida de BatchNorm, que vuelve a escalar las activaciones. Así la restricción de Lipschitz no se cumple realmente. SN-GAN (Miyato et al., 2018) usa D **sin** BatchNorm. Al cambiar una sola cosa para respetar el diseño experimental, aplicamos la técnica en un contexto donde no es efectiva. Para verificarlo habría que correr SN sin BatchNorm en D.

**Lección:** no basta con elegir una técnica de estabilización; también importa su compatibilidad con el resto de la arquitectura. Además, el colapso no se ve en las métricas finales si solo se mira la última época: hay que vigilar las curvas y la rejilla durante todo el entrenamiento.
