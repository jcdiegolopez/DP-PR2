# Galería final y prueba de vecinos más cercanos

## Modelo y selección

- **Modelo final:** corrida `hinge` (DCGAN con pérdida hinge), pesos en `artifacts/runs/hinge/G_final.pt`.
- **Por qué `hinge` y no `base`:** la pregunta central del proyecto es si los personajes son nuevos y **variados**. `hinge` tiene la mayor razón de diversidad (0.992 frente a 0.966) y más variedad visual (pelo largo, lentes, cuadros, tonos de piel). `base` tiene un FID 10 puntos mejor, pero con una sola semilla esa diferencia puede ser ruido.
- **Tasa de selección:** **10 de 200 (5 %)**. Los candidatos son las semillas 100000 a 100199 (`artifacts/figures/hinge_candidates.png`).
- **Criterio:** (1) rostro legible y poco ruido de color, y (2) máxima variedad entre los 10, sin dos que compartan la misma combinación de pelo o sombrero, piel, ropa y accesorios.

| Personaje | Candidato | Semilla | Rasgo distintivo |
|---|---|---|---|
| personaje_01 | 36 | 100036 | Pelo negro largo, ropa oscura |
| personaje_02 | 100 | 100100 | Pelo negro con flequillo, camisa morada |
| personaje_03 | 30 | 100030 | Gorro verde y lentes |
| personaje_04 | 13 | 100013 | Camisa a cuadros, sonrisa con dientes |
| personaje_05 | 3 | 100003 | Ojos de estrella, piel oscura, camisa naranja |
| personaje_06 | 62 | 100062 | Sombrero plano negro, piel oscura |
| personaje_07 | 103 | 100103 | Lentes, pelo claro, camisa verde |
| personaje_08 | 102 | 100102 | Gorro verde, cuadros, boca abierta |
| personaje_09 | 140 | 100140 | Ojos grandes, piel oscura, camisa azul |
| personaje_10 | 151 | 100151 | Pelo negro, piel clara, camisa naranja |

Regenerar: `python src/gallery.py final --run hinge --pick 36 100 30 13 3 62 103 102 140 151`.

## Prueba de vecinos más cercanos

Figura: `artifacts/figures/hinge_nearest_neighbors.png`. Datos: `artifacts/nearest_neighbors.json`.

Para cada personaje se busca, entre las 5,000 imágenes de entrenamiento, la más parecida con dos medidas:
- **CNN:** distancia coseno entre las características de Inception-v3 (ImageNet).
- **Píxeles:** error cuadrático medio (RMS) por píxel en [-1, 1].

**Referencia:** la distancia CNN de cada imagen **real** a su vecina real más cercana tiene mediana 0.070 (percentil 5: 0.043). Los 10 personajes están entre **0.191 y 0.276**, es decir, de 3 a 4 veces más lejos de su vecino que lo que suelen estar dos imágenes reales entre sí.

### Interpretación honesta

1. **Ningún personaje es una copia.** En los 10 casos, el vecino más cercano difiere en al menos un rasgo visible: color de camisa, peinado, accesorios o tono de piel.
2. **La distancia CNN exagera la novedad.** Parte de esa distancia viene del **ruido de color y la textura del fondo** de las imágenes generadas, que Inception detecta y que no existen en los datos. Además, varios vecinos CNN no se parecen a simple vista (por ejemplo, personaje_02 tiene pelo negro y su vecino CNN es rubio). Inception, entrenada con fotos, no es una buena medida de similitud para personajes voxel de 64×64. Una distancia alta no prueba por sí sola que el personaje sea "nuevo".
3. **El vecino por píxeles es más informativo aquí**, y muestra que los personajes son **recombinaciones de piezas del dataset**: el mismo tipo de pelo, de ropa o de ojos, en combinaciones que no aparecen tal cual en el entrenamiento.
4. **El caso más difícil de defender es personaje_08** (gorro verde, camisa a cuadros, piel oscura). Su vecino por píxeles (#2525) tiene **los mismos tres rasgos**; solo cambian la boca y detalles de la ropa. Es más una variante de un personaje existente que un personaje nuevo.

### Qué dice esto sobre lo que aprendió la GAN

La GAN aprendió el "vocabulario" del universo (la silueta de bloque, los tipos de pelo, la ropa y los ojos) y los **recombina**. No aprendió a inventar piezas nuevas, y cuando la combinación elegida es común en los datos, el resultado queda muy cerca de un personaje existente (personaje_08). También mezcla piezas de forma imperfecta: ese es el origen del ruido de color en la ropa.
