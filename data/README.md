# Datos: Voxel Art (DiceBear)

| Campo | Valor |
|---|---|
| Origen | Estilo de avatar **"Voxel Art"** de [DiceBear](https://www.dicebear.com/styles/voxel-art/), paquete `@dicebear/styles` 10.6.0 |
| Autor | DiceBear (https://www.dicebear.com) |
| Licencia | **CC0 1.0** (dominio público): https://creativecommons.org/publicdomain/zero/1.0/ |
| Atribución (no obligatoria) | „Voxel Art” (https://www.dicebear.com) by „DiceBear”, licensed under „CC0 1.0” |
| Tamaño | 5,000 imágenes PNG únicas (~22 MB) |
| Resolución | 64 × 64 px, renderizadas directamente desde SVG con `@resvg/resvg-js` |
| Fondo | Color único `#e0e0e0` (el estilo original usa 5 colores de fondo) |

## Cómo se generó

```bash
cd dataset
npm install
node generate.mjs --n 5000 --size 64 --bg e0e0e0 --out ../data/voxel64
```

- Las semillas son deterministas (`voxel-0`, `voxel-1`, ...), así que el conjunto se puede regenerar idéntico.
- Se descartan duplicados exactos comparando el hash SHA-1 del SVG. En la generación de 5,000 no hubo duplicados.
- `voxel64/metadata.csv` guarda `file, seed, sha1` de cada imagen.

## Variabilidad del estilo

Componentes por personaje: 21 sombreros o peinados, 10 outfits, 8 ojos, 10 bocas, 4 narices, 6 animaciones (poses), y barba, mejillas, cejas y lentes opcionales. Colores: 8 tonos de piel, 12 de pelo, 8 de sombrero, 10 de camisa, 6 de pantalón, 6 de zapatos y 6 de chaqueta.

## Preparación para el entrenamiento

Se cargan como RGB (se descarta el canal alfa; el fondo ya es opaco) y se normalizan a [-1, 1].
