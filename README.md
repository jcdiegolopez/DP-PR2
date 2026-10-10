# Proyecto 2: Diseño generativo de personajes — Cubópolis

Universidad del Valle de Guatemala · Deep Learning 2026
Diego José López Campos · 23242

DCGAN entrenada desde cero para generar habitantes nuevos de **Cubópolis**, una ciudad hecha de bloques (`docs/universo.md`). Los datos son 5,000 personajes del estilo "Voxel Art" de DiceBear (CC0). El modelo final es una DCGAN con pérdida hinge.

## Estructura

```
dataset/              generador del dataset (Node + DiceBear)
data/README.md        origen, licencia y preparación de los datos
src/
  data.py             carga de imágenes a tensor en [-1, 1]
  models.py           generador y discriminador DCGAN (opciones: spectral norm, sin BN en D)
  losses.py           pérdidas BCE no saturante, minimax e hinge
  train.py            entrenamiento + rejilla de ruido fijo + curvas + checkpoints
  plot_run.py         figuras de curvas de pérdida y evolución
  evaluate.py         FID y razón de diversidad por corrida
  features.py         características de Inception-v3 (métricas y vecinos)
  gallery.py          candidatos, galería final y prueba de vecinos más cercanos
artifacts/
  fixed_z.pt          los 16 z fijos, comunes a todas las corridas
  runs/<corrida>/     config.json, losses.csv, snapshots/ (rejilla por época)
  runs/hinge/G_final.pt  pesos del modelo final
  figures/            figuras de la presentación
  metrics.csv         FID y diversidad de cada corrida
  nearest_neighbors.json
galeria/              los 10 personajes (64×64 y ampliados ×4), seeds.json y z_vectors.pt
docs/                 universo, bitácora de experimentos, galería y vecinos, personajes
```

## Reproducir

Requisitos: Python 3.14, Node 24 y una GPU NVIDIA (probado con RTX 5070 Laptop, CUDA 12.8).

```bash
# 1. Entorno
python -m venv .venv && .venv/Scripts/activate        # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# 2. Datos (5,000 imágenes 64×64, deterministas)
cd dataset && npm install && node generate.mjs && cd ..

# 3. Regenerar la galería con los pesos incluidos (no requiere entrenar)
python src/gallery.py final --run hinge --pick 36 100 30 13 3 62 103 102 140 151

# 4. (Opcional) Reentrenar las cuatro corridas, ~6 min cada una
python src/train.py --name base    --loss bce
python src/train.py --name hinge   --loss hinge
python src/train.py --name sn      --loss bce --spectral
python src/train.py --name sn_nobn --loss bce --spectral --no-d-bn
python src/plot_run.py base hinge sn sn_nobn
python src/evaluate.py base hinge sn sn_nobn
python src/gallery.py candidates --run hinge
```

El paso 3 regenera exactamente las imágenes de `galeria/` a partir de `artifacts/runs/hinge/G_final.pt` y las semillas de `galeria/seeds.json`. Reentrenar en GPU puede dar resultados ligeramente distintos, porque algunas operaciones de cuDNN no son deterministas.

## Semillas

| Uso | Semilla |
|---|---|
| Dataset | `voxel-0` … `voxel-4999` (DiceBear) |
| Entrenamiento (todas las corridas) | 23242 |
| Rejilla de ruido fijo (16 z) | 2026 → `artifacts/fixed_z.pt` |
| Métricas (2,000 z) | 777 |
| Candidatos de la galería | 100000 + índice (0–199) |

## Resultados

| Corrida | Cambio respecto a `base` | FID ↓ | Diversidad |
|---|---|---|---|
| `base` | — (BCE no saturante) | 151.8 | 0.966 |
| `hinge` | pérdida hinge | 161.9 | 0.992 |
| `sn` | normalización espectral en D | 401.8 | 0.093 (colapso) |
| `sn_nobn` | normalización espectral, sin BatchNorm en D | 224.9 | 0.814 |

Hipótesis, resultados y análisis: `docs/experimentos.md`. Galería, tasa de selección (10 de 200) y prueba de vecinos: `docs/galeria_y_vecinos.md`.

## Versiones

torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · numpy 2.5.2 · pillow 12.3.0 · matplotlib 3.11.2 · pandas 3.0.6 · scipy 1.18.1 · @dicebear/core 10.7.0 · @dicebear/styles 10.6.0 · @resvg/resvg-js 2.6.2
