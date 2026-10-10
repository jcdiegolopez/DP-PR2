// Generates the training set: DiceBear "Voxel Art" characters (CC0 1.0) rendered to PNG.
// Usage: node generate.mjs [--n 5000] [--size 64] [--bg e0e0e0] [--out ../data/voxel64]
import { Style, Avatar } from '@dicebear/core';
import { Resvg } from '@resvg/resvg-js';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith('--')) pairs.push([arg.slice(2), all[i + 1]]);
    return pairs;
  }, []),
);
const N = Number(args.n ?? 5000);
const SIZE = Number(args.size ?? 64);
// One flat background so the GAN does not spend capacity on background colors.
const BG = String(args.bg ?? 'e0e0e0');
const OUT = path.resolve(args.out ?? '../data/voxel64');

const require = createRequire(import.meta.url);
const definition = require('@dicebear/styles/voxel-art.json');
const style = new Style(definition);

fs.mkdirSync(OUT, { recursive: true });

const seen = new Set();
const rows = ['file,seed,sha1'];
let attempt = 0;
let duplicates = 0;

while (rows.length - 1 < N) {
  // Deterministic seeds make the dataset reproducible.
  const seed = `voxel-${attempt++}`;
  const svg = new Avatar(style, { seed, backgroundColor: BG }).toString();
  const hash = createHash('sha1').update(svg).digest('hex');
  if (seen.has(hash)) {
    duplicates++;
    continue;
  }
  seen.add(hash);

  const png = new Resvg(svg, { fitTo: { mode: 'width', value: SIZE } }).render().asPng();
  const file = `${String(rows.length - 1).padStart(5, '0')}.png`;
  fs.writeFileSync(path.join(OUT, file), png);
  rows.push(`${file},${seed},${hash}`);
}

fs.writeFileSync(path.join(OUT, 'metadata.csv'), rows.join('\n') + '\n');
console.log(`Wrote ${N} unique images (${SIZE}x${SIZE}) to ${OUT}; skipped ${duplicates} duplicate seeds.`);
