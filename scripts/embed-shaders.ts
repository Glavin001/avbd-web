/**
 * Embed WGSL shader files as TypeScript string constants.
 * Run: npx tsx scripts/embed-shaders.ts
 *
 * Reads all .wgsl files from src/shaders/ and generates
 * src/shaders/embedded.ts with exported constants.
 */
import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { join, basename } from 'path';

const SHADERS_DIR = join(import.meta.dirname ?? '.', '..', 'src', 'shaders');
const OUTPUT_FILE = join(SHADERS_DIR, 'embedded.ts');

const wgslFiles = readdirSync(SHADERS_DIR)
  .filter(f => f.endsWith('.wgsl'))
  .sort();

let output = '// AUTO-GENERATED — do not edit. Run: npx tsx scripts/embed-shaders.ts\n\n';

for (const file of wgslFiles) {
  const content = readFileSync(join(SHADERS_DIR, file), 'utf-8');
  // Convert filename to constant name: primal-update-2d.wgsl → PRIMAL_UPDATE_2D
  const name = basename(file, '.wgsl')
    .toUpperCase()
    .replace(/-/g, '_');
  output += `export const ${name}_WGSL = \`${content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;\n\n`;
}

writeFileSync(OUTPUT_FILE, output);
console.log(`Generated ${OUTPUT_FILE} with ${wgslFiles.length} shader(s):`);
for (const f of wgslFiles) {
  console.log(`  - ${f}`);
}
