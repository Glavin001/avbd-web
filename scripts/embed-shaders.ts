/**
 * Embed WGSL shader files as TypeScript string constants.
 * Run: npx tsx scripts/embed-shaders.ts
 *
 * Reads all .wgsl files from src/shaders/ (including subdirectories)
 * and generates src/shaders/embedded.ts with exported constants.
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, basename, relative } from 'path';

const SHADERS_DIR = join(import.meta.dirname ?? '.', '..', 'src', 'shaders');
const OUTPUT_FILE = join(SHADERS_DIR, 'embedded.ts');

/**
 * Recursively find all .wgsl files in a directory.
 */
function findWgslFiles(dir: string): string[] {
  const results: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      results.push(...findWgslFiles(fullPath));
    } else if (entry.endsWith('.wgsl')) {
      results.push(fullPath);
    }
  }
  return results.sort();
}

const wgslFiles = findWgslFiles(SHADERS_DIR);

let output = '// AUTO-GENERATED — do not edit. Run: npx tsx scripts/embed-shaders.ts\n\n';

for (const filePath of wgslFiles) {
  const content = readFileSync(filePath, 'utf-8');
  // Convert relative path to constant name:
  //   primal-update-2d.wgsl → PRIMAL_UPDATE_2D_WGSL
  //   collision/morton-codes-2d.wgsl → MORTON_CODES_2D_WGSL
  // We use just the filename (not path) to keep names short
  const name = basename(filePath, '.wgsl')
    .toUpperCase()
    .replace(/-/g, '_');
  output += `export const ${name}_WGSL = \`${content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;\n\n`;
}

writeFileSync(OUTPUT_FILE, output);
const relFiles = wgslFiles.map(f => relative(SHADERS_DIR, f));
console.log(`Generated ${OUTPUT_FILE} with ${wgslFiles.length} shader(s):`);
for (const f of relFiles) {
  console.log(`  - ${f}`);
}
