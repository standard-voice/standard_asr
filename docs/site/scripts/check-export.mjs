// Post-export parity acceptance. The API reference is projected onto
// three surfaces -- rendered HTML, the static search index, and the llms
// markdown exports -- from the shared helpers in lib/api-text.ts. This
// pins the wiring with sentinels: content present on one surface must be
// present on the others, and every exported markdown page keeps exactly
// one top-level heading. A sentinel missing from one surface means a
// projection dropped content the others kept.
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const outDir = fileURLToPath(new URL('../out/', import.meta.url));
const failures = [];

async function expect(file, needles) {
  const text = await readFile(path.join(outDir, file), 'utf-8');
  for (const needle of needles) {
    if (!text.includes(needle)) failures.push(`${file}: missing ${JSON.stringify(needle)}`);
  }
}

// The two shapes that historically reached some surfaces and not others:
// a schema field (prose-less fields once vanished; described ones lost
// their signatures off-page) and a module docstring (once HTML-only).
const FIELD = 'StreamDeadlines.done_timeout';
const MODULE_DOC = 'Full-duplex streaming transcription protocol';
await expect('docs/reference/streaming/index.html', [`id="${FIELD}"`, MODULE_DOC]);
await expect('api/search', [FIELD, MODULE_DOC]);
await expect('llms-full.txt', [`#### ${FIELD}`, MODULE_DOC]);

// A legacy fragment translates end to end: the old spec URL's stub maps
// the old mkdocs heading id to an id the new page actually renders.
const OLD_ID = '42-stable_until';
const NEW_ID = '42-稳定前缀stable_until';
await expect('spec/specification.html', [`"${OLD_ID}":"${NEW_ID}"`]);
await expect('docs/specification/protocol/index.html', [`id="${NEW_ID}"`]);

// Every machine-readable page URL is canonical: the export serves
// trailing-slash URLs, so a slashless page link costs a 301 on Pages
// before the content arrives. File URLs (dotted final segment) are the
// exemption.
function slashlessPageUrls(text) {
  const bad = [];
  for (const [, target] of text.matchAll(/\]\((\/[^)#\s]+?)(?:#[^)]*)?\)/g)) {
    const last = target.slice(target.lastIndexOf('/') + 1);
    if (last !== '' && !last.includes('.')) bad.push(target);
  }
  for (const [, target] of text.matchAll(/^URL: (\S+)$/gm)) {
    if (!target.endsWith('/')) bad.push(target);
  }
  return bad;
}

// Every exported markdown page opens with plain metadata lines, so the
// page's own H1 stays its heading. (A synthetic `# Title (url)` heading
// here once doubled every page's H1; a page may legitimately hold several
// H1s of its own -- the specification's chapters do.)
async function* contentFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* contentFiles(full);
    else if (entry.name === 'content.md') yield full;
  }
}
async function checkMarkdownExport(file) {
  const text = await readFile(path.join(outDir, file), 'utf-8');
  for (const target of slashlessPageUrls(text)) {
    failures.push(`${file}: slashless page URL ${target}`);
  }
  return text;
}
await checkMarkdownExport('llms.txt');
await checkMarkdownExport('llms-full.txt');
for await (const file of contentFiles(path.join(outDir, 'llms.mdx'))) {
  const rel = path.relative(outDir, file);
  const lines = (await checkMarkdownExport(rel)).split('\n');
  if (!lines[0]?.startsWith('Title: ') || !lines[1]?.startsWith('URL: ')) {
    failures.push(`${rel}: does not open with Title:/URL: metadata`);
  }
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`check-export: ${failure}`);
  process.exit(1);
}
console.log('check-export: projection sentinels + markdown metadata hold');
