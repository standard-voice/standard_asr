// Build-time guards for the content tree contract.
//
// Every meta.json entry must resolve to a real page: Fumadocs silently
// skips an entry that matches nothing, so a rename leaves the page built
// but absent from the sidebar with a green build; this restores the loud
// failure the mkdocs `--strict` nav check used to provide. (New files
// need no entry: the "..." rest token appends them.)
//
// And the tree holds plain Markdown only: the Fumadocs collection would
// also compile a stray .mdx (JSX, imports, arbitrary JavaScript) into a
// published page, which the Vale corpus and the raw-HTML guard never see.
// The one-rule boundary is "every file under docs/content is a published
// .md page", so anything else fails here.
import { access, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const contentDir = fileURLToPath(new URL('../../content/', import.meta.url));
const require = createRequire(import.meta.url);
const meta = require(path.join(contentDir, 'meta.json'));

const foreign = [];
let fileCount = 0;
async function walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    // Editor and OS droppings (.DS_Store and friends) are not content.
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await walk(full);
    } else {
      fileCount += 1;
      if (!entry.name.endsWith('.md') && entry.name !== 'meta.json') {
        foreign.push(path.relative(contentDir, full));
      }
    }
  }
}
await walk(contentDir);
if (foreign.length > 0) {
  console.error(
    `docs/content holds files that are not published .md pages: ${foreign.join(', ')}. ` +
      'Published prose is plain Markdown only (no MDX); move anything else out of docs/content.',
  );
  process.exit(1);
}

const missing = [];
for (const entry of meta.pages) {
  if (entry === '...' || /^---.*---$/.test(entry) || /^\[.*\]\(.*\)$/.test(entry)) continue;
  try {
    await access(path.join(contentDir, `${entry}.md`));
  } catch {
    missing.push(entry);
  }
}
if (missing.length > 0) {
  console.error(
    `docs/content/meta.json names pages that do not exist: ${missing.join(', ')}. ` +
      'Fix the entry or restore the file; a stale entry silently drops the page from the sidebar.',
  );
  process.exit(1);
}
console.log(`check-content: ${meta.pages.length} meta entries resolve, ${fileCount} files all .md`);
