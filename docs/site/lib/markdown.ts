import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';
import type { Code, Html, Link, Root } from 'mdast';
import { codeToHtml } from 'shiki';
import { resolveSymbol } from './api';

// Renders docstring Markdown (from the API dump) to HTML at build time.
// The input is this repository's own docstrings — trusted content.

function remarkXrefs() {
  return (tree: Root) => {
    visit(tree, 'link', (node: Link, index, parent) => {
      if (!node.url.startsWith('xref:')) return;
      const target = node.url.slice('xref:'.length);
      const href = resolveSymbol(target);
      if (href) {
        // This HTML bypasses the framework's Link, so the base path must
        // be prefixed here.
        node.url = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}${href}`;
        return;
      }
      // Undocumented target: unwrap to the code-span label. Renaming the
      // node instead leaves the link handler's href on the element.
      if (parent && index !== undefined) {
        parent.children.splice(index, 1, ...node.children);
        return index;
      }
    });
  };
}

// Docstrings are trusted repository content, but raw HTML in Markdown is
// still an accident magnet: an unquoted `<mode>` placeholder lexes as a
// tag and the browser swallows it, silently eating text. Published .md
// pages already fail their build on raw HTML; docstrings get the same
// rule. (remarkShiki injects its html nodes after this pass, so
// highlighted code stays allowed.)
function remarkForbidRawHtml() {
  return (tree: Root) => {
    visit(tree, 'html', (node: Html) => {
      throw new Error(
        `raw HTML in a docstring: ${JSON.stringify(node.value.slice(0, 80))}. ` +
          'The browser would swallow it silently; wrap placeholders in a code span.',
      );
    });
  };
}

function remarkShiki() {
  return async (tree: Root) => {
    const blocks: Code[] = [];
    visit(tree, 'code', (node: Code) => {
      blocks.push(node);
    });
    for (const node of blocks) {
      const html = await codeToHtml(node.value, {
        lang: node.lang ?? 'python',
        themes: { light: 'github-light', dark: 'github-dark' },
        defaultColor: false,
      });
      const replacement = node as unknown as { type: string; value: string };
      replacement.type = 'html';
      replacement.value = html;
    }
  };
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkForbidRawHtml)
  .use(remarkXrefs)
  .use(remarkShiki)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeStringify, { allowDangerousHtml: true });

export async function markdownToHtml(markdown: string): Promise<string> {
  const file = await processor.process(markdown);
  return String(file);
}
