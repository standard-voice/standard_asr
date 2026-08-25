import { defineConfig } from 'fumadocs-mdx/config';
import type { Html, Root } from 'mdast';
import { visit } from 'unist-util-visit';

// Rewrites the mkdocs attr_list heading anchor (`# Heading {#id}`) into the
// `[#id]` form the default heading plugin resolves, so the spec's explicit
// anchors survive the renderer change byte-for-byte. Runs before the default
// plugin chain (function form below), which owns slugs and the TOC.
function remarkMkdocsAnchors() {
  return (tree: Root) => {
    visit(tree, 'heading', (node) => {
      const last = node.children.at(-1);
      if (last?.type !== 'text') return;
      const match = /^(.*)\{#([^}]+)\}\s*$/s.exec(last.value);
      if (match) last.value = `${match[1]}[#${match[2]}]`;
    });
  };
}

// The md pipeline drops raw HTML silently (no allowDangerousHtml). A silent
// drop is exactly the failure mode this project's docs forbid, so: convert
// the one benign inline case (`<br>`) into a hard break, and fail the build
// on anything else.
function remarkHtmlGuard() {
  return (tree: Root, file: { path?: string }) => {
    visit(tree, 'html', (node: Html, index, parent) => {
      if (parent === undefined || index === undefined) return;
      // An HTML comment renders to nothing by design (and is Vale's
      // in-document control syntax), so dropping it is the desired
      // outcome, not a silent loss.
      if (/^(?:<!--[\s\S]*?-->\s*)+$/.test(node.value.trim())) {
        parent.children.splice(index, 1);
        return index;
      }
      if (/^<br\s*\/?>$/.test(node.value.trim())) {
        parent.children.splice(index, 1, { type: 'break' });
        return;
      }
      throw new Error(
        `raw HTML in ${file.path ?? 'unknown file'} would be dropped by the ` +
          `Markdown pipeline: ${JSON.stringify(node.value.slice(0, 80))}. ` +
          `Wrap it in a code span, or extend remarkHtmlGuard.`,
      );
    });
  };
}

export default defineConfig({
  mdxOptions: {
    remarkPlugins: (v) => [remarkMkdocsAnchors, ...v, remarkHtmlGuard],
  },
});
