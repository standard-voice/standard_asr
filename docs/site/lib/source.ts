import { loader } from 'fumadocs-core/source';
import { defineDocs } from 'fumadocs-mdx/macro';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import { z } from 'zod';

// One rule, no list: every Markdown file under docs/content is published,
// and nothing outside it is. Anything internal (design notes, research,
// working notes, legacy pages) lives in docs/internal and is invisible here
// by construction, so adding a page is only ever adding a file.
const docs = defineDocs({
  dir: '../content',
  docs: {
    schema: pageSchema.extend({
      // Names a Python module whose generated API reference renders after
      // the page's own prose (see components/api).
      api_module: z.string().optional(),
      // BCP-47 tag for a page not in the site's default English, stamped
      // on the article container so assistive tech and translators see
      // the real language (WCAG 3.1.2). The site chrome stays lang="en".
      lang: z.string().optional(),
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

export type Page = (typeof source)['$inferPage'];

// GitHub Pages serves the site under a project base path. next/link adds
// it automatically, but a URL that reaches the client as a plain string
// (a fetch target, exported text) must carry it itself -- fumadocs-ui's
// own withBasePath reads import.meta.env.BASE_URL, which only Vite
// defines, so under Next it is a no-op.
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function withBasePath(pathname: string): string {
  return `${BASE_PATH}${pathname}`;
}

// The export is trailingSlash: on GitHub Pages the slashless form costs
// a 301 before the content arrives, so machine-readable page URLs are
// emitted canonical, matching the API xref resolver. Only for page URLs;
// file URLs (content.md, llms.txt) take no trailing slash.
export function canonicalPageUrl(pageUrl: string): string {
  return withBasePath(pageUrl.endsWith('/') ? pageUrl : `${pageUrl}/`);
}

export function getPageMarkdownUrl(page: Page) {
  const segments = [...page.slugs, 'content.md'];

  return {
    segments,
    url: withBasePath('/' + ['llms.mdx/docs', ...segments].join('/')),
  };
}

// The processed markdown keeps the source's relative .md links; resolve
// each against the page's own file so machine consumers get real site
// URLs. A target that does not resolve to a page is left untouched.
function absolutizeMarkdownLinks(markdown: string, page: Page): string {
  const dir = page.path.split('/').slice(0, -1);
  return markdown.replace(
    /\]\((\.{1,2}\/[^)#\s]+\.md)(#[^)]*)?\)/g,
    (match, target: string, fragment: string | undefined) => {
      const parts = [...dir];
      for (const segment of target.replace(/\.md$/, '').split('/')) {
        if (segment === '.') continue;
        else if (segment === '..') parts.pop();
        else parts.push(segment);
      }
      if (parts.at(-1) === 'index') parts.pop();
      const resolved = source.getPage(parts);
      if (!resolved) return match;
      return `](${canonicalPageUrl(resolved.url)}${fragment ?? ''})`;
    },
  );
}

export async function getLLMText(page: Page) {
  const processed = await page.data.getText('processed');

  // Metadata as plain lines, not a heading: every content page opens with
  // its own H1 (which can differ from the nav title), and a synthetic
  // second H1 above it would make each exported page double-headed.
  return `Title: ${page.data.title}
URL: ${canonicalPageUrl(page.url)}

${absolutizeMarkdownLinks(processed, page)}`;
}
