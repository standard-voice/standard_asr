import { getPageMarkdownUrl, source } from '@/lib/source';
import {
  DocsBody,
  DocsPage,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from 'fumadocs-ui/layouts/docs/page';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/components/mdx';
import type { Metadata } from 'next';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { ApiModuleDoc } from '@/components/api';
import { apiToc, getApiModule } from '@/lib/api';
import { gitConfig, siteUrl } from '@/lib/shared';

export default async function Page(props: PageProps<'/docs/[[...slug]]'>) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const apiModule = page.data.api_module ? getApiModule(page.data.api_module) : undefined;
  const toc = apiModule
    ? [...page.data.toc, ...apiToc(apiModule).map((entry) => ({ ...entry, title: entry.title }))]
    : page.data.toc;

  return (
    <DocsPage toc={toc} full={page.data.full}>
      <div className="flex flex-row gap-2 items-center border-b pb-4">
        <MarkdownCopyButton markdownUrl={markdownUrl} />
        <ViewOptionsPopover
          markdownUrl={markdownUrl}
          githubUrl={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/docs/content/${page.path}`}
        />
      </div>
      <DocsBody lang={page.data.lang}>
        <MDX
          components={getMDXComponents({
            // Resolves `./sibling.md` and `../other/page.md` to page URLs.
            a: createRelativeLink(source, page),
          })}
        />
        {apiModule ? <ApiModuleDoc module={apiModule} /> : null}
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: PageProps<'/docs/[[...slug]]'>): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
    // Absolute, not metadataBase-relative: relative resolution drops the
    // GitHub Pages base path. The trailing slash matches the export shape.
    alternates: { canonical: `${siteUrl}${page.url}/` },
    openGraph: {
      url: `${siteUrl}${page.url}/`,
      images: '/' + ['og/docs', ...page.slugs, 'image.png'].join('/'),
    },
  };
}
