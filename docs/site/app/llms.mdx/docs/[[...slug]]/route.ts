import { pageLLMText } from '@/lib/api-text';
import { getPageMarkdownUrl, source } from '@/lib/source';
import { notFound } from 'next/navigation';

export const dynamic = 'force-static';
export const revalidate = false;

export async function GET(_req: Request, { params }: RouteContext<'/llms.mdx/docs/[[...slug]]'>) {
  const { slug } = await params;
  // The trailing "content.md" segment names the format, not a page.
  const page = source.getPage(slug?.slice(0, -1));
  if (!page) notFound();

  return new Response(await pageLLMText(page), {
    headers: {
      'Content-Type': 'text/markdown',
    },
  });
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({
    slug: getPageMarkdownUrl(page).segments,
  }));
}
