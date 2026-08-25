import { source, withBasePath } from '@/lib/source';
import { llms } from 'fumadocs-core/source';

export const dynamic = 'force-static';
export const revalidate = false;

export function GET() {
  // The helper emits site-root links; prefix the Pages base path and
  // canonicalize page URLs to the trailing-slash form the export serves.
  const index = llms(source)
    .index()
    .replaceAll('](/', `](${withBasePath('/')}`)
    .replace(/\]\((\/[^)#\s]+?)\)/g, (match, path: string) => {
      const last = path.slice(path.lastIndexOf('/') + 1);
      return last === '' || last.includes('.') ? match : `](${path}/)`;
    });
  return new Response(index);
}
