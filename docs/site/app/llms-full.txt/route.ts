import { pageLLMText } from '@/lib/api-text';
import { source } from '@/lib/source';

export const dynamic = 'force-static';
export const revalidate = false;

export async function GET() {
  const scan = source.getPages().map(pageLLMText);
  const scanned = await Promise.all(scan);

  return new Response(scanned.join('\n\n'));
}
