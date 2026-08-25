import { getApiModule } from '@/lib/api';
import { apiModuleStructuredData } from '@/lib/api-text';
import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

export const dynamic = 'force-static';
export const revalidate = false;

// Default multilingual tokenization: the corpus mixes English guides with
// the Chinese normative specification. Reference pages fold the generated
// API members into their structured data, so the reference is searchable
// even though it renders outside the markdown pipeline; result ids match
// the rendered anchors, so hits deep-link to the member.
export const { staticGET: GET } = createFromSource(source, {
  buildIndex(page) {
    const base = {
      title: page.data.title,
      description: page.data.description,
      url: page.url,
      id: page.url,
      structuredData: page.data.structuredData,
    };
    if (!page.data.api_module) return base;
    const api = apiModuleStructuredData(getApiModule(page.data.api_module));
    return {
      ...base,
      structuredData: {
        headings: [...base.structuredData.headings, ...api.headings],
        contents: [...base.structuredData.contents, ...api.contents],
      },
    };
  },
});
