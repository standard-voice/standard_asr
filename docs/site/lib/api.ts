import apiData from '@/.generated/api.json';
import { source } from './source';

// Types for the neutral API data emitted by scripts/dump_api.py. The dump
// is renderer-agnostic; everything presentation-related lives on this side.

export interface AnnotationToken {
  text: string;
  target?: string;
}

export type Annotation = AnnotationToken[] | null;

export interface NamedItem {
  name?: string | null;
  annotation: Annotation;
  description: string;
  default?: string;
  constraints?: string[];
}

export type DocSection =
  | { kind: 'text'; value: string }
  | { kind: 'parameters' | 'attributes'; items: NamedItem[] }
  | { kind: 'returns' | 'yields' | 'receives'; items: NamedItem[] }
  | { kind: 'raises' | 'warns'; items: NamedItem[] }
  | { kind: 'examples'; value: string }
  | { kind: 'admonition'; type: string; value: string }
  | { kind: 'deprecated'; value: string };

export interface SourceLocation {
  file: string;
  line: number;
}

export interface ApiParameter {
  name: string;
  kind: string;
  annotation: Annotation;
  default?: string;
}

export interface ApiFunction {
  kind: 'function';
  name: string;
  async: boolean;
  /** griffe labels: "classmethod", "staticmethod", "async", ... */
  labels: string[];
  parameters: ApiParameter[];
  returns: Annotation;
  docstring: DocSection[];
  source: SourceLocation | null;
}

export interface ApiAttribute {
  kind: 'attribute';
  name: string;
  /** Class attributes only: whether the field renders (set by dump_api). */
  published?: boolean;
  annotation: Annotation;
  /** The shown default: a pydantic Field() call arrives pre-unpacked. */
  default: string | null;
  /** Leftover Field() keywords (gt, ge, ...), verbatim `name=value`. */
  constraints: string[];
  /** Field(description=...), the fallback when no docstring prose exists. */
  field_description: string | null;
  labels: string[];
  docstring: DocSection[];
  source: SourceLocation | null;
}

export interface ApiClass {
  kind: 'class';
  name: string;
  bases: Annotation[];
  labels: string[];
  parameters: ApiParameter[];
  docstring: DocSection[];
  attributes: ApiAttribute[];
  methods: ApiFunction[];
  source: SourceLocation | null;
}

export type ApiMember = ApiClass | ApiFunction | ApiAttribute;

export interface ApiModuleData {
  path: string;
  docstring: DocSection[];
  members: ApiMember[];
}

interface ApiData {
  package: string;
  version: string;
  modules: ApiModuleData[];
  symbols: Record<string, { page: string; anchor: string | null }>;
}

const data = apiData as unknown as ApiData;

// Module -> page URL, derived from the one declaration that already has
// to exist: each reference page's api_module frontmatter. Drift in either
// direction fails the build instead of silently unlinking.
const PAGE_URLS: Record<string, string> = {};
for (const page of source.getPages()) {
  const moduleName = page.data.api_module;
  if (!moduleName) continue;
  if (PAGE_URLS[moduleName] !== undefined) {
    throw new Error(`api_module ${moduleName} is declared by more than one page`);
  }
  PAGE_URLS[moduleName] = page.url;
}
for (const module of data.modules) {
  if (PAGE_URLS[module.path] === undefined) {
    throw new Error(
      `dump_api documents ${module.path} but no page under docs/content declares ` +
        `api_module: ${module.path}`,
    );
  }
}

export function getApiModule(path: string): ApiModuleData {
  const module = data.modules.find((candidate) => candidate.path === path);
  if (!module) {
    throw new Error(
      `api_module names ${path}, which dump_api does not document; ` +
        'add it to MODULE_PAGES (docs/site/scripts/dump_api.py) or fix the frontmatter',
    );
  }
  return module;
}

export function packageVersion(): string {
  return data.version;
}

/** Resolve a dotted symbol path to its documentation URL, when documented. */
export function resolveSymbol(dotted: string): string | undefined {
  const entry = data.symbols[dotted];
  if (!entry) return undefined;
  const page = PAGE_URLS[entry.page];
  if (!page) return undefined;
  // Trailing slash first: the site exports with `trailingSlash`, and the
  // slashless form costs a redirect on GitHub Pages before the fragment
  // can resolve.
  return entry.anchor === null ? `${page}/` : `${page}/#${entry.anchor}`;
}

export interface ApiTocEntry {
  title: string;
  url: string;
  depth: number;
}

/** Table-of-contents entries for a module's members, mirroring mkdocstrings. */
export function apiToc(module: ApiModuleData): ApiTocEntry[] {
  const entries: ApiTocEntry[] = [];
  for (const member of module.members) {
    entries.push({ title: member.name, url: `#${member.name}`, depth: 2 });
    if (member.kind === 'class') {
      for (const method of member.methods) {
        entries.push({
          title: method.name,
          url: `#${member.name}.${method.name}`,
          depth: 3,
        });
      }
    }
  }
  return entries;
}
