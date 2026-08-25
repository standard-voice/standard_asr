// Text projections of the neutral API data: Python signatures, plain-text
// docstrings, and the derived search/LLM surfaces. The HTML renderer
// (components/api.tsx), the llms exports, and the search index all read
// the same helpers here -- one member list, one merged field-row builder,
// one module docstring -- so a member present on one surface is present
// on the others (scripts/check-export.mjs pins the wiring after export).
import {
  type Annotation,
  type ApiClass,
  type ApiFunction,
  type ApiMember,
  type ApiModuleData,
  type ApiParameter,
  type DocSection,
  type NamedItem,
  getApiModule,
} from '@/lib/api';
import { getLLMText, type Page } from '@/lib/source';
import type { StructuredData } from 'fumadocs-core/mdx-plugins';

export function annotationText(annotation: Annotation): string {
  return annotation?.map((token) => token.text).join('') ?? '';
}

export function paramPrefix(param: ApiParameter): string {
  if (param.kind === 'variadic positional') return '*';
  if (param.kind === 'variadic keyword') return '**';
  return '';
}

export function parameterParts(parameters: ApiParameter[]): string[] {
  const parts: string[] = [];
  let sawKeywordOnly = false;
  let inPositionalOnly = false;
  for (const param of parameters) {
    if (inPositionalOnly && param.kind !== 'positional-only') {
      parts.push('/');
      inPositionalOnly = false;
    }
    if (param.kind === 'positional-only') inPositionalOnly = true;
    if (param.kind === 'keyword-only' && !sawKeywordOnly) {
      sawKeywordOnly = true;
      parts.push('*');
    }
    if (param.kind === 'variadic positional') sawKeywordOnly = true;
    let text = `${paramPrefix(param)}${param.name}`;
    if (param.annotation) text += `: ${annotationText(param.annotation)}`;
    if (param.default !== undefined) text += `${param.annotation ? ' = ' : '='}${param.default}`;
    parts.push(text);
  }
  if (inPositionalOnly) parts.push('/');
  return parts;
}

export function signatureOf(fn: ApiFunction, owner?: string): string {
  const parts = parameterParts(fn.parameters);
  const name = owner ? `${owner}.${fn.name}` : fn.name;
  const returns = fn.returns ? ` -> ${annotationText(fn.returns)}` : '';
  const prefix = fn.async ? 'async def' : 'def';
  // The binding decorators are part of the calling contract; `cls` and
  // `self` are already stripped, so the shown parameters are what the
  // caller passes either way.
  const decorators = fn.labels
    .filter((label) => label === 'classmethod' || label === 'staticmethod')
    .map((label) => `@${label}\n`)
    .join('');
  const oneLine = `${prefix} ${name}(${parts.join(', ')})${returns}`;
  if (oneLine.length <= 88) return `${decorators}${oneLine}`;
  const indented = parts.map((p) => `    ${p},`).join('\n');
  return `${decorators}${prefix} ${name}(\n${indented}\n)${returns}`;
}

export function classSignatureOf(cls: ApiClass): string {
  const bases = cls.bases.map((base) => annotationText(base)).filter(Boolean);
  const head = bases.length > 0 ? `class ${cls.name}(${bases.join(', ')})` : `class ${cls.name}`;
  if (cls.parameters.length === 0) return head;
  const params = parameterParts(cls.parameters)
    .map((part) => `    ${part},`)
    .join('\n');
  return `${head}\n\n${cls.name}(\n${params}\n)`;
}

// The xref links dump_api emits are for the HTML renderer; text surfaces
// keep the code-span label.
function stripXrefs(markdown: string): string {
  return markdown.replace(/\[(`[^`]+`)\]\(xref:[^)]+\)/g, '$1');
}

function itemLine(item: {
  name?: string | null;
  annotation: Annotation;
  description: string;
  default?: string;
}): string {
  const name = item.name ? `${item.name}` : '';
  const annotation = annotationText(item.annotation);
  const head = [name, annotation && `(${annotation})`].filter(Boolean).join(' ');
  const suffix = item.default !== undefined ? ` Default: \`${item.default}\`.` : '';
  return `- ${head ? `${head}: ` : ''}${stripXrefs(item.description)}${suffix}`;
}

const SECTION_LABELS: Record<string, string> = {
  parameters: 'Parameters',
  attributes: 'Attributes',
  returns: 'Returns',
  yields: 'Yields',
  receives: 'Receives',
  raises: 'Raises',
  warns: 'Warns',
};

export function sectionsToMarkdown(sections: DocSection[]): string {
  const blocks: string[] = [];
  for (const section of sections) {
    switch (section.kind) {
      case 'text':
        blocks.push(stripXrefs(section.value));
        break;
      case 'parameters':
      case 'attributes':
      case 'returns':
      case 'yields':
      case 'receives':
      case 'raises':
      case 'warns':
        blocks.push(
          `${SECTION_LABELS[section.kind]}:\n${section.items.map((item) => itemLine(item)).join('\n')}`,
        );
        break;
      case 'examples':
        blocks.push(`Examples:\n\n${section.value}`);
        break;
      case 'admonition':
        blocks.push(`${section.type}: ${stripXrefs(section.value)}`);
        break;
      case 'deprecated':
        blocks.push(`Deprecated: ${stripXrefs(section.value)}`);
        break;
    }
  }
  return blocks.join('\n\n');
}

function memberSignature(member: ApiMember): string {
  if (member.kind === 'class') return classSignatureOf(member);
  if (member.kind === 'attribute') {
    const annotation = member.annotation ? `: ${annotationText(member.annotation)}` : '';
    const value = member.default !== null ? ` = ${member.default}` : '';
    const constraints = member.constraints.length > 0 ? `  # ${member.constraints.join(', ')}` : '';
    return `${member.name}${annotation}${value}${constraints}`;
  }
  return signatureOf(member);
}

// Merge the typed attribute list with the class docstring's Attributes
// section: the section carries the authored descriptions; the typed list
// carries annotations, defaults, and constraints. Which fields appear is
// dump_api's `published` decision (every annotated field, plus anything
// described), so a schema field cannot vanish for lack of prose. Every
// surface -- the HTML table, the llms markdown, the search index -- reads
// these rows.
export function mergedAttributeRows(cls: ApiClass): NamedItem[] {
  const typed = new Map(cls.attributes.map((attr) => [attr.name, attr]));
  const documented = new Map<string, NamedItem>();
  for (const section of cls.docstring) {
    if (section.kind !== 'attributes') continue;
    for (const item of section.items) {
      if (item.name) documented.set(item.name, item);
    }
  }
  const rows: NamedItem[] = [];
  for (const attr of cls.attributes) {
    if (!attr.published) continue;
    const doc = documented.get(attr.name);
    const own = attr.docstring.find((s) => s.kind === 'text');
    let description = doc?.description ?? '';
    if (!description && own?.kind === 'text') description = own.value;
    if (!description && attr.field_description) description = attr.field_description;
    rows.push({
      name: attr.name,
      annotation: attr.annotation ?? doc?.annotation ?? null,
      description,
      default: attr.default ?? doc?.default,
      constraints: attr.constraints.length > 0 ? attr.constraints : undefined,
    });
  }
  // Documented names with no typed counterpart still render: the
  // capability classes describe fields inherited from their base here.
  for (const [name, item] of documented) {
    if (!typed.has(name)) rows.push(item);
  }
  return rows;
}

function fieldSignature(row: NamedItem): string {
  let sig = row.name ?? '';
  if (row.annotation) sig += `: ${annotationText(row.annotation)}`;
  if (row.default !== undefined) sig += ` = ${row.default}`;
  if (row.constraints?.length) sig += `  # ${row.constraints.join(', ')}`;
  return sig;
}

// Markdown rendering of a module's API reference for the llms exports.
// Mirrors the HTML renderer's structure: module docstring, then member
// heading, signature, docstring, then fields and methods. The class
// docstring's Attributes section is omitted like the HTML page omits it:
// its descriptions reach the per-field entries via mergedAttributeRows.
export function apiModuleToMarkdown(module: ApiModuleData): string {
  const blocks: string[] = [`## API reference: \`${module.path}\``];
  const intro = sectionsToMarkdown(module.docstring);
  if (intro) blocks.push(intro);
  for (const member of module.members) {
    blocks.push(`### ${member.name}`);
    blocks.push(`\`\`\`python\n${memberSignature(member)}\n\`\`\``);
    const sections =
      member.kind === 'class'
        ? member.docstring.filter((section) => section.kind !== 'attributes')
        : member.docstring;
    const body = sectionsToMarkdown(sections);
    if (body) blocks.push(body);
    if (member.kind === 'class') {
      for (const row of mergedAttributeRows(member)) {
        blocks.push(`#### ${member.name}.${row.name}`);
        blocks.push(`\`\`\`python\n${fieldSignature(row)}\n\`\`\``);
        if (row.description) blocks.push(stripXrefs(row.description));
      }
      for (const method of member.methods) {
        blocks.push(`#### ${member.name}.${method.name}`);
        blocks.push(`\`\`\`python\n${signatureOf(method, member.name)}\n\`\`\``);
        const text = sectionsToMarkdown(method.docstring);
        if (text) blocks.push(text);
      }
    }
  }
  return blocks.join('\n\n');
}

// Search-index projection: one heading and one content chunk per member,
// method, and field, ids matching the rendered anchors so results
// deep-link into the reference. Module docstrings index as page-level
// content.
export function apiModuleStructuredData(module: ApiModuleData): StructuredData {
  const headings: StructuredData['headings'] = [];
  const contents: StructuredData['contents'] = [];
  const push = (id: string, text: string) => {
    headings.push({ id, content: id });
    contents.push({ heading: id, content: text });
  };
  const intro = sectionsToMarkdown(module.docstring);
  if (intro) contents.push({ heading: undefined, content: intro });
  for (const member of module.members) {
    push(member.name, `${memberSignature(member)}\n${sectionsToMarkdown(member.docstring)}`);
    if (member.kind === 'class') {
      for (const row of mergedAttributeRows(member)) {
        push(
          `${member.name}.${row.name}`,
          `${fieldSignature(row)}\n${stripXrefs(row.description)}`,
        );
      }
      for (const method of member.methods) {
        push(
          `${member.name}.${method.name}`,
          `${signatureOf(method, member.name)}\n${sectionsToMarkdown(method.docstring)}`,
        );
      }
    }
  }
  return { headings, contents };
}

// The complete machine-readable text of a page: its prose, then its
// generated API reference when it declares one.
export async function pageLLMText(page: Page): Promise<string> {
  const base = await getLLMText(page);
  if (!page.data.api_module) return base;
  return `${base}\n\n${apiModuleToMarkdown(getApiModule(page.data.api_module))}`;
}
