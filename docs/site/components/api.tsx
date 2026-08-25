import { Callout } from 'fumadocs-ui/components/callout';
import Link from 'next/link';
import type { ReactNode } from 'react';
import {
  type Annotation,
  type ApiAttribute,
  type ApiClass,
  type ApiFunction,
  type ApiMember,
  type ApiModuleData,
  type DocSection,
  type NamedItem,
  resolveSymbol,
} from '@/lib/api';
import { annotationText, classSignatureOf, mergedAttributeRows, signatureOf } from '@/lib/api-text';
import { markdownToHtml } from '@/lib/markdown';
import { codeToHtml } from 'shiki';
import { gitConfig } from '@/lib/shared';

// Server-rendered API reference: pure presentation over the neutral data in
// .generated/api.json. Docstring prose stays authored in the Python source;
// nothing here invents wording.

function AnnotationCode({ annotation }: { annotation: Annotation }) {
  if (!annotation) return null;
  return (
    <code className="text-[13px]">
      {annotation.map((token, i) => {
        const href = token.target ? resolveSymbol(token.target) : undefined;
        return href ? (
          <Link key={i} href={href} className="text-fd-primary hover:underline">
            {token.text}
          </Link>
        ) : (
          <span key={i}>{token.text}</span>
        );
      })}
    </code>
  );
}

async function Signature({ code }: { code: string }) {
  const html = await codeToHtml(code, {
    lang: 'python',
    themes: { light: 'github-light', dark: 'github-dark' },
    defaultColor: false,
  });
  return (
    <div
      className="api-signature not-prose overflow-x-auto rounded-lg border bg-fd-card px-4 py-3 text-[13px]"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

async function Prose({ markdown }: { markdown: string }) {
  const html = await markdownToHtml(markdown);
  // prose-shiki scopes the code-block styling for shiki output embedded in
  // the typography context (see global.css).
  return <div className="prose-shiki" dangerouslySetInnerHTML={{ __html: html }} />;
}

async function NamedItemsTable({
  items,
  nameHeader,
  rowId,
}: {
  items: NamedItem[];
  nameHeader: string;
  // Anchors the row (attribute deep links); parameters tables pass none.
  rowId?: (item: NamedItem) => string | undefined;
}) {
  return (
    <div className="not-prose overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-fd-muted/50 text-left">
            <th className="px-3 py-2 font-medium">{nameHeader}</th>
            <th className="px-3 py-2 font-medium">Type</th>
            <th className="min-w-64 px-3 py-2 font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {await Promise.all(
            items.map(async (item, i) => (
              <tr
                key={i}
                id={rowId?.(item)}
                className="scroll-mt-24 border-b last:border-b-0 align-top"
              >
                <td className="px-3 py-2 whitespace-nowrap">
                  {item.name ? <code className="text-[13px]">{item.name}</code> : null}
                  {item.default !== undefined ? (
                    <div className="mt-0.5 text-xs text-fd-muted-foreground">
                      = <code>{item.default}</code>
                    </div>
                  ) : null}
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <AnnotationCode annotation={item.annotation} />
                  {item.constraints ? (
                    <div className="mt-0.5 text-xs text-fd-muted-foreground">
                      {item.constraints.join(', ')}
                    </div>
                  ) : null}
                </td>
                <td className="px-3 py-2 text-fd-muted-foreground [&_p]:my-1 [&_code]:text-[12px]">
                  <div
                    dangerouslySetInnerHTML={{ __html: await markdownToHtml(item.description) }}
                  />
                </td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

const SECTION_TITLES: Record<string, string> = {
  parameters: 'Parameters',
  attributes: 'Attributes',
  returns: 'Returns',
  yields: 'Yields',
  receives: 'Receives',
  raises: 'Raises',
  warns: 'Warns',
  examples: 'Examples',
};

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="not-prose mt-4 mb-2 text-xs font-semibold uppercase tracking-wider text-fd-muted-foreground">
      {children}
    </div>
  );
}

async function Sections({ sections }: { sections: DocSection[] }) {
  return (
    <>
      {await Promise.all(
        sections.map(async (section, i) => {
          switch (section.kind) {
            case 'text':
              return <Prose key={i} markdown={section.value} />;
            case 'parameters':
            case 'attributes':
              return (
                <div key={i}>
                  <SectionLabel>{SECTION_TITLES[section.kind]}</SectionLabel>
                  <NamedItemsTable items={section.items} nameHeader="Name" />
                </div>
              );
            case 'returns':
            case 'yields':
            case 'receives':
            case 'raises':
            case 'warns':
              return (
                <div key={i}>
                  <SectionLabel>{SECTION_TITLES[section.kind]}</SectionLabel>
                  <ul className="not-prose flex flex-col gap-1.5 text-sm">
                    {await Promise.all(
                      section.items.map(async (item, j) => (
                        <li key={j} className="flex flex-col gap-0.5">
                          <AnnotationCode annotation={item.annotation} />
                          <div
                            className="text-fd-muted-foreground [&_p]:my-0.5"
                            dangerouslySetInnerHTML={{
                              __html: await markdownToHtml(item.description),
                            }}
                          />
                        </li>
                      )),
                    )}
                  </ul>
                </div>
              );
            case 'examples':
              return (
                <div key={i}>
                  <SectionLabel>Examples</SectionLabel>
                  <Prose markdown={section.value} />
                </div>
              );
            case 'admonition': {
              // An "Example:" admonition is code content: render it like the
              // Examples section, not as a callout box.
              if (section.type.toLowerCase().startsWith('example')) {
                return (
                  <div key={i}>
                    <SectionLabel>{section.type}</SectionLabel>
                    <Prose markdown={section.value} />
                  </div>
                );
              }
              return (
                <Callout
                  key={i}
                  title={section.type}
                  type={section.type.toLowerCase() === 'warning' ? 'warn' : 'info'}
                >
                  <div
                    className="prose-shiki [&_p]:my-1"
                    dangerouslySetInnerHTML={{ __html: await markdownToHtml(section.value) }}
                  />
                </Callout>
              );
            }
            case 'deprecated':
              return (
                <Callout key={i} title="Deprecated" type="warn">
                  <div
                    className="prose-shiki"
                    dangerouslySetInnerHTML={{ __html: await markdownToHtml(section.value) }}
                  />
                </Callout>
              );
            default:
              return null;
          }
        }),
      )}
    </>
  );
}

function KindBadge({ kind }: { kind: string }) {
  return (
    <span className="rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-medium text-fd-muted-foreground">
      {kind}
    </span>
  );
}

function SourceLink({ source }: { source: { file: string; line: number } | null }) {
  if (!source) return null;
  const href = `https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/${source.file}#L${source.line}`;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="ms-auto text-xs text-fd-muted-foreground hover:text-fd-foreground"
    >
      Source
    </a>
  );
}

function MemberHeader({
  id,
  name,
  kind,
  source,
  level,
}: {
  id: string;
  name: string;
  kind: string;
  source: { file: string; line: number } | null;
  level: 2 | 3;
}) {
  const HeadingTag = level === 2 ? 'h2' : 'h3';
  return (
    <HeadingTag id={id} className="not-prose flex scroll-mt-24 items-center gap-2.5 border-b pb-2">
      <a href={`#${id}`} className="group flex min-w-0 items-center gap-2.5">
        <code
          className={`truncate font-mono font-semibold ${level === 2 ? 'text-lg' : 'text-base'}`}
        >
          {name}
        </code>
        <span className="opacity-0 transition-opacity group-hover:opacity-60">#</span>
      </a>
      <KindBadge kind={kind} />
      <SourceLink source={source} />
    </HeadingTag>
  );
}

async function FunctionDoc({ fn, owner }: { fn: ApiFunction; owner?: string }) {
  const id = owner ? `${owner}.${fn.name}` : fn.name;
  const noun = owner
    ? fn.labels.includes('classmethod')
      ? 'class method'
      : fn.labels.includes('staticmethod')
        ? 'static method'
        : 'method'
    : 'function';
  return (
    <section className="flex flex-col gap-3">
      <MemberHeader
        id={id}
        name={fn.name}
        kind={fn.async ? `async ${noun}` : noun}
        source={fn.source}
        level={owner ? 3 : 2}
      />
      <Signature code={signatureOf(fn, owner)} />
      <Sections sections={fn.docstring} />
    </section>
  );
}

async function ClassDoc({ cls }: { cls: ApiClass }) {
  const attributeRows = mergedAttributeRows(cls);
  const prose = cls.docstring.filter((section) => section.kind !== 'attributes');
  return (
    <section className="flex flex-col gap-3">
      <MemberHeader id={cls.name} name={cls.name} kind="class" source={cls.source} level={2} />
      <Signature code={classSignatureOf(cls)} />
      <Sections sections={prose} />
      {attributeRows.length > 0 ? (
        <div>
          <SectionLabel>Attributes</SectionLabel>
          <NamedItemsTable
            items={attributeRows}
            nameHeader="Name"
            rowId={(item) => (item.name ? `${cls.name}.${item.name}` : undefined)}
          />
        </div>
      ) : null}
      {cls.methods.length > 0 ? (
        <div className="mt-2 flex flex-col gap-8 border-s ps-5">
          {cls.methods.map((method) => (
            <FunctionDoc key={method.name} fn={method} owner={cls.name} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

async function AttributeDoc({ attr }: { attr: ApiAttribute }) {
  let code = attr.name;
  if (attr.annotation) code += `: ${annotationText(attr.annotation)}`;
  if (attr.default !== null) code += ` = ${attr.default}`;
  if (attr.constraints.length > 0) code += `  # ${attr.constraints.join(', ')}`;
  return (
    <section className="flex flex-col gap-3">
      <MemberHeader
        id={attr.name}
        name={attr.name}
        kind="attribute"
        source={attr.source}
        level={2}
      />
      <Signature code={code} />
      <Sections sections={attr.docstring} />
    </section>
  );
}

function MemberDoc({ member }: { member: ApiMember }) {
  if (member.kind === 'class') return <ClassDoc cls={member} />;
  if (member.kind === 'attribute') return <AttributeDoc attr={member} />;
  return <FunctionDoc fn={member} />;
}

export async function ApiModuleDoc({ module }: { module: ApiModuleData }) {
  return (
    <div className="flex flex-col gap-10">
      <Sections sections={module.docstring} />
      {module.members.map((member) => (
        <MemberDoc key={member.name} member={member} />
      ))}
    </div>
  );
}
