import { codeToHtml } from 'shiki';

// A statically highlighted code panel for the landing page.
export async function CodeCard({
  title,
  code,
  lang = 'python',
}: {
  title: string;
  code: string;
  lang?: string;
}) {
  const html = await codeToHtml(code, {
    lang,
    themes: { light: 'github-light', dark: 'github-dark' },
    defaultColor: false,
  });
  return (
    <figure className="min-w-0 overflow-hidden rounded-xl border bg-fd-card shadow-sm">
      <figcaption className="flex items-center gap-2 border-b px-4 py-2.5 font-mono text-[12px] font-medium text-fd-muted-foreground">
        <span aria-hidden className="size-1.5 bg-(--color-brand)" />
        {title}
      </figcaption>
      <div
        className="landing-code overflow-x-auto p-4 text-[13px] leading-relaxed"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </figure>
  );
}
