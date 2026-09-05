import Link from 'next/link';
import type { Metadata } from 'next';
import {
  ArrowRight,
  AudioLines,
  Blocks,
  FileCheck2,
  MessageSquareWarning,
  Server,
  SlidersHorizontal,
} from 'lucide-react';
import { CodeCard } from '@/components/code-card';
import { StreamingTimeline } from '@/components/streaming-timeline';
import { githubUrl, siteUrl } from '@/lib/shared';

export const metadata: Metadata = {
  description:
    'The open standard interface between applications and speech-recognition engines. Integrate once, gain every compliant engine.',
  // Absolute, not metadataBase-relative: relative resolution drops the
  // GitHub Pages base path.
  alternates: { canonical: `${siteUrl}/` },
  openGraph: { url: `${siteUrl}/` },
};

const APP_DEV_CODE = `from standard_asr import discover_models

registry = discover_models()
engine = registry.create("faster-whisper/large-v3")

result = engine.transcribe("meeting.wav")
print(result.text)

# Switching engines is a one-line change:
engine = registry.create("mlx-audio/qwen3-asr-0.6b")`;

const ENGINE_DEV_CODE = `from standard_asr.engine import EngineBase, PreparedAudio

class MyEngine(EngineBase):
    # Declare properties, capabilities, and a typed config...

    def _transcribe(self, prepared: PreparedAudio, params):
        audio = prepared.array  # 16 kHz float32 mono, per your Properties
        return my_model_infer(audio)

# One interface -> CLI, HTTP/WebSocket server, and a
# compliance test suite, for free.`;

const FEATURES = [
  {
    icon: AudioLines,
    title: 'Audio negotiation',
    body: 'Hand over a path, bytes, an array, or a URL. The standard layer negotiates the form your engine accepts, deterministically.',
  },
  {
    icon: SlidersHorizontal,
    title: 'Capability discovery',
    body: 'engine.supports("batch.word_timestamps") answers before you call. Fail-closed: an absent declaration means unsupported.',
  },
  {
    icon: MessageSquareWarning,
    title: 'Structured diagnostics',
    body: 'Lossy conversions and degraded paths surface as structured diagnostics. Silent wrong results are the cardinal sin.',
  },
  {
    icon: FileCheck2,
    title: 'Compliance suite',
    body: 'standard-asr compliance run verifies an implementation against the specification, one command.',
  },
  {
    icon: Server,
    title: 'Reference server',
    body: 'Expose any compliant engine over HTTP and WebSocket, so non-Python applications get the same capabilities.',
  },
  {
    icon: Blocks,
    title: 'Plugin discovery',
    body: 'Entry-point based: install a plugin and it appears in discover_models(). Zero application-side configuration.',
  },
];

const STAKEHOLDERS = [
  {
    title: 'Application developers',
    body: 'One integration that works with every compliant engine. Zero vendor lock-in. Automatic discovery of whatever the user installs.',
    href: '/docs/app-developers/discover-and-use',
    cta: 'Discover & Use',
  },
  {
    title: 'ASR engine developers',
    body: 'Focus on the model. Implement one interface and get a CLI, a reference server, and a compliance test suite for free.',
    href: '/docs/engine-authors/adapt-an-asr-system',
    cta: 'Adapt an ASR System',
  },
  {
    title: 'End users',
    body: 'Install the plugin whose engine fits your language or domain and use it immediately, without waiting for the app author to add support.',
    href: '/docs/quickstart',
    cta: 'Quickstart',
  },
];

function Waveform() {
  const bars = [14, 26, 40, 30, 52, 38, 60, 34, 48, 24, 44, 56, 36, 50, 28, 42, 18, 32, 22, 12];
  return (
    <div aria-hidden className="flex h-16 items-center gap-1.5 opacity-80">
      {bars.map((height, i) => (
        <span
          key={i}
          className="waveform-bar w-1 rounded-full bg-(--color-brand)"
          style={{ height: `${height}%`, animationDelay: `${i * 90}ms` }}
        />
      ))}
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* Hero */}
      <section className="mx-auto w-full max-w-[1200px] px-6 pb-16 pt-20 md:pt-28">
        <Waveform />
        <h1 className="mt-8 max-w-3xl font-mono text-4xl font-bold leading-tight tracking-tight md:text-6xl">
          The standard interface for ASR inference
          <span aria-hidden className="ms-3 inline-block size-4 bg-(--color-brand) md:size-6" />
        </h1>
        <p className="mt-6 max-w-2xl text-lg text-fd-muted-foreground">
          Standard ASR defines a vendor-neutral protocol between applications and speech-recognition
          engines. Applications integrate speech-to-text once and gain every compliant engine;
          engines implement it once and reach every application. Think USB-C for ASR.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            href="/docs/quickstart"
            className="inline-flex items-center gap-2 rounded-lg bg-(--color-brand) px-5 py-2.5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started <ArrowRight className="size-4" />
          </Link>
          <Link
            href="/docs/specification/protocol"
            className="inline-flex items-center gap-2 rounded-lg border px-5 py-2.5 font-medium transition-colors hover:bg-fd-accent"
          >
            Read the specification
          </Link>
          <code className="rounded-lg border bg-fd-card px-4 py-2.5 font-mono text-[13px] text-fd-muted-foreground">
            pip install &quot;standard-asr[audio]&quot;
          </code>
        </div>
        <p className="mt-6 text-sm text-fd-muted-foreground">
          <span className="me-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-400">
            Pre-release
          </span>
          Breaking changes may occur before v1.0.0. Try it out and tell us what you think.
        </p>
      </section>

      {/* Two sides, one protocol */}
      <section className="border-t bg-fd-muted/30">
        <div className="mx-auto w-full max-w-[1200px] px-6 py-16">
          <h2 className="font-mono text-2xl font-bold tracking-tight md:text-3xl">
            Implement once. Interoperate with everything.
          </h2>
          <p className="mt-3 max-w-2xl text-fd-muted-foreground">
            Like the OpenAI Chat Completion API did for LLMs: once a protocol is the common
            language, both sides of it stop writing adapters.
          </p>
          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            <CodeCard title="the application side" code={APP_DEV_CODE} />
            <CodeCard title="the engine side" code={ENGINE_DEV_CODE} />
          </div>
        </div>
      </section>

      {/* Streaming */}
      <section className="border-t">
        <div className="mx-auto w-full max-w-[1200px] px-6 py-16">
          <h2 className="font-mono text-2xl font-bold tracking-tight md:text-3xl">
            One event protocol for every streaming engine
          </h2>
          <p className="mt-3 max-w-2xl text-fd-muted-foreground">
            Real-time ASR is the most fragmented part of the ecosystem: some engines rewrite interim
            results, some never revise a token, some merge segments after a second pass. Standard
            ASR unifies all of it under six event types with explicit stability guarantees —
            designed against a survey of 30+ real engine APIs.
          </p>
          <div className="mt-8">
            <StreamingTimeline />
          </div>
          <p className="mt-4 text-sm text-fd-muted-foreground">
            The right panel runs the documented core reduce — the same logic as{' '}
            <code className="rounded bg-fd-muted px-1 py-0.5 text-[12px]">reduce_event</code> in{' '}
            <Link href="/docs/app-developers/streaming" className="text-fd-primary hover:underline">
              the streaming guide
            </Link>
            .
          </p>
        </div>
      </section>

      {/* Features */}
      <section className="border-t bg-fd-muted/30">
        <div className="mx-auto w-full max-w-[1200px] px-6 py-16">
          <h2 className="font-mono text-2xl font-bold tracking-tight md:text-3xl">
            The plumbing, standardized
          </h2>
          <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature) => (
              <div key={feature.title} className="rounded-xl border bg-fd-card p-5 shadow-sm">
                <feature.icon className="size-5 text-(--color-brand)" />
                <h3 className="mt-3 font-semibold">{feature.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                  {feature.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stakeholders */}
      <section className="border-t">
        <div className="mx-auto w-full max-w-[1200px] px-6 py-16">
          <h2 className="font-mono text-2xl font-bold tracking-tight md:text-3xl">
            Who is it for?
          </h2>
          <div className="mt-8 grid gap-5 lg:grid-cols-3">
            {STAKEHOLDERS.map((item) => (
              <Link
                key={item.title}
                href={item.href}
                className="group rounded-xl border bg-fd-card p-6 shadow-sm transition-colors hover:border-(--color-brand)"
              >
                <h3 className="font-semibold">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-fd-muted-foreground">{item.body}</p>
                <span className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-fd-primary">
                  {item.cta}
                  <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t">
        <div className="mx-auto flex w-full max-w-[1200px] flex-wrap items-center gap-x-6 gap-y-2 px-6 py-8 text-sm text-fd-muted-foreground">
          <span className="inline-flex items-center gap-2 font-mono font-semibold text-fd-foreground">
            Standard ASR
            <span aria-hidden className="inline-block size-1.5 bg-(--color-brand)" />
          </span>
          <span>Apache-2.0</span>
          <Link href="/docs" className="hover:text-fd-foreground">
            Documentation
          </Link>
          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer noopener"
            className="hover:text-fd-foreground"
          >
            GitHub
          </a>
          <a
            href="https://pypi.org/project/standard-asr/"
            target="_blank"
            rel="noreferrer noopener"
            className="hover:text-fd-foreground"
          >
            PyPI
          </a>
        </div>
      </footer>
    </main>
  );
}
