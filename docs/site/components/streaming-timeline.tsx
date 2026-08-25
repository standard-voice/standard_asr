'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, RotateCcw, StepForward } from 'lucide-react';

// An interactive playback of the streaming event protocol: the wire events
// on the left, the application state they reduce to on the right. The
// reduce mirrors the documented core reduce (reading-order list + text
// map, supersede splices replacements at the retired block's position).

type EventType = 'partial' | 'final' | 'supersede' | 'progress' | 'done';

interface DemoEvent {
  type: EventType;
  segment_id?: string;
  text?: string;
  stable_until?: number;
  old_ids?: string[];
  new_ids?: string[];
  finality?: 'final' | 'closed';
  note?: string;
}

interface Scenario {
  id: string;
  label: string;
  intro: string;
  events: DemoEvent[];
}

const SCENARIOS: Scenario[] = [
  {
    id: 'rewrite',
    label: 'Rewriting interims',
    intro: 'Interim text may change with every event until the segment goes final.',
    events: [
      {
        type: 'partial',
        segment_id: 's1',
        text: 'welcome',
        note: 'A partial may change on the next event for this segment.',
      },
      { type: 'partial', segment_id: 's1', text: 'welcome everyone' },
      {
        type: 'partial',
        segment_id: 's1',
        text: 'welcome everyone to the standard',
        note: 'Same segment_id: the new text replaces the old guess.',
      },
      {
        type: 'final',
        segment_id: 's1',
        text: 'Welcome, everyone, to the standards meeting.',
        note: 'final: this text is settled against new audio.',
      },
      { type: 'partial', segment_id: 's2', text: 'the agenda' },
      { type: 'partial', segment_id: 's2', text: 'the agenda has one item' },
      { type: 'final', segment_id: 's2', text: 'The agenda has one item.' },
      { type: 'done', note: 'done: the session is complete. No more events follow.' },
    ],
  },
  {
    id: 'append',
    label: 'Append-only stream',
    intro: 'Some engines never revise: the stable prefix only grows (stable_until).',
    events: [
      {
        type: 'partial',
        segment_id: 's1',
        text: 'switching',
        stable_until: 9,
        note: 'stable_until marks the frozen prefix, in codepoints.',
      },
      { type: 'partial', segment_id: 's1', text: 'switching engines', stable_until: 9 },
      {
        type: 'partial',
        segment_id: 's1',
        text: 'switching engines is a one line',
        stable_until: 20,
        note: 'The frozen prefix only ever grows.',
      },
      {
        type: 'partial',
        segment_id: 's1',
        text: 'switching engines is a one-line change',
        stable_until: 38,
      },
      {
        type: 'final',
        segment_id: 's1',
        text: 'switching engines is a one-line change',
        note: 'final: the whole segment is now settled.',
      },
      { type: 'done' },
    ],
  },
  {
    id: 'supersede',
    label: 'Two-pass rescoring',
    intro: 'A second decoding pass may replace segments that were already final.',
    events: [
      { type: 'partial', segment_id: 's1', text: 'so the total is' },
      { type: 'final', segment_id: 's1', text: 'so the total is' },
      { type: 'partial', segment_id: 's2', text: 'fourteen' },
      {
        type: 'final',
        segment_id: 's2',
        text: 'fourteen',
        note: 'Both segments are final - but a supersede can still replace them.',
      },
      {
        type: 'supersede',
        old_ids: ['s1', 's2'],
        new_ids: ['s3'],
        note: 'supersede retires s1 and s2; s3 takes their position in reading order.',
      },
      { type: 'progress', note: 'progress: a heartbeat, no transcript content.' },
      {
        type: 'final',
        segment_id: 's3',
        text: 'So the total is 14.',
        note: "The replacement segment's own events arrive after the supersede.",
      },
      { type: 'done' },
    ],
  },
];

interface SegmentState {
  text: string;
  stableUntil?: number;
  final: boolean;
}

interface ReducedState {
  order: string[];
  segments: Record<string, SegmentState>;
  done: boolean;
}

// stable_until counts codepoints (the spec's unit), not UTF-16 units.
function slicePoints(text: string, start: number, end?: number): string {
  return [...text].slice(start, end).join('');
}

function reduce(events: DemoEvent[]): ReducedState {
  const order: string[] = [];
  const segments: Record<string, SegmentState> = {};
  let done = false;
  for (const event of events) {
    if ((event.type === 'partial' || event.type === 'final') && event.segment_id) {
      if (!order.includes(event.segment_id)) order.push(event.segment_id);
      segments[event.segment_id] = {
        text: event.text ?? '',
        stableUntil: event.stable_until,
        final: event.type === 'final',
      };
    } else if (event.type === 'supersede') {
      const oldIds = event.old_ids ?? [];
      const positions = oldIds.map((id) => order.indexOf(id));
      const start = positions[0];
      // reduce_event raises on an empty, unknown, reordered, or
      // non-contiguous old_ids run; scenarios are authored data, so a
      // violation is an authoring bug -- fail like the library, never
      // splice a wrong transcript.
      if (oldIds.length === 0 || start === -1 || !positions.every((at, i) => at === start + i)) {
        throw new Error(`supersede old_ids must be a known, in-order, contiguous run: ${oldIds}`);
      }
      order.splice(start, oldIds.length, ...(event.new_ids ?? []));
      for (const oldId of oldIds) delete segments[oldId];
    } else if (event.type === 'done') {
      done = true;
    }
  }
  return { order, segments, done };
}

const TYPE_STYLES: Record<EventType, string> = {
  partial: 'bg-amber-500/15 text-amber-700 dark:text-amber-400',
  final: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400',
  supersede: 'bg-violet-500/15 text-violet-700 dark:text-violet-400',
  progress: 'bg-slate-500/15 text-slate-600 dark:text-slate-400',
  done: 'bg-(--color-brand-soft) text-(--color-brand)',
};

function eventSummary(event: DemoEvent): string {
  if (event.type === 'partial' || event.type === 'final') {
    return `${event.segment_id}  "${event.text}"`;
  }
  if (event.type === 'supersede') {
    return `old_ids=[${event.old_ids?.join(', ')}]  new_ids=[${event.new_ids?.join(', ')}]`;
  }
  return '';
}

export function StreamingTimeline() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const scenario = useMemo(() => SCENARIOS.find((s) => s.id === scenarioId)!, [scenarioId]);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const emitted = scenario.events.slice(0, cursor);
  const state = useMemo(() => reduce(emitted), [emitted]);
  const atEnd = cursor >= scenario.events.length;
  const running = playing && !atEnd;
  const lastNote = emitted.findLast((event) => event.note)?.note;

  const selectScenario = useCallback((id: string) => {
    setScenarioId(id);
    setCursor(0);
    setPlaying(true);
  }, []);

  useEffect(() => {
    if (!running) return;
    // Timer-driven playback: each tick advances the cursor by one event.
    const timer = setTimeout(() => setCursor((value) => value + 1), 900);
    return () => clearTimeout(timer);
  }, [running, cursor]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [cursor]);

  return (
    <div className="rounded-xl border bg-fd-card text-sm shadow-sm">
      <div className="flex flex-wrap items-center gap-1.5 border-b p-2.5">
        {SCENARIOS.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => selectScenario(option.id)}
            className={`rounded-md px-2.5 py-1 text-[13px] transition-colors ${
              option.id === scenarioId
                ? 'bg-(--color-brand) text-fd-primary-foreground'
                : 'text-fd-muted-foreground hover:bg-fd-accent'
            }`}
          >
            {option.label}
          </button>
        ))}
        <div className="ms-auto flex items-center gap-1">
          <button
            type="button"
            aria-label={running ? 'Pause' : 'Play'}
            onClick={() => {
              if (atEnd) {
                setCursor(0);
                setPlaying(true);
                return;
              }
              setPlaying((value) => !value);
            }}
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-foreground"
          >
            {running ? <Pause className="size-4" /> : <Play className="size-4" />}
          </button>
          <button
            type="button"
            aria-label="Step forward"
            disabled={atEnd}
            onClick={() => {
              setPlaying(false);
              setCursor((value) => Math.min(value + 1, scenario.events.length));
            }}
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-foreground disabled:opacity-40"
          >
            <StepForward className="size-4" />
          </button>
          <button
            type="button"
            aria-label="Reset"
            onClick={() => {
              setPlaying(false);
              setCursor(0);
            }}
            className="rounded-md p-1.5 text-fd-muted-foreground hover:bg-fd-accent hover:text-fd-foreground"
          >
            <RotateCcw className="size-4" />
          </button>
        </div>
      </div>

      <p className="border-b px-3.5 py-2 text-[13px] text-fd-muted-foreground">{scenario.intro}</p>

      <div className="grid md:grid-cols-2">
        <div className="border-b md:border-b-0 md:border-e">
          <div className="px-3.5 pt-3 text-[11px] font-semibold uppercase tracking-wider text-fd-muted-foreground">
            Session events
          </div>
          <div ref={logRef} className="h-56 overflow-y-auto p-2.5 font-mono text-[12.5px]">
            {emitted.length === 0 ? (
              <div className="px-1 py-2 text-fd-muted-foreground">
                Press play to start the session.
              </div>
            ) : null}
            {emitted.map((event, i) => (
              <div
                key={i}
                className={`flex items-baseline gap-2 rounded-md px-1.5 py-1 ${
                  i === emitted.length - 1 ? 'bg-fd-accent' : ''
                }`}
              >
                <span
                  className={`inline-block w-20 shrink-0 rounded px-1.5 py-px text-center text-[11px] font-semibold ${TYPE_STYLES[event.type]}`}
                >
                  {event.type}
                </span>
                <span className="truncate text-fd-muted-foreground">{eventSummary(event)}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="px-3.5 pt-3 text-[11px] font-semibold uppercase tracking-wider text-fd-muted-foreground">
            Your application&apos;s view
          </div>
          <div className="flex h-56 flex-col gap-1.5 overflow-y-auto p-3.5">
            {state.order.map((id) => {
              const segment = state.segments[id];
              if (!segment) return null;
              const stable =
                segment.stableUntil !== undefined
                  ? slicePoints(segment.text, 0, segment.stableUntil)
                  : undefined;
              const tail =
                segment.stableUntil !== undefined
                  ? slicePoints(segment.text, segment.stableUntil)
                  : undefined;
              return (
                <div
                  key={id}
                  className={`rounded-lg border px-3 py-1.5 transition-colors ${
                    segment.final
                      ? 'border-fd-border bg-fd-background'
                      : 'border-dashed border-amber-500/50 bg-amber-500/5'
                  }`}
                >
                  <span className="me-2 font-mono text-[10.5px] text-fd-muted-foreground">
                    {id}
                  </span>
                  {stable !== undefined ? (
                    <>
                      <span>{stable}</span>
                      <span className="text-fd-muted-foreground italic">{tail}</span>
                    </>
                  ) : (
                    <span className={segment.final ? '' : 'text-fd-muted-foreground italic'}>
                      {segment.text}
                    </span>
                  )}
                </div>
              );
            })}
            {state.done ? (
              <div className="mt-auto pt-2 font-mono text-[11.5px] text-(--color-brand)">
                session complete - stream == result
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="min-h-9 border-t px-3.5 py-2 text-[12.5px] text-fd-muted-foreground">
        {lastNote ?? 'Events on the left; the reduced transcript state on the right.'}
      </div>
    </div>
  );
}
