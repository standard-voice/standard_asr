# Keeping Engines Alive in the Reference Server (Design Case)

**Status:** Draft, 2026-10-22, for roadmap item 12 (#27). Section 4 asks the maintainer to choose what a second call gets while an instance is busy; the rest of the design follows from that choice and assumes the recommended option. Nothing is implemented yet.

## 1. Problem

The reference server constructs a new engine for every REST request and every WebSocket session: `asyncio.to_thread(registry.create, model)` in `src/standard_asr/toolchain/server.py` at line 1564 for REST and line 799 for streaming, and `ModelRegistry.create` calls the plugin's factory every time (`src/standard_asr/plugins/discovery.py`, line 792). The first-party engines load their model weights lazily, on the first call. So every request pays for a model load, and a wire application that sends one utterance per request reloads the model for each utterance.

The size of the cost, measured on 2026-10-22 with std-mlx-audio on an Apple M5 Max: the first call on a new instance of Qwen3-ASR 0.6B (4-bit) took 1.3 to 2.3 seconds, and each later call on the same instance took 0.2 seconds, for the same 8-second recording. Parakeet TDT 0.6B v3 took 1.1 to 1.3 seconds, then 0.1 seconds.

## 2. Facts the design has to respect

Read on `origin/main` of each repository on 2026-10-22: standard_asr 31612eb, std-faster-whisper ef23a00, std-mlx-audio 080beb1, std-qwen3-asr 7b0bc54.

**Where engine work runs.**

- REST: `transcribe` runs in the event loop's default executor through `asyncio.to_thread` (`server.py`, line 1638), so successive calls can land on different threads.
- WebSocket: `start_transcription` runs on the event-loop thread, and so do the session's `_open`, `_produce`, and `_close` (the bridge, `server.py` lines 1356 to 1384; the producer task, `runtime/streaming.py` line 3089).
- Inside the engines: std-faster-whisper's session decodes in `asyncio.to_thread`; std-mlx-audio's session decodes inline on the event-loop thread; std-qwen3-asr does network I/O and starts no threads.

**What a call can leave behind.**

- A canceled `asyncio.to_thread` does not stop its thread (the specification says so in AR.6).
- When a REST client disconnects, uvicorn does not cancel the handler: the call runs to completion and the response is discarded.
- When a WebSocket client disconnects, the server's audio pump treats it as the end of audio and calls `end_audio()`, and teardown starts at the next send. `TranscriptionSession.__aexit__` cancels the producer and feed tasks, awaits them, then awaits `_close` (`runtime/streaming.py`, lines 3098 to 3103). If std-faster-whisper's producer was waiting on a decode, the decode thread keeps running after `__aexit__` returns, and its session does not await it in `_close`.
- When the engine's `_produce` yields a terminal event itself, or the event buffer overflows, the session stops iterating `_produce` without closing it (`runtime/streaming.py`, lines 3155 to 3166). The engine's async generator stays suspended, and its cleanup, such as a `finally` block that closes a connection, runs later in a task the event loop starts when the generator is collected, after `__aexit__` returned. A minimal reproduction of the pattern on 2026-10-22 showed the cleanup starting after the session had ended.

**Shutdown.** uvicorn stops accepting connections, closes every open WebSocket with code 1012, and waits for the HTTP requests in progress and for the handlers to return (`uvicorn/server.py` in 0.54.0; `run()` in `server.py` passes no `timeout_graceful_shutdown`, so the wait has no limit). Then it runs the application's lifespan shutdown. The application defines no lifespan today.

**The protocol.**

- It has no engine-wide close or release. `prepare()` on `EngineBase` is an idempotent warm-up (IC.11) that the server never calls.
- It says nothing about one instance serving several calls, in sequence or at the same time. Applications already create an engine once and call it many times; nothing states that this must work across threads and event loops.

**The first-party plugins.** None keeps per-session state on the engine: a session holds its own buffers and a reference to the engine. What the instance holds is the loaded model (std-faster-whisper, std-mlx-audio) or an HTTP client (std-qwen3-asr), with no lock. Per-request construction hides four problems that reuse exposes:

1. **std-qwen3-asr, a stream then a batch call.** Its backend keeps one `httpx.AsyncClient`, created lazily (`backends/vllm.py`, lines 133 and 146 to 154; the DashScope backend is the same). A batch call runs on a new event loop and closes the client afterwards, setting it to `None` (`vllm.py`, lines 329 to 333), so batch then batch, and batch then stream, work. A stream over SSE or DashScope uses the client on the server's event loop and no session closes it, so the next batch call takes a client bound to another loop.
2. **std-mlx-audio, a failed load.** `self._model = load(...)` (`engine.py`, line 481) runs before the model-family check (line 490), so a family mismatch leaves the wrong model on the instance, and the next call uses it.
3. **std-mlx-audio, threads.** MLX ties arrays to the thread that created them. The engine primes a freshly loaded model with 0.1 seconds of silence on the loading thread, after which "generates from any thread are stable" (`engine.py`, lines 498 to 512). A priming failure only logs a warning, and the same comment says that a generate on another thread can then abort the whole process. The comment at `_streaming.py` lines 495 to 501, which says generation must stay on the loading thread, predates the priming. Section 7 records a test of thread switching.
4. **std-faster-whisper, the session's close.** The decode thread can outlive the session, as described above.

**Error mapping** (`server.py`, lines 1563 to 1600 and 1672 to 1732; `docs/content/specification/server-api.md` §3.7): caller errors get 4xx (`AudioProcessingError` 400, `UnsupportedFeatureError` 422), operator states get 503 (absent configuration, unavailable artifacts), and everything else gets one scrubbed 500.

## 3. Goals and non-goals

Goals:

- Calls for one model key, one after another, use one instance and load the model once.
- Concurrent calls see no change in behavior.
- An instance serves a new call only when the server knows that the engine's work for the previous call has finished.
- An instance that faulted is dropped, and the next call constructs a new one.

Non-goals, which stay with item 19: eviction and memory limits, warm-up over the wire, and configuration per request. Also out of scope: whether one instance may serve two calls at the same time, which the protocol has not decided.

## 4. Decision needed: a second call while the instance is busy

The roadmap's starting point is "one instance per model key, one call on it at a time". That leaves open what a second call for the same model gets while the first runs. There are three answers.

| Option | What a second call gets | Memory | What changes for clients |
| --- | --- | --- | --- |
| **A. One idle instance per model key** | A new instance, as today. When a call ends cleanly, its instance becomes the idle one if the slot is empty, and is dropped otherwise. | N concurrent calls hold N copies, as today; one copy stays loaded afterwards. | Nothing, apart from speed: calls in sequence stop reloading. |
| **B. One instance, a queue** | It waits, first come first served, until the instance is free. | One copy per model key. | A long stream, such as an hour of live captions, blocks every other call for that model. To be usable it needs a wait limit and an error when the limit passes: new policy. |
| **C. One instance, fail fast** | A 503, or `service_unavailable` on a WebSocket, with a hint to retry. | One copy per model key. | Concurrent clients that work today start failing. |

A fourth option, a pool with a bound on instances and an explicit answer for the call that exceeds it, combines a memory bound with concurrency. It needs the same decisions as eviction, so it belongs with item 19.

**Recommendation: A.** It fixes the problem the roadmap names, a reload per utterance, without changing what concurrent clients get, and it adds no policy that item 19 would later have to undo. B and C buy a memory bound with a change in behavior. Item 19 decides eviction and limits anyway, and a bound on copies belongs with those decisions. The cost of A: concurrent use keeps today's memory exposure, so N concurrent streams on a GPU still load N copies, and while they run, the idle instances of other model keys stay loaded too. One copy per model key used stays loaded until the server stops.

## 5. Design, under option A

### 5.1 The idle instances

`create_app` creates one holder per application: a mapping from model key to one idle engine instance. A call takes the idle instance for its key, or constructs one with `registry.create` in a worker thread, exactly as today. Taking and returning an instance happen only on the event-loop thread, so the holder needs no lock. An instance is either idle in the holder or held by exactly one call, so one call at a time on an instance holds by construction.

Construction errors map to HTTP statuses and WebSocket error codes as they do today. A failed construction leaves nothing behind, and the next call tries again.

### 5.2 When a call ends, and when its instance is kept

An instance returns to the holder only after a call that succeeded, and only when the server can see that the engine's work for the call has finished. Every other ending drops it.

- **REST.** Kept when `transcribe` returned a result and the server encoded the response. Dropped on every exception, a caller's error included, and when the handler is canceled, which uvicorn does only at shutdown after a graceful timeout, while the worker thread may still be running.
- **WebSocket.** Kept when the session yielded its terminal `done` event to the bridge and the session's `__aexit__` returned without raising. Dropped on every other ending: a terminal `error` event, whether the engine sent it or a deadline ended the session; a client that left before `done`; a fault while forwarding events; a fault in `_close`; a canceled handler; and every establishment failure. The bridge reports this verdict to the route as a private return value, because it catches a fault in `_close` and a failed send itself.

Why not keep the instance after a caller's error: `AudioProcessingError` and `UnsupportedFeatureError` usually come from the standard layer before the engine runs, but nothing stops an engine from raising either one after it loaded or changed state, and neither exception promises that the instance still works. Keeping only after success needs no such promise.

Why `done` is enough, with one fix in the core: the session appends `done` only after the engine's `_produce` ended, so every decode it awaited has finished, and with `_close` returned, the session's work is over, for an engine that obeys rule 2 in 5.4. Today that fails in one case: an engine that yields its own terminal event leaves its generator suspended, and its cleanup runs after the session ended (section 2). So the session will close the engine's producer, and wait for its cleanup, before it appends the engine's terminal event, and likewise before it appends the error for a buffer overflow. The fix is worth making without reuse: an engine's cleanup then runs inside the session, every time. A deadline can also deliver a `done` that was already buffered; the producer had ended by then, so keeping the instance is still right. Whether the server then managed to send `done` to the client does not matter.

The cost: after an error, or a client that leaves before `done`, the next call for that model constructs a new instance, which is what every call costs today.

### 5.3 Shutdown

On an ordinary shutdown, uvicorn closes the open WebSocket sessions, so a session that had not yet yielded `done` ends without it and its instance is dropped. It waits for the HTTP requests in progress and for the handlers, and their instances are kept or dropped as in 5.2. Then the application's lifespan shutdown, new in this change, drops every idle instance. Two paths skip part of this: a second interrupt forces the exit without the wait or the lifespan shutdown, and an operator's graceful timeout cancels the handlers still running when it expires, which drops their instances. The protocol gets no engine-wide close in this change; item 19 needs one for eviction and can define it there. A thread that a canceled call left running finishes on its own, and the interpreter waits for it before the process exits.

### 5.4 Two rules the protocol gains

The specification's Chinese text gains two rules, in the section on engine instances and in §6.5 on the session lifecycle:

1. **Instances serve calls in sequence.** One engine instance MUST serve any number of calls one after another, in any order, each starting after the previous one completed: `transcribe` on any thread that runs no event loop, and `transcribe_async` and streaming sessions on any event loop. A call completes when it returns; a session completes when its `__aexit__` returns. The rule makes no promise about an instance after a call that raised or was canceled. Whether two calls may overlap stays unspecified.
2. **A session's close waits for its work.** A session's `_close` MUST return only after every operation the session started has finished, a thread it handed work to included; work it cannot cancel, it waits for. This sits beside the existing rule of §6.5 that every resource bound to an event loop is created in `__aenter__`.

Both describe what applications already assume when they create an engine once and call it many times. The server is the first host that depends on them across threads and event loops. A compliance check for rule 1 needs real inference, so it waits for the decision on side effects in compliance runs (items 15 and 17); until then, the first-party plugins are checked by hand (section 7).

### 5.5 What the operator sees

- The server logs each construction and each drop at INFO, and each reuse at DEBUG, so an operator sees every model load without an extra line for each request.
- `server-api.md` says that the server keeps one idle instance per model key, that each model key used keeps a model loaded until the server stops, and that an instance reads its configuration once, at construction (IC.9), so a change to the environment needs a restart. The help text of `standard-asr serve` says the same in a line.
- The text that assumes construction per request changes with the code: `server-api.md` lines 114, 130, and 187, and the docstring of `_create_engine_or_http_error`.

## 6. Plugin changes, merged before a release ships this

Each problem in section 2 becomes a companion pull request in its plugin's repository, so that no release of the core reuses instances while a first-party plugin breaks under reuse.

- **std-qwen3-asr:** a session creates the HTTP client it streams with when it opens and closes it when it ends, as §6.5 already requires of every resource bound to an event loop, so no later batch call or session on another event loop receives a client bound to the first one (rule 1). A cache of clients per event loop would also work, but it would need its own cleanup, since every `SyncSession` runs its own loop.
- **std-faster-whisper:** the session keeps the future of each decode, so that it stays awaitable after the producer is canceled, and waits for it in `_close` (rule 2).
- **std-mlx-audio:** publish `_model` only after both the model-family check and the priming succeed, and fail the load otherwise, so that no call can use an unchecked model or generate on another thread without priming, which can abort the process; and update the stale comment in `_streaming.py`.

## 7. Evidence and tests

**Thread switching in std-mlx-audio,** tested on 2026-10-22 on an Apple M5 Max with macOS 27.0.1, mlx 0.31.2, mlx-audio 0.4.4, and std-mlx-audio 080beb1, with the models from the local cache. One instance each of Qwen3-ASR 0.6B (4-bit) and Parakeet TDT 0.6B v3 served batch calls on two worker threads and streams on the event loops of three threads, starting with a batch call and, on a second instance, starting with a stream. Every call returned the same text. This shows that priming works for these two model families on this machine. It does not cover a family whose priming fails.

**Tests the implementation adds,** with a scripted engine that counts its constructions:

- two REST calls in sequence construct one instance; two concurrent calls construct two, and one stays idle afterwards;
- every exception drops the instance: a caller's error, an engine error, and a 503;
- a stream that ends with `done` keeps the instance, a `done` delivered at a deadline included; an engine `error`, a deadline that ends the session, a client that leaves early, and a fault in `_close` each drop it;
- in the session runtime, an engine that yields its own terminal event, or overflows the buffer, has its producer closed and its cleanup finished before the terminal event reaches the consumer;
- a canceled REST handler drops the instance;
- the lifespan shutdown drops the idle instances.

## 8. Alternatives not taken

- **Keep an instance after a canceled call** by running engine calls on a server-owned executor and watching each thread to its end. Cancellation happens only at shutdown, and dropping the instance is safe, so the machinery buys nothing.
- **Let the session report how it ended** through a new public property. The bridge already sees the terminal event, so the runtime needs no new API.
- **Define an engine-wide close now.** Nothing in this change needs it: the server releases an instance by dropping its last reference, both for an overflow instance while it serves and at shutdown, which frees what the instance holds, as the end of every request does today. Eviction, in item 19, may want a deterministic release, for example to return accelerator memory at a known moment, and can define the call there.
