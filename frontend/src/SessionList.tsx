import { useEffect, useState } from "react";
import { Upload, FileText, Clock, ArrowRight, Plus } from "@phosphor-icons/react";
import type { LanguageInfo, SessionSummary } from "./types";

function timeAgo(dateStr: string): string {
  const diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(dateStr).toLocaleDateString();
}

export default function SessionList() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [lang, setLang] = useState("sogdian");
  const [script, setScript] = useState("standard");
  const [fragmentPath, setFragmentPath] = useState("");
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const [s, l] = await Promise.all([
          fetch("/api/sessions").then((r) => r.json()),
          fetch("/api/languages").then((r) => r.json()),
        ]);
        if (disposed) return;
        setSessions(s);
        setLanguages(l);
        if (l[0]) setLang(l[0].code);
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!disposed) setLoading(false);
      }
    })();
    return () => { disposed = true; };
  }, []);

  async function createSession(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setFormError(null);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          language: lang,
          script_variant: script,
          ingestion_path: "local_file",
          source: fragmentPath,
          crop_manuscript_area: false,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      window.location.href = `/ui/${data.session_id}`;
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 text-stone-900 dark:text-stone-100">
      <header className="h-14 flex items-center justify-between px-6 border-b border-stone-200 dark:border-stone-800">
        <h1 className="text-sm font-medium uppercase tracking-wider text-stone-600 dark:text-stone-400">
          msocr · Sogdian Annotation
        </h1>
        <a
          href="/"
          className="text-xs text-stone-500 hover:text-accent transition-colors"
        >
          Plates
        </a>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 grid gap-8 md:grid-cols-[1fr_360px]">
        <section>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-stone-500 mb-3">
            <Clock size={14} /> Recent Sessions
          </div>
          {loading ? (
            <div className="text-sm text-stone-400 italic">Loading…</div>
          ) : error ? (
            <div className="text-sm text-red-600 dark:text-red-400">{error}</div>
          ) : sessions.length === 0 ? (
            <div className="text-sm text-stone-400 italic">No sessions yet — create one.</div>
          ) : (
            <ul className="space-y-2">
              {sessions.map((s) => (
                <li key={s.session_id}>
                  <a
                    href={`/ui/${s.session_id}`}
                    className="flex gap-3 items-center p-3 rounded-lg border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 hover:border-accent hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors group"
                  >
                    <img
                      src={`/api/sessions/${s.session_id}/image`}
                      alt="fragment"
                      className="w-16 h-16 object-cover rounded bg-stone-200 dark:bg-stone-800 shrink-0"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="font-mono text-xs font-semibold text-accent truncate">
                        {s.session_id}
                      </div>
                      <div className="text-xs text-stone-500 dark:text-stone-400 truncate">
                        {s.language} · {s.script_variant} · {s.source}
                      </div>
                      <div className="text-[11px] text-stone-400">
                        {s.line_count} lines · Updated {timeAgo(s.updated_at)}
                      </div>
                    </div>
                    <ArrowRight
                      size={16}
                      className="text-stone-300 group-hover:text-accent transition-colors shrink-0"
                    />
                  </a>
                </li>
              ))}
            </ul>
          )}
        </section>

        <aside>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-stone-500 mb-3">
            <Plus size={14} /> New Session
          </div>
          <form
            onSubmit={createSession}
            className="space-y-3 p-4 rounded-lg border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900"
          >
            <div>
              <label className="text-[10px] uppercase font-bold text-stone-400 mb-1 block">
                Language
              </label>
              <select
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                className="w-full px-2 py-2 text-xs rounded-lg border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 outline-none focus:ring-1 focus:ring-accent"
              >
                {languages.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.code} ({l.direction})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] uppercase font-bold text-stone-400 mb-1 block">
                Script Variant
              </label>
              <input
                value={script}
                onChange={(e) => setScript(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 outline-none focus:ring-1 focus:ring-accent"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase font-bold text-stone-400 mb-1 block">
                Fragment Path (absolute)
              </label>
              <input
                value={fragmentPath}
                onChange={(e) => setFragmentPath(e.target.value)}
                placeholder="/path/to/page.png"
                className="w-full px-3 py-2 text-xs rounded-lg border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 outline-none focus:ring-1 focus:ring-accent font-mono"
              />
              <p className="text-[10px] text-stone-400 mt-1 flex items-center gap-1">
                <Upload size={10} /> Absolute path to deskewed fragment PNG.
              </p>
            </div>
            {formError && (
              <div className="text-xs text-red-600 dark:text-red-400">{formError}</div>
            )}
            <button
              type="submit"
              disabled={creating || !fragmentPath}
              className="w-full py-2.5 rounded-lg font-medium text-sm bg-accent text-white hover:bg-accent/90 disabled:bg-stone-300 dark:disabled:bg-stone-800 disabled:text-stone-500 transition-all flex items-center justify-center gap-2"
            >
              {creating ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Plus size={16} />}
              {creating ? "Creating…" : "Create Session"}
            </button>
          </form>
          <div className="mt-4 flex items-center gap-2 text-xs text-stone-400">
            <FileText size={12} />
            Sessions auto-segment via Kraken BLLA on upload.
          </div>
        </aside>
      </main>
    </div>
  );
}