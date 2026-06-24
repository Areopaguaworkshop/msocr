import { useEffect, useState } from "react";
import type { PlateInfo } from "./types";

function statusBadge(status: PlateInfo["status"]): {
  label: string;
  color: string;
  dotClass: string;
} {
  if (!status) {
    return {
      label: "Not started",
      color: "text-stone-400",
      dotClass: "bg-stone-300 dark:bg-stone-600",
    };
  }
  if (status.transcribed_count === status.line_count) {
    return {
      label: "Complete",
      color: "text-emerald-600 dark:text-emerald-400",
      dotClass: "bg-emerald-500",
    };
  }
  if (status.transcribed_count === 0) {
    return {
      label: "In progress",
      color: "text-amber-600 dark:text-amber-400",
      dotClass: "bg-amber-500",
    };
  }
  return {
    label: `${status.transcribed_count}/${status.line_count}`,
    color: "text-amber-600 dark:text-amber-400",
    dotClass: "bg-amber-500",
  };
}

export default function PlateGallery() {
  const [plates, setPlates] = useState<PlateInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creatingPlate, setCreatingPlate] = useState<number | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const res = await fetch("/api/corpus/c2av/plates");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!disposed) setPlates(data);
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!disposed) setLoading(false);
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  async function openPlate(plate: PlateInfo) {
    setCreatingPlate(plate.plate_number);
    setCreateError(null);
    try {
      const res = await fetch(
        `/api/corpus/c2av/plates/${plate.filename}/session`,
        { method: "POST" },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      window.location.href = `/ui/${data.session_id}`;
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : String(e));
      setCreatingPlate(null);
    }
  }

  const totalPlates = plates.length;
  const platesWithSessions = plates.filter((p) => p.status).length;
  const totalTranscribed = plates.reduce(
    (sum, p) => sum + (p.status?.transcribed_count ?? 0),
    0,
  );

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 text-stone-900 dark:text-stone-100">
      <header className="h-14 flex items-center justify-between px-6 border-b border-stone-200 dark:border-stone-800">
        <h1 className="text-sm font-medium uppercase tracking-wider text-stone-600 dark:text-stone-400">
          msocr · Christian Sogdian C2AV
        </h1>
        <a
          href="/sessions"
          className="text-xs text-stone-500 hover:text-accent transition-colors"
        >
          Sessions
        </a>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {!loading && plates.length > 0 && (
          <div className="flex gap-6 text-xs text-stone-500 mb-6 pb-4 border-b border-stone-200 dark:border-stone-800">
            <span>{totalPlates} plates</span>
            <span>{platesWithSessions} with sessions</span>
            <span>{totalTranscribed} lines transcribed</span>
          </div>
        )}

        {createError && (
          <div className="mb-4 text-sm text-red-600 dark:text-red-400">
            {createError}
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {Array.from({ length: 12 }).map((_, i) => (
              <div
                key={i}
                className="rounded-lg border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 overflow-hidden animate-pulse"
              >
                <div className="aspect-[3/4] bg-stone-200 dark:bg-stone-800" />
                <div className="p-3 space-y-2">
                  <div className="h-3 w-16 bg-stone-200 dark:bg-stone-800 rounded" />
                  <div className="h-2.5 w-20 bg-stone-100 dark:bg-stone-800 rounded" />
                </div>
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="text-sm text-red-600 dark:text-red-400">{error}</div>
        ) : plates.length === 0 ? (
          <div className="text-sm text-stone-400 italic">
            No plates found. Run PDF extraction first.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {plates.map((plate) => {
              const badge = statusBadge(plate.status);
              const isCreating = creatingPlate === plate.plate_number;
              return (
                <button
                  key={plate.plate_number}
                  onClick={() => openPlate(plate)}
                  disabled={isCreating}
                  className="text-left rounded-lg border border-stone-200 dark:border-stone-800 bg-white dark:bg-stone-900 hover:border-accent hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors overflow-hidden group"
                >
                  <div className="aspect-[3/4] bg-stone-100 dark:bg-stone-900 flex items-center justify-center">
                    {isCreating ? (
                      <div className="w-6 h-6 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                    ) : (
                      <img
                        src={plate.thumbnail_url}
                        alt={`Plate ${plate.plate_number}`}
                        className="w-full h-full object-contain"
                        loading="lazy"
                      />
                    )}
                  </div>
                  <div className="p-3">
                    <div className="font-mono text-xs font-semibold text-accent">
                      Plate {plate.plate_number}
                    </div>
                    <div className="flex items-center gap-1.5 mt-1">
                      <span
                        className={`w-2 h-2 rounded-full shrink-0 ${badge.dotClass}`}
                      />
                      <span className={`text-[11px] ${badge.color}`}>
                        {badge.label}
                      </span>
                    </div>
                    {plate.status && (
                      <div className="text-[10px] text-stone-400 mt-0.5">
                        {plate.status.line_count} lines
                      </div>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
