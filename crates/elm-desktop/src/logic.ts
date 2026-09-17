import type { JobRecord, JobState, MaskRule, Relation } from "./types";

export function parseRelation(value: string): Relation {
  const parts = value.split(".").map((part) => part.trim());
  if (parts.some((part) => !part)) throw new Error("Relation must use non-empty name components.");
  if (parts.length === 1) return { name: parts[0] };
  if (parts.length === 2) return { schema: parts[0], name: parts[1] };
  if (parts.length === 3) return { catalog: parts[0], schema: parts[1], name: parts[2] };
  throw new Error("Relation must be name, schema.name, or catalog.schema.name.");
}

/** Backs the Jobs screen's "Filter by name or id" input and state dropdown. */
export function filterJobs(jobs: JobRecord[], query: string, stateFilter: string | JobState): JobRecord[] {
  const search = query.trim().toLocaleLowerCase();
  return jobs.filter((job) => {
    const matchesSearch =
      !search ||
      job.spec.id.toLocaleLowerCase().includes(search) ||
      (job.spec.name ?? "").toLocaleLowerCase().includes(search);
    return matchesSearch && (!stateFilter || job.progress.state === stateFilter);
  });
}

/**
 * `apply_masks` (crates/elm-core/src/masking.rs) overwrites a column once per matching rule, so
 * the last selected rule for a column silently wins; the wizard must surface that ambiguity
 * instead of submitting it. Returns the columns targeted by more than one selected rule.
 */
export function findDuplicateMaskColumns(masks: MaskRule[], selectedMaskIds: Set<string>): string[] {
  const counts = new Map<string, number>();
  for (const rule of masks) {
    if (!selectedMaskIds.has(rule.id)) continue;
    counts.set(rule.column, (counts.get(rule.column) ?? 0) + 1);
  }
  return [...counts.entries()].filter(([, count]) => count > 1).map(([column]) => column);
}
