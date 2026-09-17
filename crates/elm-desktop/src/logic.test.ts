import { describe, expect, it } from "vitest";
import { filterJobs, findDuplicateMaskColumns, isActiveJobState, parseRelation } from "./logic";
import type { JobRecord, JobState, MaskRule } from "./types";

function job(id: string, name: string | undefined, state: JobState): JobRecord {
  return {
    spec: {
      version: 1,
      id,
      name,
      source: { kind: "file", path: "in.csv", format: "csv" },
      sink: { kind: "file", path: "out.csv", format: "csv" },
      write_mode: "append",
      consistency: "atomic",
      memory_budget_bytes: 512 * 1024 * 1024,
      batch_target_bytes: 16 * 1024 * 1024,
      masks: [],
      random_seed: [],
      conversions: [],
      submitted_at: "2026-09-17T00:00:00Z",
    },
    progress: {
      job_id: id,
      state,
      rows: 0,
      bytes: 0,
      batches: 0,
      rows_per_second: 0,
      bytes_per_second: 0,
      elapsed: { secs: 0, nanos: 0 },
      warnings: [],
      occurred_at: "2026-09-17T00:00:00Z",
    },
    attempt: 1,
    updated_at: "2026-09-17T00:00:00Z",
  };
}

function mask(id: string, column: string): MaskRule {
  return { id, name: id, column, algorithm: "star" };
}

describe("parseRelation", () => {
  it("accepts a bare name", () => {
    expect(parseRelation("customers")).toEqual({ name: "customers" });
  });

  it("accepts schema.name", () => {
    expect(parseRelation("public.customers")).toEqual({ schema: "public", name: "customers" });
  });

  it("accepts catalog.schema.name", () => {
    expect(parseRelation("db.public.customers")).toEqual({
      catalog: "db",
      schema: "public",
      name: "customers",
    });
  });

  it("trims whitespace around components", () => {
    expect(parseRelation(" public . customers ")).toEqual({ schema: "public", name: "customers" });
  });

  it("rejects an empty component", () => {
    expect(() => parseRelation("public..customers")).toThrow(/non-empty/);
  });

  it("rejects more than three components", () => {
    expect(() => parseRelation("a.b.c.d")).toThrow(/name, schema\.name, or catalog\.schema\.name/);
  });

  it("rejects an empty string", () => {
    expect(() => parseRelation("")).toThrow(/non-empty/);
  });
});

describe("filterJobs", () => {
  const jobs: JobRecord[] = [
    job("11111111-aaaa", "nightly export", "succeeded"),
    job("22222222-bbbb", "customer backfill", "failed"),
    job("33333333-cccc", undefined, "running"),
  ];

  it("returns every job when the query and state filter are empty", () => {
    expect(filterJobs(jobs, "", "")).toHaveLength(3);
  });

  it("matches by name, case-insensitively", () => {
    expect(filterJobs(jobs, "NIGHTLY", "")).toEqual([jobs[0]]);
  });

  it("matches by job id substring", () => {
    expect(filterJobs(jobs, "bbbb", "")).toEqual([jobs[1]]);
  });

  it("falls back to id matching when a job has no name", () => {
    expect(filterJobs(jobs, "cccc", "")).toEqual([jobs[2]]);
  });

  it("filters by state alone", () => {
    expect(filterJobs(jobs, "", "failed")).toEqual([jobs[1]]);
  });

  it("combines a text query and a state filter", () => {
    expect(filterJobs(jobs, "customer", "failed")).toEqual([jobs[1]]);
    expect(filterJobs(jobs, "customer", "succeeded")).toEqual([]);
  });

  it("returns nothing when no job matches", () => {
    expect(filterJobs(jobs, "does-not-exist", "")).toEqual([]);
  });
});

describe("isActiveJobState", () => {
  it("treats queued, preflighting, running, publishing, and cancelling as active", () => {
    const active: JobState[] = ["queued", "preflighting", "running", "publishing", "cancelling"];
    for (const state of active) expect(isActiveJobState(state)).toBe(true);
  });

  it("treats every terminal state as not active", () => {
    const terminal: JobState[] = ["succeeded", "failed", "cancelled", "interrupted"];
    for (const state of terminal) expect(isActiveJobState(state)).toBe(false);
  });
});

describe("findDuplicateMaskColumns", () => {
  it("reports nothing when no rules are selected", () => {
    const masks = [mask("a", "email"), mask("b", "email")];
    expect(findDuplicateMaskColumns(masks, new Set())).toEqual([]);
  });

  it("reports nothing when selected rules target distinct columns", () => {
    const masks = [mask("a", "email"), mask("b", "phone")];
    expect(findDuplicateMaskColumns(masks, new Set(["a", "b"]))).toEqual([]);
  });

  it("reports a column with two selected rules", () => {
    const masks = [mask("a", "email"), mask("b", "email")];
    expect(findDuplicateMaskColumns(masks, new Set(["a", "b"]))).toEqual(["email"]);
  });

  it("ignores an unselected duplicate", () => {
    const masks = [mask("a", "email"), mask("b", "email")];
    expect(findDuplicateMaskColumns(masks, new Set(["a"]))).toEqual([]);
  });

  it("reports every conflicting column when several collide", () => {
    const masks = [mask("a", "email"), mask("b", "email"), mask("c", "phone"), mask("d", "phone")];
    expect(findDuplicateMaskColumns(masks, new Set(["a", "b", "c", "d"]))).toEqual(["email", "phone"]);
  });
});
