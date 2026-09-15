"""Benchmark the installed distribution, without changing it or saved user settings."""
import contextlib
import datetime
import importlib.metadata
import json
import os
import sys
import tempfile
import time


def main():
    if os.environ.get("ELM_PERF_DISPOSABLE") != "yes":
        raise RuntimeError("requires a fresh disposable elm_perf PostgreSQL fixture")
    with tempfile.TemporaryDirectory(prefix="elm-installed-perf-") as settings:
        os.environ["ELM_TOOL_HOME"] = settings
        # Import only after isolating configuration. -I selects the installed package.
        import psycopg2
        from elm.core import copy, environment

        with contextlib.redirect_stdout(sys.stderr):
            result = environment.create_environment(
                name="perf", host="127.0.0.1", port=55439, user="postgres",
                password="disposable-trust-only", service="elm_perf", database="POSTGRES",
            )
        if not result.success:
            raise RuntimeError("disposable environment creation failed")
        connection = psycopg2.connect(host="127.0.0.1", port=55439, user="postgres", dbname="elm_perf", sslmode="disable")
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("CREATE TABLE perf_source (id bigint PRIMARY KEY, payload text NOT NULL)")
                cursor.execute("INSERT INTO perf_source SELECT i, repeat(md5(i::text),32) FROM generate_series(1,250000) i")
                cursor.execute("ANALYZE perf_source")
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
            runs = []
            for trial in range(1, 4):
                started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    result = copy.copy_db_to_db(
                        source_env="perf", target_env="perf", query="SELECT id, payload FROM public.perf_source",
                        table=f"perf_target_{trial}", mode="REPLACE", batch_size=1000,
                        parallel_workers=1, apply_masks=False, verbose_batch_logs=False,
                    )
                seconds = time.perf_counter() - started
                if not result.success:
                    raise RuntimeError("installed elm-tool transfer failed")
                with connection.cursor() as cursor:
                    cursor.execute(f"""SELECT count(*), count(DISTINCT id),
                        count(*) FILTER (WHERE id IS NULL OR id < 1 OR id > 250000 OR payload IS NULL
                        OR payload <> repeat(md5(id::text),32)) FROM perf_target_{trial}""")
                    if cursor.fetchone() != (250000, 250000, 0):
                        raise RuntimeError("copied data failed full validation")
                    cursor.execute(f"DROP TABLE perf_target_{trial}")
                runs.append(dict(trial=trial, seconds=seconds, rows=250000, logical_bytes=258000000,
                                 rows_per_second=250000/seconds,
                                 logical_mib_per_second=258000000/1048576/seconds, verified=True))
                print(f"Completed trial {trial}: {seconds:.3f}s, verified all rows", file=sys.stderr, flush=True)
            print(json.dumps(dict(
                captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                installed_version=importlib.metadata.version("elm-tool"), python_version=sys.version,
                package_location=copy.__file__, database_version=version, runs=runs,
                batch_size=1000, parallel_workers=1, masking=False,
                scope="Installed unchanged elm.core.copy.copy_db_to_db; CLI batch defaults; REPLACE into new targets; native history included",
                dependencies={name: importlib.metadata.version(name) for name in ["pandas", "sqlalchemy", "psycopg2-binary"]},
                notes="Same 258MB dataset, sequential warm-cache trials; setup/verification excluded; direct publication unlike Rust atomic staging; no 10GB gate claim",
            ), indent=2))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
