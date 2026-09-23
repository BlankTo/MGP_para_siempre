import hashlib
import json
import sqlite3

from .diagnostics import DIAGNOSTIC_COLUMNS

from .diversity import WORKSPACE_CLASSES


RUN_COLUMNS = (
    "run_id", "target", "configuration_id", "repetition",
    "merge_count", "internal_merge_count", "external_merge_count", "internal_fallback_count",
    "termination_reason", "final_type", "final_workspace_size",
    "best_equation", "best_fitness", "final_equation", "final_fitness",
    "first_target_step", "success", "completed", "subtarget_count", "archive_size", "duration_seconds",
)

TRACE_COLUMNS = ("merge_kind", "workspace_structure", "workspace_evaluated", "best_fitness")


def step_metrics():
    names = list(WORKSPACE_CLASSES) + ["workspace_size"]
    names += ["fraction_" + name for name in WORKSPACE_CLASSES]
    names += [
        "semantic_count", "nonfinite_semantic_count", "unique_semantic_count",
        "pair_count", "valid_pair_count", "undefined_pair_count", "mean_comparison",
    ]

    return names


def manifest_fingerprint(manifest):
    text = json.dumps(manifest, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class StudyStore:
    def __init__(self, path, bins):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.metrics = step_metrics()
        self.histogram_columns = [f"bin_{index:03d}_pairs" for index in range(bins)]

        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS configurations (
                configuration_id INTEGER PRIMARY KEY,
                esel REAL NOT NULL,
                esur REAL NOT NULL,
                variable_atom_count INTEGER NOT NULL,
                operator_count INTEGER NOT NULL,
                parameters TEXT NOT NULL,
                UNIQUE (esel, esur, variable_atom_count, operator_count)
            );
            CREATE TABLE IF NOT EXISTS seeds (
                repetition INTEGER PRIMARY KEY,
                seed INTEGER NOT NULL UNIQUE,
                workspace_seed INTEGER NOT NULL,
                derivation_seed INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS subtargets (
                target TEXT NOT NULL,
                subtarget_id INTEGER NOT NULL,
                expression TEXT NOT NULL,
                PRIMARY KEY (target, subtarget_id)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY,
                target TEXT NOT NULL,
                configuration_id INTEGER NOT NULL REFERENCES configurations,
                repetition INTEGER NOT NULL REFERENCES seeds,
                merge_count INTEGER NOT NULL,
                internal_merge_count INTEGER NOT NULL,
                external_merge_count INTEGER NOT NULL,
                internal_fallback_count INTEGER NOT NULL,
                termination_reason TEXT NOT NULL,
                final_type TEXT NOT NULL,
                final_workspace_size INTEGER NOT NULL,
                best_equation TEXT,
                best_fitness REAL,
                final_equation TEXT,
                final_fitness REAL,
                first_target_step INTEGER,
                success INTEGER NOT NULL,
                completed INTEGER NOT NULL CHECK (completed = 1),
                subtarget_count INTEGER NOT NULL,
                archive_size INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                UNIQUE (target, configuration_id, repetition)
            );
            CREATE TABLE IF NOT EXISTS subtarget_hits (
                run_id INTEGER NOT NULL REFERENCES runs,
                subtarget_id INTEGER NOT NULL,
                first_step INTEGER NOT NULL,
                PRIMARY KEY (run_id, subtarget_id)
            );
            CREATE TABLE IF NOT EXISTS archive_subtarget_hits (
                run_id INTEGER NOT NULL REFERENCES runs,
                subtarget_id INTEGER NOT NULL,
                first_step INTEGER NOT NULL,
                PRIMARY KEY (run_id, subtarget_id)
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_run_id INTEGER NOT NULL,
                manifest_sha256 TEXT
            );
            INSERT OR IGNORE INTO progress VALUES (1, 1, NULL);
            CREATE VIEW IF NOT EXISTS completed_runs AS
                SELECT r.*, c.esel, c.esur, c.variable_atom_count, c.operator_count,
                       s.seed, s.workspace_seed, s.derivation_seed
                FROM runs r JOIN configurations c USING (configuration_id)
                            JOIN seeds s USING (repetition);
        """)

        definitions = []

        for name in DIAGNOSTIC_COLUMNS:
            kind = "REAL" if name in ("best_probe_rmse", "diagnostic_seconds") else "INTEGER"
            definitions.append(name + " " + kind)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS run_diagnostics (run_id INTEGER PRIMARY KEY REFERENCES runs, "
            + ", ".join(definitions) + ")"
        )

        total_columns = ["contributors"]
        definitions = ["contributors INTEGER NOT NULL"]
        updates = ["contributors = contributors + 1"]

        for name in self.metrics:
            count, mean, squared = name + "_count", name + "_mean", name + "_m2"
            total_columns += [count, mean, squared]
            definitions += [count + " INTEGER NOT NULL", mean + " REAL NOT NULL", squared + " REAL NOT NULL"]
            new_mean = mean + " + (excluded." + mean + " - " + mean + ") / (" + count + " + 1)"
            updates += [
                count + " = " + count + " + excluded." + count,
                mean + " = CASE WHEN excluded." + count + " = 0 THEN " + mean + " ELSE " + new_mean + " END",
                squared + " = CASE WHEN excluded." + count + " = 0 THEN " + squared
                + " ELSE " + squared + " + (excluded." + mean + " - " + mean + ") * (excluded."
                + mean + " - (" + new_mean + ")) END",
            ]

        for name in self.histogram_columns:
            total_columns.append(name)
            definitions.append(name + " INTEGER NOT NULL")
            updates.append(name + " = " + name + " + excluded." + name)

        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS step_totals ("
            "target TEXT NOT NULL, configuration_id INTEGER NOT NULL, step INTEGER NOT NULL, "
            + ", ".join(definitions) + ", PRIMARY KEY (target, configuration_id, step))"
        )
        columns = ["target", "configuration_id", "step"] + total_columns
        placeholders = ", ".join("?" for name in columns)
        self.total_insert = (
            "INSERT INTO step_totals (" + ", ".join(columns) + ") VALUES (" + placeholders + ") "
            "ON CONFLICT (target, configuration_id, step) DO UPDATE SET " + ", ".join(updates)
        )

        raw_columns = self.metrics + self.histogram_columns
        definitions = ['"' + name + '" REAL' for name in raw_columns]
        definitions += ["merge_kind TEXT", "workspace_structure TEXT", "workspace_evaluated TEXT", "best_fitness REAL"]
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS run_steps ("
            "run_id INTEGER NOT NULL REFERENCES runs, step INTEGER NOT NULL, "
            + ", ".join(definitions) + ", PRIMARY KEY (run_id, step))"
        )
        self.connection.commit()

    def initialize(self, combinations, seeds, subtargets, manifest):
        with self.connection:
            for number, parameters in enumerate(combinations, 1):
                current = {**manifest["configuration"], **parameters}
                row = [number, current["esel"], current["esur"], current["variable_atom_count"],
                       current["operator_count"], json.dumps(parameters)]
                self.connection.execute("INSERT INTO configurations VALUES (?, ?, ?, ?, ?, ?)", row)

            self.connection.executemany("INSERT INTO seeds VALUES (?, ?, ?, ?)", seeds)

            for target, entries in subtargets.items():
                rows = [(target, identifier, expression) for identifier, expression, values in entries]
                self.connection.executemany("INSERT INTO subtargets VALUES (?, ?, ?)", rows)

            self.connection.execute("UPDATE progress SET manifest_sha256 = ? WHERE id = 1", (manifest_fingerprint(manifest),))

    def check_manifest(self, manifest):
        saved = self.connection.execute("SELECT manifest_sha256 FROM progress WHERE id = 1").fetchone()[0]

        if saved != manifest_fingerprint(manifest):
            self.close()
            raise ValueError("The saved study configuration or metadata has changed.")

    def next_run_id(self):
        return self.connection.execute("SELECT next_run_id FROM progress WHERE id = 1").fetchone()[0]

    def is_completed(self, run_id):
        row = self.connection.execute("SELECT completed FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row is not None and row[0] == 1

    def save_run(self, run, steps, subtarget_steps, save_trace=False,
                 diagnostics=None, archive_subtarget_steps=None):
        totals = []
        raw_steps = []

        for step, metrics in enumerate(steps):
            row = [run["target"], run["configuration_id"], step, 1]
            raw = [run["run_id"], step]

            for name in self.metrics:
                value = metrics[name]
                raw.append(value)
                row.extend((0, 0, 0) if value is None else (1, value, 0))

            for name in self.histogram_columns:
                row.append(metrics[name])
                raw.append(metrics[name])

            totals.append(row)

            if save_trace:
                raw.extend(metrics[name] for name in TRACE_COLUMNS)
                raw_steps.append(raw)

        placeholders = ", ".join("?" for name in RUN_COLUMNS)
        run_values = [run[name] for name in RUN_COLUMNS]
        hits = [(run["run_id"], identifier, step) for identifier, step in subtarget_steps.items()]

        with self.connection:
            self.connection.execute("INSERT INTO runs VALUES (" + placeholders + ")", run_values)
            self.connection.executemany(self.total_insert, totals)
            self.connection.executemany("INSERT INTO subtarget_hits VALUES (?, ?, ?)", hits)

            if diagnostics is not None:
                values = [run["run_id"]] + [diagnostics[name] for name in DIAGNOSTIC_COLUMNS]
                placeholders = ", ".join("?" for value in values)
                self.connection.execute("INSERT INTO run_diagnostics VALUES (" + placeholders + ")", values)

            if archive_subtarget_steps:
                hits = [(run["run_id"], identifier, step) for identifier, step in archive_subtarget_steps.items()]
                self.connection.executemany("INSERT INTO archive_subtarget_hits VALUES (?, ?, ?)", hits)

            if raw_steps:
                placeholders = ", ".join("?" for value in raw_steps[0])
                self.connection.executemany("INSERT INTO run_steps VALUES (" + placeholders + ")", raw_steps)

            self.connection.execute("UPDATE progress SET next_run_id = ? WHERE id = 1", (run["run_id"] + 1,))

    def close(self):
        self.connection.close()
