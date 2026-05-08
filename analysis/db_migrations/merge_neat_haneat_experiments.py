"""
Merge NEAT runs from multi_task_neat_vs_haneat_final (exp 9)
and ALL HA-NEAT runs from multi_task_haneat_finalv2 (exp 11)
into a new experiment: multi_task_final_combined

Rules:
- From exp 9 (final): only NEAT runs, only FINISHED status
- From exp 11 (finalv2): all HA-NEAT runs (all FINISHED)
- Skip any RUNNING runs
- Copy all metrics, latest_metrics, params, tags, inputs, input_tags
- artifact_uri: keep pointing to original path (no file copying needed)
"""

import sqlite3
import uuid
import time

DB_PATH = "/home/20240503/tensorneat/mlflow.db"
NEW_EXP_NAME = "multi_task_final_combined"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- 1. Identify runs to migrate ---

    # NEAT FINISHED runs from exp 9
    cur.execute("""
        SELECT r.run_uuid FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = 'multi_task_neat_vs_haneat_final'
          AND r.name LIKE 'neat_%'
          AND r.status = 'FINISHED'
    """)
    neat_runs = [row['run_uuid'] for row in cur.fetchall()]

    # HA-NEAT FINISHED runs from exp 11
    cur.execute("""
        SELECT r.run_uuid FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = 'multi_task_haneat_finalv2'
          AND r.status = 'FINISHED'
    """)
    haneat_runs = [row['run_uuid'] for row in cur.fetchall()]

    print(f"NEAT runs to migrate:    {len(neat_runs)}")
    print(f"HA-NEAT runs to migrate: {len(haneat_runs)}")
    for uuid_ in neat_runs:
        cur.execute("SELECT name FROM runs WHERE run_uuid=?", (uuid_,))
        print(f"  NEAT   | {cur.fetchone()['name']}")
    for uuid_ in haneat_runs:
        cur.execute("SELECT name FROM runs WHERE run_uuid=?", (uuid_,))
        print(f"  HANEAT | {cur.fetchone()['name']}")

    all_source_runs = neat_runs + haneat_runs
    print(f"\nTotal runs to migrate: {len(all_source_runs)}")

    # --- 2. Create new experiment ---
    cur.execute("SELECT experiment_id FROM experiments WHERE name=?", (NEW_EXP_NAME,))
    existing = cur.fetchone()
    if existing:
        new_exp_id = existing['experiment_id']
        print(f"\nExperiment '{NEW_EXP_NAME}' already exists (id={new_exp_id}), reusing.")
    else:
        # Find next experiment_id
        cur.execute("SELECT MAX(CAST(experiment_id AS INTEGER)) FROM experiments")
        max_id = cur.fetchone()[0] or 0
        new_exp_id = str(int(max_id) + 1)
        artifact_loc = f"/home/20240503/tensorneat/mlruns/{new_exp_id}"
        cur.execute("""
            INSERT INTO experiments (experiment_id, name, artifact_location, lifecycle_stage)
            VALUES (?, ?, ?, 'active')
        """, (new_exp_id, NEW_EXP_NAME, artifact_loc))
        print(f"\nCreated experiment '{NEW_EXP_NAME}' with id={new_exp_id}")

    # --- 3. Copy each run and its associated data ---
    tables_with_run_uuid = ['metrics', 'latest_metrics', 'params', 'tags', 'inputs', 'input_tags']

    for source_run_uuid in all_source_runs:
        # Get run row
        cur.execute("SELECT * FROM runs WHERE run_uuid=?", (source_run_uuid,))
        run_row = cur.fetchone()
        run_name = run_row['name']

        # Check if already migrated (same name in new experiment)
        cur.execute("""
            SELECT run_uuid FROM runs
            WHERE experiment_id=? AND name=?
        """, (new_exp_id, run_name))
        if cur.fetchone():
            print(f"  SKIP (already exists): {run_name}")
            continue

        # Generate new run_uuid
        new_run_uuid = uuid.uuid4().hex

        # Build new artifact_uri pointing to new experiment folder
        # but keep original run uuid subfolder for traceability
        new_artifact_uri = f"/home/20240503/tensorneat/mlruns/{new_exp_id}/{new_run_uuid}/artifacts"

        # Insert new run
        cur.execute("""
            INSERT INTO runs (
                run_uuid, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version,
                lifecycle_stage, artifact_uri, experiment_id, deleted_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_run_uuid,
            run_row['name'],
            run_row['source_type'],
            run_row['source_name'],
            run_row['entry_point_name'],
            run_row['user_id'],
            run_row['status'],
            run_row['start_time'],
            run_row['end_time'],
            run_row['source_version'],
            run_row['lifecycle_stage'],
            new_artifact_uri,
            new_exp_id,
            run_row['deleted_time'],
        ))

        # Copy metrics
        cur.execute("SELECT * FROM metrics WHERE run_uuid=?", (source_run_uuid,))
        metrics = cur.fetchall()
        cur.executemany("""
            INSERT INTO metrics (run_uuid, key, value, timestamp, step, is_nan)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [(new_run_uuid, r['key'], r['value'], r['timestamp'], r['step'], r['is_nan']) for r in metrics])

        # Copy latest_metrics
        cur.execute("SELECT * FROM latest_metrics WHERE run_uuid=?", (source_run_uuid,))
        lm = cur.fetchall()
        cur.executemany("""
            INSERT INTO latest_metrics (run_uuid, key, value, timestamp, step, is_nan)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [(new_run_uuid, r['key'], r['value'], r['timestamp'], r['step'], r['is_nan']) for r in lm])

        # Copy params
        cur.execute("SELECT * FROM params WHERE run_uuid=?", (source_run_uuid,))
        params = cur.fetchall()
        cur.executemany("""
            INSERT INTO params (run_uuid, key, value)
            VALUES (?, ?, ?)
        """, [(new_run_uuid, r['key'], r['value']) for r in params])

        # Copy tags
        cur.execute("SELECT * FROM tags WHERE run_uuid=?", (source_run_uuid,))
        tags = cur.fetchall()
        cur.executemany("""
            INSERT INTO tags (run_uuid, key, value)
            VALUES (?, ?, ?)
        """, [(new_run_uuid, r['key'], r['value']) for r in tags])

        # Copy inputs (if any)
        cur.execute("SELECT * FROM inputs WHERE destination_id=?", (source_run_uuid,))
        inputs_rows = cur.fetchall()
        if inputs_rows:
            cols = inputs_rows[0].keys()
            for inp in inputs_rows:
                new_vals = [new_run_uuid if v == source_run_uuid else v for v in inp]
                cur.execute(f"INSERT INTO inputs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", new_vals)

        print(f"  COPIED: {run_name} -> new_uuid={new_run_uuid[:8]}")

    conn.commit()
    conn.close()

    print(f"\nDone. Verify with:")
    print(f"  sqlite3 {DB_PATH} \"SELECT name, status FROM runs WHERE experiment_id={new_exp_id};\"")

if __name__ == "__main__":
    main()
