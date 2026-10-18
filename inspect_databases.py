
import sqlite3

databases = [
    "mlflow.db",
    "feature_repo/data/online_store.db",
    "feature_repo/data/registry.db",
]

for path in databases:
    print("\n" + "=" * 60)
    print("DATABASE:", path)

    try:
        conn = sqlite3.connect(
            "file:" + path + "?mode=ro", uri=True
        )
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        ).fetchall()

        if not tables:
            print("No tables found.")

        for (table,) in tables:
            print("\nTABLE:", table)
            safe_name = '"' + table.replace('"', '""') + '"'

            columns = conn.execute(
                "PRAGMA table_info(" + safe_name + ")"
            ).fetchall()
            print("COLUMNS:", [col[1] for col in columns])

            rows = conn.execute(
                "SELECT * FROM " + safe_name + " LIMIT 5"
            ).fetchall()

            for row in rows:
                print(row)

        conn.close()

    except Exception as error:
        print("ERROR:", error)
