import psycopg2
import os
import time

NEW_DB_NAME = os.environ.get("NEW_DB_NAME")
NEW_DB_USER = os.environ.get("NEW_DB_USER")
NEW_DB_PASSWORD = os.environ.get("NEW_DB_PASSWORD")

DB_HOST = "timescaledb"
POSTGRES_NAME = "postgres"
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")


# Create database
def execute_sql(sql, autocommit=False, role_exists_ok=False, db_exists_ok=False):
    retries = 10
    retry_delay = 5

    for i in range(retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=POSTGRES_NAME,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            if autocommit:
                conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            if sql.startswith("CREATE USER") and not role_exists_ok:
                cur.execute(f"SELECT 1 FROM pg_roles WHERE rolname='{NEW_DB_USER}'")
                if cur.fetchone():
                    print(f"User {NEW_DB_USER} already exists, skipping creation.")
                    return
            if sql.startswith("CREATE DATABASE") and not db_exists_ok:
                cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{NEW_DB_NAME}'")
                if cur.fetchone():
                    print(f"Database {NEW_DB_NAME} already exists, skipping creation.")
                    return
            cur.execute(sql)
            if not autocommit:
                conn.commit()
            print(f"SQL executed successfully: {sql}")
            return
        except psycopg2.Error as e:
            print(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            if i < retries - 1:
                print(f"Retrying ({i + 1}/{retries})...")
                time.sleep(retry_delay)
            else:
                raise
        finally:
            if conn:
                cur.close()
                conn.close()

if __name__ == "__main__":
    execute_sql(f"CREATE DATABASE {NEW_DB_NAME};", autocommit=True)
    execute_sql(f"CREATE USER {NEW_DB_USER} WITH PASSWORD '{NEW_DB_PASSWORD}';")
    execute_sql(f"GRANT ALL PRIVILEGES ON DATABASE {NEW_DB_NAME} TO {NEW_DB_USER};")