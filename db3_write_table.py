import psycopg2
from eodhd import APIClient
import os


FROM_DATE = '2000-01-01'
TO_DATE = '2025-01-01'


API_KEY = os.environ.get("API_KEY")
columns_api = ("date", "open", "high", "low", "close", "adjusted_close", "volume")


DB_HOST = "timescaledb"
NEW_DB_NAME = os.environ.get("NEW_DB_NAME")
NEW_DB_USER = os.environ.get("NEW_DB_USER")
NEW_DB_PASSWORD = os.environ.get("NEW_DB_PASSWORD")

# Update  stock & crypto item by item.

def insert_data(conn, cursor):
    api = APIClient(API_KEY)
    resp = api.get_eod_historical_stock_market_data(symbol=SYMBOL, period='d', from_date=FROM_DATE, to_date=TO_DATE, order='a')
    new_data_inserted = False

    for data in resp:
        date = data['date']
        cursor.execute(f"SELECT 1 FROM {TABLE_NAME} WHERE date = %s", (date,))
        exists = cursor.fetchone()

        if not exists:
            placeholders = "(" + ", ".join(["%s"] * len(columns_api)) + ")"
            values = tuple(data[c] for c in columns_api)
            cursor.execute(
                f"INSERT INTO {TABLE_NAME} ({', '.join(columns_api)}) VALUES {placeholders}",
                values
            )

            print(f"API Insert new data：{data['date']}")
            new_data_inserted = True

    conn.commit()
    if new_data_inserted:
        print(f"API Data successfully inserted into table，{TABLE_NAME} Contains new data")
    else:
        print(f"API Data checked，{TABLE_NAME} No new data inserted")

if __name__ == "__main__":
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=NEW_DB_NAME, user=NEW_DB_USER, password=NEW_DB_PASSWORD)
        cur = conn.cursor()

        cur.execute("""
                           SELECT table_schema, table_name
                           FROM information_schema.tables
                           WHERE table_schema NOT IN ('_timescaledb_catalog', '_timescaledb_internal', '_timescaledb_config','_timescaledb_cache','pg_catalog', 'information_schema') AND table_type = 'BASE TABLE' OR table_type = 'FOREIGN TABLE';
                       """)

        finance_tables = cur.fetchall()
        print("\nWrite all tables in the database:")
        for schema, table in finance_tables:
            print(f"- {schema}.{table}")

        for schema, table in finance_tables:
            TABLE_NAME = schema + "." + table

            if schema == "stock":
                SYMBOL = table + '.us'
            elif schema == "crypto":
                SYMBOL = table + '-usd.cc'
            else:
                print(f"The data source for schema {schema} is unknown, skipping processing.")
                continue

            insert_data(conn, cur)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()