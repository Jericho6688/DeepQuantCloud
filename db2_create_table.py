import psycopg2
import os
import yaml

# 数据库配置
DB_HOST = "timescaledb"
NEW_DB_NAME = os.environ.get("NEW_DB_NAME")
NEW_DB_USER = os.environ.get("NEW_DB_USER")
NEW_DB_PASSWORD = os.environ.get("NEW_DB_PASSWORD")

# 表的定义
column_definitions = """
    date DATE PRIMARY KEY,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adjusted_close DOUBLE PRECISION,
    volume BIGINT
"""
hyper_index = "date"
interval = "1 week"

# 从 YAML 文件读取表集合
def load_tables_from_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['tables']  # 假设 YAML 文件的根元素是 'tables'

# 表的集合 (从 YAML 文件加载)
tables_to_create = load_tables_from_yaml('db2_create_table.yaml')  # 指定 YAML 文件路径  <--- 这里修改了文件名


if __name__ == "__main__":
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=NEW_DB_NAME, user=NEW_DB_USER, password=NEW_DB_PASSWORD)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        for table_config in tables_to_create:
            schema_name = table_config['schema']
            table_name = table_config['table']
            full_table_name = f"{schema_name}.{table_name}"

            # 创建 schema (如果不存在)
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")

            # 创建表 (如果不存在)
            cur.execute(f"CREATE TABLE IF NOT EXISTS {full_table_name} ({column_definitions});")

            # 创建 hypertable
            try:  # 添加try except 避免已经创建的表重复创建hypertable
                cur.execute(f"SELECT create_hypertable('{full_table_name}', '{hyper_index}', if_not_exists => TRUE);")
            except psycopg2.Error as e:
                print(f"Hypertable creation error for {full_table_name}: {e}")

            # 设置 chunk 时间间隔
            cur.execute(f"SELECT set_chunk_time_interval('{full_table_name}', INTERVAL '{interval}');")

            print(f"Schema and hypertable '{full_table_name}' created successfully!")

        # # 获取和打印所有表的信息
        # cur.execute("""
        #     SELECT format('%I.%I', n.nspname, c.relname), pg_relation_filepath(format('%I.%I', n.nspname, c.relname))
        #     FROM pg_class c
        #     JOIN pg_namespace n ON n.oid = c.relnamespace
        #     WHERE c.relkind = 'r';
        # """)
        # all_tables_info = cur.fetchall()
        # print("\nAll tables in the database:")
        # for table_name, table_path in all_tables_info:
        #     print(f"- Table Name: {table_name}, Path: {table_path}")

        # 获取和打印所有表
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('_timescaledb_catalog', '_timescaledb_internal', '_timescaledb_config','_timescaledb_cache','pg_catalog', 'information_schema') AND table_type = 'BASE TABLE' OR table_type = 'FOREIGN TABLE';
        """)
        tables = cur.fetchall()
        print("\nAll tables in the database:")
        for schema, table in tables:
            print(f"- {schema}.{table}")

    except psycopg2.Error as e:
        print(f"Error during table creation: {e}")
        if conn:
            conn.rollback()

    finally:
        if conn:
            cur.close()
            conn.close()