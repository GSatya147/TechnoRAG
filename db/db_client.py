from contextlib import contextmanager
import logging 
from typing import Any

import psycopg2
import psycopg2.extras # RealDictCursor, execute_batch
from pgvector.psycopg2 import register_vector

from configurables.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

logger = logging.getLogger(__name__)

def get_connection():
    kwargs_dict = {
        "host" : DB_HOST,
        "port" : DB_PORT, # 5432
        "dbname" : DB_NAME,
        "user" : DB_USER,
        "password" : DB_PASSWORD,
    }

    return psycopg2.connect(**kwargs_dict)

@contextmanager
def db_connection():
    "yields a connection, commits on clean exit, rolls back on exception, always closes."
    conn = get_connection()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def execute_query(sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
    "Execute a SELECT query and return all the records as list of dicts."
    with db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        
def execute_write(sql: str, params: tuple | None = None ) -> None:
    "Execute a INSERT, UPDATE and DELETE query."
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)

def execute_batch(sql: str, params_list: list[tuple]) -> None:
    "Execute a write query for many rows efficiently."
    with db_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list)
    
def init_db(schema_path: str = "./db/schema.sql") -> None:
    "Run the sql schema file against db."
    with open(schema_path, "r") as f:
        sql = f.read()

    conn = get_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    cur.execute(statement)
    finally:
        conn.close()

    logger.info("Database schema initialised from %s", schema_path)

if __name__=="__main__":
    init_db()