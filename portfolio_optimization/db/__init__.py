from configparser import ConfigParser

import psycopg2
import sys
import traceback
from psycopg2 import extras
import pandas as pd
import os

from portfolio_optimization import PORTFOLIO_BASE_DIR


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_connection():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    DB_USER = parser.get("postgresql", "DB_USER")
    DB_PASS = parser.get("postgresql", "DB_PASS")
    DB_HOST = parser.get("postgresql", "DB_HOST")
    DB_NAME = parser.get("postgresql", "DB_NAME")

    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connection parameters
        params = {
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASS,
            'host': DB_HOST
        }

        # connect to the PostgreSQL server
        # print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print("CONNECTION ERROR: ", error)
        if conn is not None:
            conn.close()

def execute_db_commands(commands):
    conn = None
    try:
        # connect to the PostgreSQL server
        conn = get_connection()
        cur = conn.cursor()
        # execute command one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        traceback.print_stack()
        eprint(error)
    finally:
        if conn is not None:
            conn.close()

def insert_df_into_table(df, tablename):

    import pandas as pd
    df = df.where(pd.notnull(df), None)
    # print(f"INSERT DF INTO {tablename}")
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    query = "INSERT INTO %s(%s) VALUES %%s ON CONFLICT DO NOTHING;" % (tablename, cols)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        traceback.print_exc()
        eprint(f"Error: {error}")
        conn.rollback()
        cursor.close()
    finally:
        #print("execute_values() done")
        cursor.close()

def create_indicator_table():
    execute_db_commands(["""
        CREATE TABLE IF NOT EXISTS public.indicator
        (
            id serial NOT NULL,
            source text NOT NULL,
            name text NOT NULL,
            date date NOT NULL,
            value numeric NOT NULL,
            PRIMARY KEY (id)
        )
        WITH (
            OIDS = FALSE
        );
        """])

def get_df_from_table(tablename, where=";"):
    conn = get_connection()
    cur = conn.cursor()
    q = f'''SELECT * FROM {tablename} {where}'''
    # print(q)
    cur.execute(q)
    data = cur.fetchall()
    cols = []
    for elt in cur.description:
        cols.append(elt[0])
    df = pd.DataFrame(data=data, columns=cols)
    cur.close()
    return df