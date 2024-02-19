import os
import MySQLdb
from MySQLdb.cursors import DictCursor
from dotenv import load_dotenv


class Database:
    def __init__(self):
        load_dotenv()
        self.host = os.getenv("DATABASE_HOST")
        self.user = os.getenv("DATABASE_USERNAME")
        self.password = os.getenv("DATABASE_PASSWORD")
        self.db = os.getenv("DATABASE")

    def __enter__(self):
        self.connection = MySQLdb.connect(
            host=self.host,
            user=self.user,
            passwd=self.password,
            db=self.db,
            cursorclass=DictCursor
        )
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self):
        self.connection.close()

    def execute(self, query, values=None):
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            return self.cursor.fetchall()
        except MySQLdb.Error as e:
            print("MySQL Error:", e)
            return None
        
    def create_schema(self, schema_name):
        query = f"CREATE SCHEMA {schema_name}"
        self.execute(query)

    def delete_schema(self, schema_name):
        query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
        self.execute(query)

    def create_table(self, schema_name, table_name, columns):
        query = f"CREATE TABLE {schema_name}.{table_name} ({', '.join(columns)})"
        self.execute(query)

    def delete_table(self, schema_name, table_name):
        query = f"DROP TABLE IF EXISTS {schema_name}.{table_name}"
        self.execute(query)