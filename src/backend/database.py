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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.connection.close()

    def execute(self, query, values=None):
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            return self.cursor.fetchall()
        except MySQLdb.Error as e:
            print("MySQL Error:", e)
            return None

    def create_table(self, table_name, columns):
        query = f"CREATE TABLE {table_name} ({', '.join(columns)});"
        self.execute(query)

    def delete_table(self, table_name):
        query = f"DROP TABLE IF EXISTS {table_name};"
        self.execute(query)