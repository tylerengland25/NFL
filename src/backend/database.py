from dotenv import load_dotenv
import os
import MySQLdb


class Database:
    def __init__(self):
        load_dotenv()
        self.connection = MySQLdb.connect(
            host=os.getenv("DATABASE_HOST"),
            user=os.getenv("DATABASE_USERNAME"),
            passwd=os.getenv("DATABASE_PASSWORD"),
            db=os.getenv("DATABASE")
        )
        self.cursor = self.connection.cursor()

    def __del__(self):
        self.cursor.close()
        self.connection.close()

    def execute(self, query, values=None):
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            return self.cursor.fetchall()
        except MySQLdb.Error as e:
            print("MySQL Error:", e)
            return None