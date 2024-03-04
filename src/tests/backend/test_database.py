import unittest
from backend.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.table_name = 'test_table'
        self.columns = ['id INT PRIMARY KEY', 'name VARCHAR(100)']

    def tearDown(self):
        with Database() as db:
            db.delete_table(self.table_name)
    
    def test_execute(self):
        with Database() as db:
            db.create_table(self.table_name, self.columns)
            query = f"SELECT * FROM {self.table_name};"
            result = db.execute(query)
            self.assertIsNotNone(result)

    def test_delete_table(self):
        with Database() as db:
            db.create_table(self.table_name, self.columns)
            db.delete_table(self.table_name)
            tables = db.execute(f"SHOW TABLES;")
            self.assertNotIn(self.table_name, [list(table.values())[0] for table in tables])


    def test_create_table(self):
        with Database() as db:
            db.create_table(self.table_name, self.columns)
            tables = db.execute(f"SHOW TABLES IN nfl;")
            self.assertIn(self.table_name, [list(table.values())[0] for table in tables])

if __name__ == "__main__":
    unittest.main()
