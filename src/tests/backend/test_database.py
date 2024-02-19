import unittest
from backend.database import Database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.schema_name = 'test_schema'
        self.table_name = 'test_table'
        self.columns = ['id INT PRIMARY KEY', 'name VARCHAR(100)']
        self.db.create_schema(self.schema_name)

    def tearDown(self):
        self.db.delete_table(self.schema_name, self.table_name)
        self.db.delete_schema(self.schema_name)

    def test_delete_schema(self):
        schema_name = 'test_delete_schema'
        self.db.create_schema(schema_name)
        self.db.delete_schema(schema_name)
        schemas = self.db.execute("SHOW DATABASES")
        self.assertNotIn((schema_name,), schemas)
    
    def test_create_schema(self):
        schema_name = 'test_create_schema'
        self.db.create_schema(schema_name)
        schemas = self.db.execute("SHOW DATABASES")
        self.assertIn((schema_name,), schemas)

    def test_delete_table(self):
        self.db.create_table(self.schema_name, self.table_name, self.columns)
        self.db.delete_table(self.schema_name, self.table_name)
        tables = self.db.execute(f"SHOW TABLES IN {self.schema_name}")
        self.assertNotIn((self.table_name,), tables)

    def test_create_table(self):
        self.db.create_table(self.schema_name, self.table_name, self.columns)
        tables = self.db.execute(f"SHOW TABLES IN {self.schema_name}")
        self.assertIn((self.table_name,), tables)

    def test_execute(self):
        # Test the execute method
        query = f"SELECT * FROM {self.schema_name}.{self.table_name}"
        result = self.db.execute(query)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
