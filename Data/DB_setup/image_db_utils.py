import mysql.connector
from Data.DB_setup.db_config import DB_CONFIG

class ImageDB:
    def __init__(self):
        self.conn = mysql.connector.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor(dictionary=True)

    def close(self):
        self.cursor.close()
        self.conn.close()

    def search_image_by_id(self, image_id):
        self.cursor.execute("SELECT * FROM ImageData WHERE id = %s", (image_id,))
        return self.cursor.fetchone()

    def search_image_by_refindex(self, ref_index):
        self.cursor.execute("SELECT * FROM ImageData WHERE ref_index = %s LIMIT 1", (ref_index,))
        return self.cursor.fetchone()

    def search_images_by_diameter(self, diameter):
        self.cursor.execute("SELECT * FROM ImageData WHERE diameter = %s", (diameter,))
        return self.cursor.fetchall()

    def search_image_by_dtr(self, diameter, thickness, ratio):
        query = """
            SELECT * FROM ImageData 
            WHERE diameter = %s AND thickness = %s AND ratio = %s
            LIMIT 1
        """
        self.cursor.execute(query, (diameter, thickness, ratio))
        return self.cursor.fetchone()

    def search_image_by_all_dtr(self, diameter, thickness, ratio,ref_index):
        query = """
            SELECT * FROM ImageData 
            WHERE diameter = %s AND thickness = %s AND ratio = %s AND ref_index = %s
            LIMIT 1
        """
        self.cursor.execute(query, (diameter, thickness, ratio, ref_index))
        return self.cursor.fetchone()
