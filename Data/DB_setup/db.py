import os
import mysql.connector

# === Configuration ===
DB_NAME = 'ImageDB'
DB_USER = 'root'
DB_PASSWORD = 'peri1908'
DB_HOST = 'localhost'
ROOT_DIR = '../../Data/results'  # Update this if your directory is elsewhere

# === Connect to MySQL ===
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()

# === Create Database ===
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")

# === Create Table ===
cursor.execute("""
CREATE TABLE IF NOT EXISTS ImageData (
    id INT AUTO_INCREMENT PRIMARY KEY,
    regular_image TEXT,
    log_image TEXT,
    diameter FLOAT,
    thickness FLOAT,
    ratio FLOAT,
    ref_index INT,
    filename VARCHAR(255),
    filepath TEXT
)
""")

# === Helper Functions ===
def parse_filename(name):
    diameter = int(name[:5])    # e.g., 04500 -> 4.5
    thickness = int(name[5:9])      # e.g., 1500 -> 150.0
    ratio = int(name[9:12])      # e.g., 4001 -> 4.001
    return diameter, thickness, ratio

def read_f06_as_text(path):
    with open(path, 'r') as f:
        return f.read()

# === Walk Through Folders ===
for folder in os.listdir(ROOT_DIR):
    if folder.startswith("Refindx"):
        ref_index = int(float(folder.replace("Refindx1.", "")) )
        folder_path = os.path.join(ROOT_DIR, folder)
        files = os.listdir(folder_path)

        # Filter only regular (a.f06) files
        for file in files:
            if file.endswith("a.f06"):
                base = file[:-5]  # Remove "a.f06"
                regular_path = os.path.join(folder_path, f"{base}a.f06")
                log_path = os.path.join(folder_path, f"{base}b.f06")

                if not os.path.exists(log_path):
                    print(f" Log file missing for: {regular_path}")
                    continue

                diameter, thickness, ratio = parse_filename(base)
                regular_image = read_f06_as_text(regular_path)
                log_image = read_f06_as_text(log_path)
                filepath = os.path.abspath(regular_path)

                # Insert into database
                cursor.execute("""
                    INSERT INTO ImageData (
                        regular_image, log_image, diameter, thickness,
                        ratio, ref_index, filename, filepath
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    regular_image, log_image, diameter, thickness,
                    ratio, ref_index, base, filepath
                ))
                print(f"✅ Inserted: {base}")

conn.commit()
cursor.close()
conn.close()
print(" Done populating the database.")
