import sqlite3
import os

DB_PATH = "database.db"

def insert_reference_image(order_id, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Reference image not found")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO orders (order_id, reference_image)
    VALUES (?, ?)
    """, (order_id, image_path))

    conn.commit()
    conn.close()