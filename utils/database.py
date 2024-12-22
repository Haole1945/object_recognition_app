import mysql.connector

def connect_to_db(db_config):
    try:
        db_connection = mysql.connector.connect(**db_config)
        db_cursor = db_connection.cursor()
        print("Connected to database successfully!")

        # Kiểm tra kết nối
        db_cursor.execute("SELECT 1")
        result = db_cursor.fetchone()
        if result:
            print("Database connection test successful!")
        else:
            print("Database connection test failed!")
        return db_connection, db_cursor

    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None, None

def create_table(db_cursor, db_connection, table_name):
    try:
        # Sử dụng IF NOT EXISTS để tránh lỗi nếu bảng đã tồn tại
        db_cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT PRIMARY KEY,
                class_name VARCHAR(255),
                confidence FLOAT
            )
            """
        )
        db_connection.commit()
        print(f"Table {table_name} created or already exists.")

        # Nếu bảng đã tồn tại, xóa dữ liệu cũ
        db_cursor.execute(f"DELETE FROM {table_name}")
        db_connection.commit()
        print(f"Data cleared from existing table {table_name}.")

    except mysql.connector.Error as err:
        print(f"Error creating or clearing table {table_name}: {err}")

def insert_data(db_cursor, db_connection, table_name, id_num, class_name, confidence):
    # Kiểm tra dữ liệu đầu vào
    if not all([id_num is not None, class_name is not None, confidence is not None]):
        print("Error: Incomplete data. Skipping insertion.")
        return

    processed_ids = set()
    if id_num not in processed_ids:
        try:
            sql = f"INSERT INTO {table_name} (id, class_name, confidence) VALUES (%s, %s, %s)"
            val = (id_num, class_name, confidence)
            db_cursor.execute(sql, val)
            db_connection.commit()
            processed_ids.add(id_num)
            print(f"Data inserted: ID={id_num}, Class={class_name}, Confidence={confidence}")
        except mysql.connector.Error as err:
            print(f"Error inserting data: {err}")

def close_db_connection(db_cursor, db_connection):
    if db_connection is not None and db_connection.is_connected():
        db_cursor.close()
        db_connection.close()
        print("Database connection closed.")