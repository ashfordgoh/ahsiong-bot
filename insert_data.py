import sqlite3
import pandas as pd
import json

def insert_into_products():
    # Connect to SQLite database
    conn = sqlite3.connect('ingredient_database.db')
    cursor = conn.cursor()

    # Load products CSV into pandas DataFrame
    products_df = pd.read_csv("./shengsiong.csv", delimiter=";")

    # Insert product data into the database
    for index, row in products_df.iterrows():
        ingredients_json = json.dumps(row['ingredients'].split(","))
        cursor.execute('''
            INSERT OR IGNORE INTO products (name, ingredients, category)
            VALUES (?, ?, ?)
        ''', (row['name'], ingredients_json, row['category']))

    conn.commit()
    conn.close()
    print("Product data inserted successfully.")
    

def insert_into_allergies():
    # Connect to SQLite database
    conn = sqlite3.connect('ingredient_database.db')
    cursor = conn.cursor()

    # Load allergies CSV into pandas DataFrame
    allergies_df = pd.read_csv("./allergy_list.csv")

    # Insert allergy data into the database
    for index, row in allergies_df.iterrows():
        allergy_ingredients_json = json.dumps(row['ingredients'].split(","))
        cursor.execute('''
            INSERT OR IGNORE INTO allergies (type, ingredients)
            VALUES (?, ?)
        ''', (row['type'], allergy_ingredients_json))

    conn.commit()
    conn.close()
    print("Allergy data inserted successfully.")