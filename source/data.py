import pymysql
import os
from dotenv import load_dotenv

load_dotenv()
conn = pymysql.connect( 
    host=os.environ['DATAHOST'], 
    user=os.environ['DATAUSER'],
    password=os.environ['DATAPWD'],  
    database=os.environ['DATADB']
)

# For each individual user (intended not actually yet)
# Tracks which crypto they looked up. If previously looked up, delete row and insert into top of databse to track history.
def create_table():
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS BallData (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    xpos INT NOT NULL,
                    ypos INT NOT NULL
                )
            ''')
            conn.commit()
        print("Table created")
    except Exception as e:
        print(f"Error creating table: {e}")

def delete_table():
    try:
        with conn.cursor() as cursor:
            cursor.execute('DROP TABLE IF EXISTS Ball Data')
            conn.commit()
        print("Table deleted")
    except Exception as e:
        print(f"Error deleting table: {e}")
