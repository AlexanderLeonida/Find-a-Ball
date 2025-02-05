# import opencv
from GameWindow import * 
from data import create_table, delete_table
import atexit

if __name__ == '__main__':
    create_table()
    get_ball()
    atexit.register(delete_table)
    # atexit.register(scheduler.shutdown)


