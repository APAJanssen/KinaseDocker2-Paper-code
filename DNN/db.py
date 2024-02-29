'''
Module that queries the database for easier coding in other scripts.

20 Oct. 2021 By Jordy Schifferstein
'''
import sqlite3


class DBConnection:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    def query(self, sql):
        '''
        Function that queries the loaded database.

        Params:
        - sql (string)

        Returns:
        - results (list)
        '''
        self.cur.execute(sql)
        results = self.cur.fetchall()

        # If only 1 item was requested, directly convert to list instead of list of tuples
        if len(results) and len(results[0]) == 1:
            results = list(map(lambda x: x[0], results))

        return results

    def delete(self, sql):
        '''
        Function that deletes from the loaded database.

        Params:
        - sql (string)
        '''
        self.cur.execute(sql)
        self.conn.commit()

    def update(self, sql):
        '''
        Function that updates the loaded database.

        Params:
        - sql (string)
        '''
        self.cur.execute(sql)
        self.conn.commit()

def setup(database):
    '''
    Connect to database
    '''
    conn = sqlite3.connect(database)
    return DBConnection(conn)