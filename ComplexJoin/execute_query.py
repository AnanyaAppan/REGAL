import psycopg2
import psycopg2.extras
import pandas as pd
import sys

def main () :
    try:
        conn = psycopg2.connect("dbname=tpch3 host='localhost' user='ananya' password='*Rasika0507'")
        sql = ""
        f_out = open("input_query.txt","r")
        for line in f_out :
            sql += " " + line
        f_out.close()
        dat = pd.read_sql_query(sql, conn)
        dat.to_csv('query_out.csv') 

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()

    
