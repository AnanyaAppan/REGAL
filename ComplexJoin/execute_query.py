import psycopg2
import psycopg2.extras
import pandas as pd

def main () :
    try:
        conn = psycopg2.connect("dbname=tpch3 host='localhost' user='ananya' password='*Rasika0507'")
        sql = """
           select 
           l_shipmode, max(l_discount)
           from lineitem
           where l_discount < 0.08 
           group by l_shipmode
            """
        dat = pd.read_sql_query(sql, conn)
        dat.to_csv('query.csv') 

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()

    
