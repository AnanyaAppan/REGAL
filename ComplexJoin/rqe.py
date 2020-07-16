import psycopg2
import psycopg2.extras
import pandas as pd
import sys
from utils import *
import time
import networkx.algorithms.isomorphism as iso

def main () :
    start = time.time()
    try:
        conn = psycopg2.connect("dbname=tpch3 host='localhost' user='ananya' password='*Rasika0507'")
        df = pd.read_csv("query.csv",header=0,index_col=0)
        df = df.sort_values(df.columns.tolist())
        tables = get_tables(conn)
        table_dict = get_table_dict(conn,tables)
        pre_process(table_dict,conn)
        print("--------------Pre processing over--------------%f"%(time.time()-start))
        cand_dict = get_c_and_lists(conn,start,table_dict,df)
        print(cand_dict)
        joinGraph = get_join_graph(conn)
        print("----Obtained CAND lists----%f"%(time.time()-start))
        for depth in range(1):
            print("----trying for depth %d----%f"%(depth,time.time()-start))
            star_ctrs,tree_dict = gen_instance_trees(conn,cand_dict,joinGraph,depth)
            valid = None
            merge = None
            blacklist = initialize_black_list(cand_dict)
            theta = df.sample(n=min(1,len(df.index)))
            for _, row in theta.iterrows():
                ExploreInstanceTree(conn,tree_dict,row,blacklist)
                print("-------Exploring Instance Tree-------%f"%(time.time()-start))
                star_ctrs_copy = [x for x in star_ctrs]
                for star in star_ctrs_copy :  
                    valid,tree_dict = updateStarCtrs(tree_dict,star,valid)
                    print("-------Update Star Centres-------%f"%(time.time()-start))
                    if valid[star] == {} :
                        if(star in star_ctrs) : star_ctrs.remove(star)
                for table in table_dict :
                    merge = get_merge_list(tree_dict,table,merge)
                print("-------Obtained merge list-------%f"%(time.time()-start))
            if(len(star_ctrs)==0): continue 
            if(valid != None):
                S = get_starred_set(valid)
                print("-------Obtained star set-------%f"%(time.time()-start))
                merged_stars = []
                for s in S :
                    merged_star = merge_stars(s,tree_dict)
                    if merged_star != None : 
                        merged_stars.append(merged_star)
                print("-------Obtained merged stars-------%f"%(time.time()-start))
                for merged_star in merged_stars :
                    initialize_tid_lists(merged_star,merge)
                    print("-------Initialized TID lists-------%f"%(time.time()-start))
                    query = gen_lattice(merged_star,merge,df,conn)
                    print("-------Gen Lattice-------%f"%(time.time()-start))
                    if(query != None):
                        print(query.split('LIMIT')[0])
                        f_out = open("extracted_query.txt","w+")
                        f_out.write(query.split('LIMIT')[0])
                        f_out.close()
                        post_process(table_dict,conn)
                        print("--------------Post processing over--------------%f"%(time.time()-start))
                        sys.exit()
        post_process(table_dict,conn)
        print("--------------Post processing over--------------%f"%(time.time()-start))

    except psycopg2.Error as e:
        print(type(e))
        print(e)

if __name__ == "__main__" :
    main ()
    
