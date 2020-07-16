import psycopg2
import psycopg2.extras
from datetime import date
from decimal import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time

alpha = 10000

def post_process(tables,connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    for table in tables :
        cursor.execute("""
                    ALTER TABLE %s
                    DROP COLUMN TID
                    """%table)
    cursor.close
    connection.commit()

def pre_process(tables,connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    for table in tables :
        cursor.execute("""
                    ALTER TABLE %s
                    ADD COLUMN TID SERIAL
                    """%table)
    cursor.close
    connection.commit()

def get_tables(connection):
    """
    Create and return a list of dictionaries with the
    schemas and names of tables in the database
    connected to by the connection argument.
    """

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""SELECT table_schema, table_name
                      FROM information_schema.tables
                      WHERE table_schema != 'pg_catalog'
                      AND table_schema != 'information_schema'
                      AND table_type='BASE TABLE'
                      ORDER BY table_schema, table_name""")

    tables = cursor.fetchall()

    cursor.close()

    return tables

def get_columns(connection, table_schema, table_name):

    """
    Creates and returns a list of dictionaries for the specified
    schema.table in the database connected to.
    """

    where_dict = {"table_schema": table_schema, "table_name": table_name}
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT column_name, ordinal_position, is_nullable, data_type, character_maximum_length
                      FROM information_schema.columns
                      WHERE table_schema = %(table_schema)s
                      AND table_name   = %(table_name)s
                      ORDER BY ordinal_position""",
                      where_dict)
    columns = cursor.fetchall()
    cursor.close()
    col_dict = {}
    for col in columns:
        col_dict[col["column_name"]] = col
    return col_dict

def get_primary_keys(connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
                    select kcu.table_name,kcu.column_name as key_column
                    from information_schema.table_constraints tco
                    join information_schema.key_column_usage kcu 
                        on kcu.constraint_name = tco.constraint_name
                        and kcu.constraint_schema = tco.constraint_schema
                        and kcu.constraint_name = tco.constraint_name
                    where tco.constraint_type = 'PRIMARY KEY'
                    order by kcu.ordinal_position;
                        """)
    data = cursor.fetchall()
    cursor.close()
    ret = {}
    for row in data :
        if row['table_name'] not in ret :
            ret[row['table_name']] = [row['key_column']]
        else :
            ret[row['table_name']].append(row['key_column'])
    for table in ret :
        ret[table].sort()
    return ret

def get_foreign_keys(connection) :
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
                    select kcu.table_schema || '.' ||kcu.table_name as foreign_table,
                    rel_tco.table_schema || '.' || rel_tco.table_name as primary_table,
                    string_agg(kcu.column_name, ', ') as fk_columns,
                    kcu.constraint_name
                    from information_schema.table_constraints tco
                    join information_schema.key_column_usage kcu
                            on tco.constraint_schema = kcu.constraint_schema
                            and tco.constraint_name = kcu.constraint_name
                    join information_schema.referential_constraints rco
                            on tco.constraint_schema = rco.constraint_schema
                            and tco.constraint_name = rco.constraint_name
                    join information_schema.table_constraints rel_tco
                            on rco.unique_constraint_schema = rel_tco.constraint_schema
                            and rco.unique_constraint_name = rel_tco.constraint_name
                    where tco.constraint_type = 'FOREIGN KEY'
                    group by kcu.table_schema,
                            kcu.table_name,
                            rel_tco.table_name,
                            rel_tco.table_schema,
                            kcu.constraint_name
                    order by kcu.table_schema,
                            kcu.table_name;
    """)
    columns = cursor.fetchall()
    cursor.close()
    for col in columns :
        col['fk_columns'] = col['fk_columns'].split(',')
        col['fk_columns'].sort()
        col['fk_columns'] = ','.join(col['fk_columns']).strip()
    return columns

def get_col_type (connection,col_name,table_name) :
    """
    Gets type of values of a clomun, given table and column name
    """
    query = """SELECT %s
                FROM %s
                LIMIT 1
                """ % (col_name,table_name)
    df = pd.read_sql_query(query, connection)
    for col_out in df :
        if(type(df[col_out][0]) == date): return str
        else: return type(df[col_out][0])

def enclose_with_quotes(string) :
    return "'" + string + "'"

def list_to_string(values) :
    val_type = type(values[0])
    ret = ""
    if(val_type == float or val_type == int):
        ret = ','.join([str(value) for value in values]).strip(',')
    else :
        ret = ','.join([enclose_with_quotes(value) for value in values]).strip(',')
    ret = '(' + ret + ')'
    return ret

def get_col_values_from_list (connection,col_name,table_name,values) :
    """
    Gets values of a clomun, given table and column name
    """
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT DISTINCT %s
                      FROM %s
                      WHERE %s IN %s
                      """ % (col_name,table_name,col_name,list_to_string(values)))
    vals = cursor.fetchall()
    cursor.close()
    ret_vals = []
    for val in vals :
        if type(val[col_name]) == str :
            ret_vals.append(val[col_name])
        elif type(val[col_name]) == date :
            ret_vals.append(str(val[col_name]))
        else :
            ret_vals.append(eval(str(val[col_name])))
    return ret_vals

def get_col_values (connection,col_name,table_name) :
    """
    Gets values of a clomun, given table and column name
    """
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT DISTINCT %s
                      FROM %s
                      """ % (col_name,table_name))
    vals = cursor.fetchall()
    cursor.close()
    ret_vals = []
    for val in vals :
        if type(val[col_name]) == str :
            ret_vals.append(val[col_name])
        elif type(val[col_name]) == date :
            ret_vals.append(str(val[col_name]))
        else :
            ret_vals.append(eval(str(val[col_name])))
    return ret_vals
    

def get_table_dict(connection, tables):
    """
    Creates and returns a dictionary with the table names
    as keys, and the dictionary returned by get_columns as 
    the corresponding values
    """
    d = {}
    for row in tables :
        d[row["table_name"]] = get_columns(connection, row["table_schema"], row["table_name"])
    return d

def get_table_values(conn, table) :
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT DISTINCT *
                      FROM %s
                      """ % (table))
    table_vals = cursor.fetchall()
    cursor.close()
    ret = {}
    for row in table_vals :
        for col in row :
            if col not in ret :
                ret[col] = set([row[col]])
            else :
                ret[col].add(row[col])
    return ret
    # for val in vals :
    #     if type(val[col_name]) == str :
    #         ret_vals.append(val[col_name])
    #     elif type(val[col_name]) == date :
    #         ret_vals.append(str(val[col_name]))
    #     else :
    #         ret_vals.append(eval(str(val[col_name])))

def get_c_and_lists(connection,start,table_dict,out) :
    
    ret_dict = {}
    for col_out in out :
        if(col_out == 'tid'): continue
        ret_dict[col_out] = []

    col_count = {}
    for table in table_dict :
        for col in table_dict[table]:
            if col == 'tid' : continue
            col_count[col] = 0
            col_type = get_col_type(connection,col,table)
            col_values = set()
            for col_out in ret_dict : 
                if(col_type == type(out[col_out][0])):
                    if col_values == set() : 
                        values = get_col_values(connection,col,table)
                        col_values = set(values)
                    if set(out[col_out]).issubset(col_values):
                        ret_dict[col_out].append(table + "." + col + "." + str(col_count[col]))
                        col_count[col] += 1
    
    # for table in table_dict :
    #     print("----Before obtaining table values----%f"%(time.time()-start))
    #     table_values = get_table_values(connection,table)
    #     print("----Obtained table values----%f"%(time.time()-start))
    #     for col in table_values :
    #         col_count[col] = 0
    #         for col_out in ret_dict :
    #             if set(out[col_out]).issubset(table_values[col]):
    #                 ret_dict[col_out].append(table + "." + col + "." + str(col_count[col]))
    #                 col_count[col] += 1

    return ret_dict

def get_foreign_key_dict(primary_keys,foreign_keys) :
    ret_dict = {}
    for row in foreign_keys :
        table = row['foreign_table'].split(".")[1]
        primary_table = row['primary_table'].split(".")[1]
        foreign_key_cols = row['fk_columns'].split(',')
        for i in range(len(foreign_key_cols)):
            d = {}
            d["foreign_col"] = foreign_key_cols[i]
            d["primary_col"] = primary_keys[primary_table][i]
            d["primary_table"] = primary_table
            if table not in ret_dict:
                ret_dict[table] = [d]
            else :
                ret_dict[table].append(d)
    return ret_dict

def get_primary_key_dict(foregin_key_dict) :
    ret_dict = {}
    for table in foregin_key_dict :
        for constraint in foregin_key_dict[table] :
            primary_table = constraint["primary_table"]
            d = {}
            d["foreign_table"] = table
            d["foreign_col"] = constraint["foreign_col"]
            d["primary_col"] = constraint["primary_col"]
            if primary_table not in ret_dict:
                ret_dict[primary_table] = [d]
            else :
                ret_dict[primary_table].append(d)
    return ret_dict

def merge_join_graph(G):
    merged_graph = nx.MultiGraph()
    nodes = list(G.nodes())
    tables = [node.split(".")[0] for node in nodes]
    for table in tables :
        merged_graph.add_node(table)
    for table in tables :
        cols = [node for node in nodes if node.split(".")[0] == table]
        for col in cols:
            for x in G[col]:
                x_table = x.split(".")[0]
                if(not (((table,x_table) in list(merged_graph.edges())) or ((x_table,table) in list(merged_graph.edges())))):
                    index = merged_graph.add_edge(table,x_table)
                    merged_graph[table][x_table][index]['join'] = G[col][x]['join']
                else:
                    joins = [merged_graph[table][x_table][index]['join'] for index in merged_graph[table][x_table]]
                    if G[col][x]['join'] not in joins:
                        index = merged_graph.add_edge(table,x_table)
                        merged_graph[table][x_table][index]['join'] = G[col][x]['join']
    return merged_graph

def get_join_graph(connection) :
    foreign_keys = get_foreign_keys(connection)
    primary_keys = get_primary_keys(connection)
    foreign_key_dict = get_foreign_key_dict(primary_keys,foreign_keys)
    primary_key_dict = get_primary_key_dict(foreign_key_dict)
    G = nx.Graph()
    for table in primary_key_dict:
        for fk_rel in primary_key_dict[table]:
            foreign_table = fk_rel['foreign_table']
            node1 = table + "." + fk_rel['primary_col']
            node2 = foreign_table + "." + fk_rel['foreign_col']
            G.add_edge(node1,node2)
            G[node1][node2]['join'] = [node1,node2] 
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            paths = list(nx.all_simple_paths(G, source=node1, target=node2))
            for path in paths:
                if(len(path)>2):
                    G.add_edge(node1,node2)
                    G[node1][node2]['join'] = [node1,node2]
    return merge_join_graph(G)


def get_instance_tree(table_count,col,joinGraph,depth) :
    table_name = col.split('.')[0]
    col_name = col.split('.')[1]
    G = nx.Graph()
    attr = {}
    if table_name + '_' + str(table_count[table_name]) not in attr :
        attr[table_name + '_' + str(table_count[table_name])] = {}
    G.add_node(table_name + '_' + str(table_count[table_name]))
    attr[table_name + '_' + str(table_count[table_name])]['col'] = [col_name]
    attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
    attr[table_name + '_' + str(table_count[table_name])]['table'] = table_name
    queue = []
    queue.append(table_name + '_' + str(table_count[table_name]))
    table_count[table_name] += 1
    for _ in range(depth) :
        new_queue = []
        while(queue != []) :
            head = queue.pop(0)
            table = head.split('_')[0]
            try :
                for table_name in joinGraph[table]:
                    for index in joinGraph[table][table_name]:
                        G.add_edge(head,table_name + '_' + str(table_count[table_name]))
                        G[head][table_name + '_' + str(table_count[table_name])]['join'] = joinGraph[table][table_name][index]['join']
                        if table_name + '_' + str(table_count[table_name]) not in attr :
                            attr[table_name + '_' + str(table_count[table_name])] = {}
                        attr[table_name + '_' + str(table_count[table_name])]['col'] = None
                        attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
                        attr[table_name + '_' + str(table_count[table_name])]['table'] = table_name
                        new_queue.append(table_name + '_' + str(table_count[table_name]))
                        table_count[table_name] += 1
            # try :
            #     for key in foreign_key_dict[table] :
            #             table_name = key['primary_table']
            #             G.add_edge(head,table_name + '_' + str(table_count[table_name]))
            #             G[head][table_name + '_' + str(table_count[table_name])]["join"] = [table + "." + key['foreign_col']
            #                                                                         ,table_name + "." + key['primary_col']]
            #             if table_name + '_' + str(table_count[table_name]) not in attr :
            #                 attr[table_name + '_' + str(table_count[table_name])] = {}
            #             attr[table_name + '_' + str(table_count[table_name])]['col'] = None
            #             attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
            #             attr[table_name + '_' + str(table_count[table_name])]['table'] = table_name
            #             new_queue.append(table_name + '_' + str(table_count[table_name]))
            #             table_count[table_name] += 1
            # except :
            #     pass

            # try :
            #     for key in primary_key_dict[table] : 
            #             table_name = key['foreign_table']
            #             G.add_edge(head,table_name + '_' + str(table_count[table_name]))
            #             G[head][table_name + '_' + str(table_count[table_name])]["join"] = [table + "." + key['primary_col']
            #                                                                             ,table_name + "." + key['foreign_col']]
            #             if table_name + '_' + str(table_count[table_name]) not in attr :
            #                 attr[table_name + '_' + str(table_count[table_name])] = {}
            #             attr[table_name + '_' + str(table_count[table_name])]['col'] = None
            #             attr[table_name + '_' + str(table_count[table_name])]['star'] = 1
            #             attr[table_name + '_' + str(table_count[table_name])]['table'] = table_name
            #             new_queue.append(table_name + '_' + str(table_count[table_name]))
            #             table_count[table_name] += 1
            except : 
                pass
        queue = new_queue
    nx.set_node_attributes(G,attr)
    return table_count,G

def bottom_up_prune(tree,root,prev_node,is_star) :
    sub_trees = [node for node in tree[root]]
    for node in sub_trees :
        if(node != prev_node): tree = bottom_up_prune(tree,node,root,is_star)
    if ((list(tree[root])==[prev_node] or len(list(tree[root]))==0) and root != list(tree.nodes())[0]) :
        if is_star[root] == 0 : 
            tree.remove_node(root)
    return tree
    

def gen_instance_trees(connection,cand_dict,joinGraph,depth) : 

    tables = get_tables(connection)
    table_dict = get_table_dict(connection,tables)
    tree_dict = {}
    table_count = {}
    for table in table_dict :
        table_count[table] = 0

    for col_out in cand_dict :
        tree_dict[col_out] = {}
        for col in cand_dict[col_out]:
            table_count,tree = get_instance_tree(table_count,col,joinGraph,depth)
            tree_dict[col_out][col] = tree
    
    star_ctrs = set([table for table in table_dict])

    for col_out in tree_dict :
        star_candidates = set()
        for col in tree_dict[col_out]:
            tree = tree_dict[col_out][col]
            star_candidates = star_candidates.union(set(table.split('_')[0] for table in tree.nodes()))
        star_ctrs = star_ctrs.intersection(star_candidates)

    for col_out in tree_dict :
        for col in tree_dict[col_out]:
            tree = tree_dict[col_out][col]
            attr = {}
            for node in list(tree.nodes()):
                attr[node] = {}
                if((node.split('_')[0] in star_ctrs)): attr[node]["star"] = 1
                else: attr[node]["star"] = 0
            nx.set_node_attributes(tree,attr)
            is_star = nx.get_node_attributes(tree,'star')
            tree = bottom_up_prune(tree,list(tree.nodes())[0],None,is_star)
        
    return star_ctrs,tree_dict

def get_tid_util(tree,prev_node,cur_node,attr,conn, wildcard) :
    if(len(attr[prev_node]['tid'])==0) : return attr,tree
    if not wildcard : 
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur_table = cur_node.split('_')[0]
        prev_table = prev_node.split('_')[0]
        query = """SELECT DISTINCT %s
                    FROM %s,%s
                    WHERE %s = %s
                    AND %s IN %s
                """%(cur_table+".TID",cur_table,prev_table,tree[prev_node][cur_node]["join"][0],
                    tree[prev_node][cur_node]["join"][1],prev_table + ".TID" ,
                    "(" + ",".join(attr[prev_node]['tid']).strip(",") + ")")
        cursor.execute(query)
        tids = cursor.fetchall()
        tids = [str(x['tid']) for x in tids]
        if len(tids) > alpha : 
            wildcard = True
            tids = ['wc']
        cursor.close()
    else :
        tids = ['wc']
    attr[cur_node]['tid'] = tids
    for node in tree[cur_node] :
        if(node != prev_node): attr,tree = get_tid_util(tree,cur_node,node,attr,conn, wildcard)
    return attr,tree


def get_tid_lists(tree,conn,val) :

    if(type(val)==date or type(val)==str):
        val = "'" + str(val) + "'"
    root = list(tree.nodes())[0]
    cols = nx.get_node_attributes(tree,"col")
    attr = {}
    for node in list(tree.nodes()):
        attr[node] = {}
        attr[node]["tid"] = []
    root_table = root.split('_')[0]
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""SELECT DISTINCT TID
                      FROM %s
                      WHERE %s = %s"""%(root_table,cols[root][0],str(val)))
    tids = cursor.fetchall()
    tids = [str(x['tid']) for x in tids]
    cursor.close()
    if len(tids) > alpha : 
        wildcard = True
        tids = ['wc']
    else: wildcard = False
    attr[root]["tid"] += tids
    for node in tree[root] : 
        attr,tree = get_tid_util (tree,root,node,attr,conn,wildcard)
    nx.set_node_attributes(tree,attr)

def ExploreInstanceTree(conn,tree_dict,row,blacklist) :
    keys = list(tree_dict.keys())
    for i in range(len(keys)):
        col_out = keys[i]
        val = row[i]
        for col in tree_dict[col_out] :
            if(val not in blacklist[col_out][col]):
                tree = tree_dict[col_out][col]
                get_tid_lists(tree,conn,val)
                wildcard_nodes = [x for x,y in tree.nodes(data=True) if y['tid']==['wc']]
                if(wildcard_nodes != []): blacklist[col_out][col].append(val)
                del_empty_tid(tree) 

def draw_tree(tree) :
    attr = nx.get_node_attributes(tree,'tid')
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos,with_labels=True)
    # nx.draw_networkx_edge_labels(tree,pos=nx.spring_layout(tree))
    # nx.draw_networkx_labels(tree, pos, labels = attr)
    plt.show()

def del_empty_tid(tree) :
    empty_nodes = [x for x,y in tree.nodes(data=True) if len(y['tid'])==0]
    tree.remove_nodes_from(empty_nodes)

def get_all_possibilities(c_and_dict,keys) : 
    if len(keys)==0 : return [[]]
    ret = []
    for possib in get_all_possibilities(c_and_dict,keys[1:]):
        for col in c_and_dict[keys[0]]:
            ret.append([col]+possib)
    return ret

def get_valid(tree_dict,star,is_table_star):
    valid = {}
    valid[star] = {} # valid list for star
    wc_dict = {}
    for col_out in tree_dict:
        wc_dict[col_out] = set()
        for col in tree_dict[col_out]:
            tree = tree_dict[col_out][col]
            tids = nx.get_node_attributes(tree,'tid')
            is_star = nx.get_node_attributes(tree,'star')
            for node in list(tree.nodes()) :
                if((node.split('_')[0]==star and is_star[node] and is_table_star) or (node.split('_')[0]==star and not is_table_star)): 
                    for tid in tids[node]:
                        if(tid not in valid[star]):
                            valid[star][tid] = {} # inverted list for star
                            valid[star][tid][col_out] = set([node])
                        else :
                            if(col_out not in valid[star][tid]):
                                valid[star][tid][col_out] = set([node])
                            else :
                                valid[star][tid][col_out].add(node)
                    if tids[node] == ['wc'] : 
                        wc_dict[col_out].add(node)

    for col_out in tree_dict:
        for tid in valid[star]:
            if col_out in valid[star][tid] :
                valid[star][tid][col_out].union(wc_dict[col_out])
            else :
                if(wc_dict[col_out] != set()):
                    valid[star][tid][col_out] = wc_dict[col_out]

    tid_keys = [k for k in valid[star].keys()]
    if is_table_star : 
        for tid in tid_keys :
            if len(valid[star][tid].keys()) != len(tree_dict.keys()):
                del valid[star][tid]
    return valid

def cross_tuple_prune(valid,prev_valid,star):
    if(prev_valid==None): return valid
    if(star not in prev_valid): prev_valid[star] = valid[star]
    prev_valid_tids = [k for k in prev_valid[star].keys()]
    valid_tids = [k for k in valid[star].keys()]
    
    for tid in valid_tids :
        if tid in prev_valid_tids :
            cols = [k for k in prev_valid[star][tid].keys()]
            for col in cols:
                if col in valid[star][tid] :
                    prev_valid[star][tid][col] = prev_valid[star][tid][col].intersection(valid[star][tid][col])
                else :
                    del prev_valid[star][tid][col]
        else :
            prev_valid[star][tid] = valid[star][tid]
    for tid in prev_valid_tids :
        # if tid not in valid_tids :
        #     del prev_valid[star][tid]
        if tid in valid_tids :
            for col in prev_valid[star][tid]:
                prev_valid[star][tid][col] = prev_valid[star][tid][col].union(valid[star][tid][col])
    return prev_valid    

def updateStarCtrs(tree_dict,star,prev_valid):

    valid = get_valid(tree_dict,star,True)
    new_valid = cross_tuple_prune(valid,prev_valid,star)
    for col_out in tree_dict:
        for col in tree_dict[col_out] :
            tree = tree_dict[col_out][col]
            is_star = nx.get_node_attributes(tree,'star')
            star_nodes = [x for x,y in tree.nodes(data=True) 
                                if (y['star']==1 and x.split('_')[0]==star)]
            updated_candidates_for_star = set()
            for tid in new_valid[star] :
                for table in new_valid[star][tid][col_out]:
                    updated_candidates_for_star.add(table)
            for star_node in star_nodes :
                if star_node not in updated_candidates_for_star and star_node.split('_')[0] == star:
                    is_star[star_node] = 0
            nx.set_node_attributes(tree,is_star,'star')
            bottom_up_prune(tree,list(tree.nodes())[0],None,is_star)
    return new_valid,tree_dict

def get_starred_set(valid) :
    ret = []
    for star in valid : 
        for tid in valid[star] : 
            l = get_all_possibilities(valid[star][tid],list(valid[star][tid].keys()))
            for i in l:
                if i not in ret : ret.append(i)
    return ret

def substitute_node(G,old_node,new_node):
    G.add_node(new_node)
    for node in G[old_node] :
        G.add_edge(new_node,node)
        G[new_node][node]['join'] = G[old_node][node]['join']
    G.remove_node(old_node)

def sort_stars(star):
    val = int(star.split('_')[1])
    return val

def merge_stars(star_list,tree_dict) :
    star_list.sort()
    star_list = sorted(star_list,key=sort_stars)
    star = star_list[0].split('_')[0]
    table_nos = '_'.join([x.split('_')[1] for x in star_list])
    merged_star = star + '_' + table_nos
    merged_tree = nx.Graph()
    merged_tree_cols = {}
    keys = list(tree_dict.keys())
    for i in range(len(keys)):
        col_out = keys[i]
        star = star_list[i]
        for col in tree_dict[col_out] :
            tree = tree_dict[col_out][col]
            if star in list(tree.nodes()): break
        tree_cols = nx.get_node_attributes(tree,'col')
        root = list(tree.nodes())[0]
        merged_tree_cols[root] = tree_cols[root].copy()
        if(star != root): 
            try :
                path = nx.Graph(tree.subgraph(list(nx.all_simple_paths(tree,root,star))[0]))
            except :
                print(root)
                print(star)
                draw_tree(tree)
                return None
        else : 
            path = nx.Graph(tree.subgraph(root))
            if(merged_star in merged_tree_cols): merged_tree_cols[merged_star] += tree_cols[root].copy()
            else : merged_tree_cols[merged_star] = tree_cols[root].copy()
        if(len(keys)>1): substitute_node(path,star,merged_star)
        merged_tree.add_nodes_from(path.nodes())
        merged_tree.add_edges_from(path.edges())
        for edge in path.edges() :
            node1 = edge[0]
            node2 = edge[1]
            merged_tree[node1][node2]['join'] = path[node1][node2]['join']
    nx.set_node_attributes(merged_tree,merged_tree_cols,'col')
    return merged_tree

def get_merge_list(tree_dict,table,prev_merge) :
    merge_list = get_valid(tree_dict,table,False)
    new_merge = cross_tuple_prune(merge_list,prev_merge,table)
    return new_merge

def initialize_tid_lists(tree, merge) :
    nodes = list(tree.nodes())
    tids = {}
    tables = set()
    for node in nodes : 
        tids[node] = set()
        table = node.split('_')[0]
        tables.add(table)
    for table in tables : 
        for tid in merge[table] :
            for col in merge[table][tid] :
                for candidate in merge[table][tid][col] :
                    if candidate in nodes:
                        tids[candidate].add(tid)
                        if(len(tids[candidate])>alpha) : 
                            tids[candidate] = set(['wc'])
                            break
    nx.set_node_attributes(tree,tids,'tid')

def merge(graph,node1,node2) :
    no = '_'.join(node1.split('_')[1:]) + '_' + '_'.join(node2.split('_')[1:])
    new_node = node1.split('_')[0] + '_' + no
    for node in graph[node1] :
        graph.add_edge(node,new_node)
        graph[node][new_node]['join'] = graph[node][node1]['join']
    for node in graph[node2] :
        graph.add_edge(node,new_node)
        graph[node][new_node]['join'] = graph[node][node2]['join']
    tids = nx.get_node_attributes(graph,'tid')
    col = nx.get_node_attributes(graph,'col')
    if(node1 in col and node2 in col) :
        col[new_node] = col[node1] + col[node2]
    elif(node1 in col) :
        col[new_node] = col[node1]
    elif(node2 in col) :
        col[new_node] = col[node2]
    if(tids[node1] == set(['wc']) and tids[node2] != set(['wc'])): new_node_tid = tids[node2]
    if(tids[node1] != set(['wc']) and tids[node2] == set(['wc'])): new_node_tid = tids[node1]
    else : new_node_tid = tids[node1].intersection(tids[node2])
    tids[new_node] = new_node_tid
    nx.set_node_attributes(graph,tids,'tid')
    nx.set_node_attributes(graph,col,'col')
    graph.remove_node(node1)
    graph.remove_node(node2)

def execute_query(query,conn) :
    dat = pd.read_sql_query(query, conn)
    return dat

def get_query_from_graph (graph,limit) :
    nodes = list(graph.nodes())
    query = " SELECT "
    col = nx.get_node_attributes(graph,'col')
    projected_tables = col.keys()
    for projected_table in projected_tables :
        for x in set(col[projected_table]):
            query += projected_table + '.' + x + " AS " + projected_table + '_' + x + " , "
    query = query.strip(", ")
    query += " FROM "
    for node in nodes :
        query += node.split('_')[0] + " " + node + " , "
    query = query.strip(", ")
    if(len(list(graph.edges()))):
        query += " WHERE "
        for edge in list(graph.edges()) :
            node1 = edge[0]
            node2 = edge[1]
            join = graph[node1][node2]["join"]
            table_1 = join[0].split('.')[0]
            if(table_1 == node1.split('_')[0]) :
                query += node1 + "." + join[0].split('.')[1] + "=" + node2 + "." + join[1].split('.')[1] + " and "
            elif (table_1 == node2.split('_')[0]) :
                query += node2 + "." + join[0].split('.')[1] + "=" + node1 + "." + join[1].split('.')[1] + " and "      
        query = query.strip(" and ")
    # query += " LIMIT %d" % (limit) 
    return query

def df_equals(query,conn,df) :
    df1 = pd.read_sql_query(query , conn)
    df2 = df
    df1 = df1.sort_index(axis=1)
    df2 = df2.sort_index(axis=1)
    print(df1)
    print(df2)
    vals_df1 = [set([str(j) for j in i]) for i in df1.values]
    vals_df2 = [set([str(j) for j in i]) for i in df2.values]
    if(len(df1.columns) != len(df2.columns)) : return False
    for row in vals_df2 :
        if row in vals_df1 :
            vals_df1.remove(row)
        else :
            del vals_df1
            del vals_df2
            return False
    del vals_df1
    del vals_df2
    return True
    # for i in range(len(vals_df1)) :
    #     for j in range(len(vals_df1[0])):
    #         if(str(vals_df1[i][j]) != str(vals_df2[i][j])):
    #             del vals_df1
    #             del vals_df2
    #             return False
    

def gen_lattice(star_graph,merge_list,df,conn):
    lattice = [star_graph]
    while(len(lattice)) :
        new_lattice = []
        while(len(lattice)):
            graph = lattice.pop()
            query = get_query_from_graph(graph,len(df.index)+1)
            if df_equals(query,conn,df):
                return query
            tids = nx.get_node_attributes(graph,'tid')
            nodes = list(graph.nodes())
            for i in range(len(nodes)):
                node1 = nodes[i]
                for j in range(i+1 , len(nodes)):
                    node2 = nodes[j]
                    table1 = node1.split('_')[0] 
                    table2 = node2.split('_')[0]
                    if table1 == table2 and (tids[node1] == set(['wc']) or tids[node2] == set(['wc']) or len(tids[node1].intersection(tids[node2]))!=0):
                        new_graph = nx.Graph(graph)
                        merge(new_graph,node1,node2)
                        is_isomorphic = False
                        for g in new_lattice :
                            if nx.is_isomorphic(g,new_graph):
                                is_isomorphic = True
                                break
                        if not is_isomorphic :
                            new_lattice.append(new_graph)
                        else : new_graph.clear()
            graph.clear()
        lattice = new_lattice

def initialize_black_list(cand_dict) :
    blacklist = {}
    for out_col in cand_dict :
        blacklist[out_col] = {}
        for col in cand_dict[out_col]:
            blacklist[out_col][col] = []
    return blacklist















        
