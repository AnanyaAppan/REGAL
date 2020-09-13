package com.sql;

import java.sql.Connection;
import java.sql.DriverManager;

public class DbTree {
	public static Connection getConnection()
    {
           Connection conn = null;
           try
        {
            String userName = "root"; 
            String password = "*Rasika0507"; 
            //String userName = "root"; 
            //String password = "root";
            // String url = "jdbc:mysql://localhost/tpch_10m";       
            //String url = "jdbc:mysql://localhost/tpch_100m";
            String url = "jdbc:mysql://localhost/tpch100MB";
            Class.forName ("com.mysql.cj.jdbc.Driver").newInstance ();
            conn = DriverManager.getConnection (url, userName, password);
            System.out.println ("Database connection established!");
        }
        catch (Exception e)
        {
     	   e.printStackTrace();
            System.err.println ("Cannot connect to database server...");
            System.exit(0);
        }
        return conn;
    }
}
