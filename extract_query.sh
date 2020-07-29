#!/bin/sh
python ComplexJoin/execute_query.py
python ComplexJoin/rqe.py
rm -rf Regal/src/com/sql/*.class
javac -cp .:/home/ananya/Documents/IISc/RQE/JARs/guava-16.0.1.jar:/home/ananya/Documents/IISc/RQE/JARs/mysql-connector-java-8.0.21.jar Regal/src/com/sql/*.java
cd Regal/src
java -Xmx2048m -cp .:/home/ananya/Documents/IISc/RQE/JARs/guava-16.0.1.jar:/home/ananya/Documents/IISc/RQE/JARs/mysql-connector-java-8.0.21.jar com.sql.BeginTree