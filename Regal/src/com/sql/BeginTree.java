package com.sql;

import java.math.BigDecimal;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner; 
import java.io.File;
import java.io.FileNotFoundException;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Multimap;

public class BeginTree {
	
	private static final long MEGABYTE = 1024L * 1024L;
	public static long bytesToMegabytes(long bytes) {
	    return bytes / MEGABYTE;
	}
	public static long count = 0;
	public static long nodecount = 0;
	public static <T extends Number & CharSequence & Comparable<T>> void main(String[] args) throws SQLException, FileNotFoundException {
		Connection conn = DbTree.getConnection();
		/***_________________________________________Base Table____________________________________***/
		
		// String baseTable = "SELECT L_SHIPMODE, L_QUANTITY, L_LINENUMBER, L_TAX FROM lineitem"; 

		File file = new File("../../extracted_join_query.txt");
		Scanner sc = new Scanner(file); 
		String baseTable = sc.nextLine();
		System.out.println(baseTable);
			
		long start_pre = System.currentTimeMillis();
		Map<Integer, List<T>> baseColumn = new HashMap<>();		
		Map<Integer, Map<String, List<Integer>>> baseFreq = new HashMap<>();
		
		//Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA = new HashMap<>(); //@@
		
		Map<Integer, List<String>> baseRowStore = new HashMap<>();
		Map<List<Integer>, List<Integer>> basePartialRow = new HashMap<>();
		BiMap<Integer, String> baseName = HashBiMap.create();
		Map<Integer, String> baseClass = new HashMap<>();
		baseClass.put(0, "count");
		//@@
		int baseCardinality = 
				PreprocessBase.makeDataStruct(conn, baseTable, baseColumn, baseFreq, baseName, baseClass, 
						baseRowStore, basePartialRow); //, baseSPGA); //@@
		System.out.println("base cardinality");
		
		Map<Integer, MathClass<Integer>> inmath = new HashMap<>();
		Map<Integer, MathClass<BigDecimal>> domath = new HashMap<>();
		PreprocessBase.mathematics(baseClass, baseColumn, inmath, domath);
		long end_pre = System.currentTimeMillis();
		System.out.println("PreproBase.makeDataStruct_cardinality " + baseCardinality);
		System.out.println("PreproBase.makeDataStruct_attribute " + baseName);
		System.out.println("PreproBase.makeDataStruct_class " + baseClass);
		double diff_pre = (end_pre - start_pre)/1e0; 
		System.out.println("__________Preprocessing base table " + diff_pre + " ms");
		
		/***___________________________________Query Output Table______________________________________***/
		
		// String resultTable = "SELECT L_SHIPMODE, COUNT(L_QUANTITY) FROM lineitem " 
		// 		+ "WHERE L_TAX >= 0.02 "
		// 		+ "AND L_QUANTITY < 46 "
		// 		+ "AND L_LINENUMBER < 6 " 
		// 		+ "GROUP BY L_SHIPMODE";

		file = new File("../../input_query.txt");
		sc = new Scanner(file); 
		String resultTable = "";
		while(sc.hasNextLine()) {
			resultTable += sc.nextLine() + " ";
		}

		System.out.println(resultTable);
		
		long start_on = System.currentTimeMillis();
		Map <Integer, List<T>> resultColumn = new HashMap<>();		
		Map<Integer, Map<String, List<Integer>>> resultFreq = new HashMap<>();	
		
		BiMap<Integer, String> columnName = HashBiMap.create();
		Map<Integer, String> columnClass = new HashMap<>();
		Map<Integer, List<String>> rowStore = new HashMap<>();
		Map<List<Integer>, List<Integer>> partialRow = new HashMap<>();
		//@@
		int cardinality = 
				PreprocessBase.makeDataStruct(conn, resultTable, resultColumn, resultFreq, 
						columnName, columnClass, 
						rowStore, partialRow); 	
		//for (int i : rowStore.keySet()) {
		//	System.out.println(i + " " + rowStore.get(i));
		//}	
		System.out.println("PreproBase.makeDataStruct_class_query " +columnClass);
		
		Map<Integer, MathClass<Integer>> inresult = new HashMap<>();
		Map<Integer, MathClass<BigDecimal>> doresult = new HashMap<>();
		PreprocessBase.mathematics(columnClass, resultColumn, inresult, doresult);
		for (int i = 1; i <= resultColumn.size(); i++) {
			if (columnClass.get(i).contains("BigDecimal")) {
				List<T> dots = resultColumn.get(i);
				if (changeable(dots)) {
					columnClass.put(i, "java.lang.Integer");
					MathClass<BigDecimal> change = doresult.get(i);
					MathClass<Integer> egnahc = new MathClass<>();
					egnahc.setMax(change.max.intValue());
					egnahc.setMin(change.min.intValue());
					egnahc.setTotpos(change.totpos.intValue());
					egnahc.setTotneg(change.totneg.intValue());
					inresult.put(i, egnahc);
				}
			}
		}
		System.out.println(columnClass);
		
		Multimap<Integer, Integer> mappingRB = MapBoth.pairwise(baseFreq, resultFreq, baseClass, columnClass);
		List<List<Integer>> rootProduct = MapBoth.productRoot(mappingRB);
		
		long end_on = System.currentTimeMillis();
		System.out.println("PreproBase.result_cardinality " + cardinality);
		System.out.println("PreproBase.result_column " + columnName);
		System.out.println("PreproBase.result_class " + columnClass);
		
		System.out.println("MapBoth.pairs_" + mappingRB);
		System.out.println("MapBoth.produce_lattice_root_" + rootProduct);
		
		double diff_on = (end_on - start_on)/1e0;
		System.out.println("__________Query output table cum Mapping table " + diff_on + " ms");
		
		Runtime runtime = Runtime.getRuntime();
	    // Run the garbage collector
	    runtime.gc();
	    // Calculate the used memory
	    long memory = runtime.totalMemory() - runtime.freeMemory();
	    System.out.println("Used memory is megabytes: "
	            + bytesToMegabytes(memory) + " MB");

		//Phase 1 : TopDown BottomUp GroupBy Candidates 
		long start_grpby = System.currentTimeMillis();
		Map<Map<String, Integer>, List<CombNode>> groupbynodes = 
			BuildLattice.buildLattice(mappingRB, columnName, resultColumn, rootProduct, baseFreq, cardinality, 
					baseRowStore, basePartialRow, rowStore, partialRow);
		long end_grpby = System.currentTimeMillis();
				
		for (Map<String, Integer> node : groupbynodes.keySet()) {
			System.out.println("Phase1.lattice_root " + node + " num_candidates " + groupbynodes.get(node).size() + "\n" + 
					groupbynodes.get(node));
		}
		
		double diff_grpby = (end_grpby - start_grpby)/1e0;
		System.out.println("__________diff_grpby " + diff_grpby + " ms");
		
		//Phase 2 : Aggregation Functions
		long start_aggre = System.currentTimeMillis();
		List<Pair<Integer, Multimap<Object, Integer>>> aggregation = 
				SingleConstraints.aggregation(baseFreq,resultFreq,baseClass,columnClass,
				inmath,domath,inresult,doresult,mappingRB,baseCardinality);
		long end_aggre = System.currentTimeMillis();
				
		for (Pair<Integer, Multimap<Object, Integer>> a : aggregation)
				System.out.println("SingleConstraints.aggregation " + a);
		
		Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA = new HashMap<>();
		Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> querySPGA = new HashMap<>();
		
				
		long start_granu = System.currentTimeMillis();
		Map<Map<String, Integer>, Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>>> whichtree = 
				GroupGranular.granularity(groupbynodes, aggregation, columnName, resultColumn,
						baseRowStore, basePartialRow, rowStore, partialRow, baseSPGA, querySPGA,
						baseColumn, baseFreq, resultFreq);
		long end_granu = System.currentTimeMillis(); 
				
		for (Map<String, Integer> key : whichtree.keySet()) {
			System.out.println("Phase2.treenode " + key);
			for (CombNode cn : whichtree.get(key).keySet()) {
				System.out.println("Phase2.combnode " + cn.attr + "\naggregation " + 
						whichtree.get(key).get(cn).size() + " " + whichtree.get(key).get(cn));
			}
		}
		
		double diff_aggre = (end_aggre - start_aggre)/1e0;
		System.out.println("__________diff_aggre " + diff_aggre + " ms");
		double diff_granu = (end_granu - start_granu)/1e0;
		System.out.println("__________diff_granu " + diff_granu + " ms");
		
		
		//Phase 3 :  Selection conditions
		
		long start_filter = System.currentTimeMillis();
		FilterSelection.filterDiscover(whichtree, columnName, baseFreq, querySPGA, baseSPGA, baseColumn, baseClass
				, basePartialRow);
		long end_filter = System.currentTimeMillis();
				
		double diff_filter = (end_filter - start_filter)/1e0;
		System.out.println("__________diff_filter " + diff_filter + " ms");	
		
		// Get the Java runtime
	    Runtime runtime2 = Runtime.getRuntime();
	    // Run the garbage collector
	    runtime2.gc();
	    // Calculate the used memory
	    long memory2 = runtime2.totalMemory() - runtime2.freeMemory();
	    System.out.println("Used memory2 is megabytes: "
	            + bytesToMegabytes(memory2) + " MB");
		
	}
	
	private static <T> boolean changeable(List<T> dots) {
		// TODO Auto-generated method stub
		int count = 0;
		for (T dot : dots) {
			if (!dot.toString().contains(".")) {
				count++;
			} else {
				break;
			}
		}
		if (count == dots.size()) return true;
		return false;
	}
}
