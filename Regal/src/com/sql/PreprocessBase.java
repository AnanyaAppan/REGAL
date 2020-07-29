package com.sql;

import java.math.BigDecimal;
import static java.lang.Math.min;
// import java.sql.Connection;
// import java.sql.SQLException;
import java.sql.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.BiMap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
// import com.mysql.jdbc.ResultSet;
// import com.mysql.jdbc.ResultSetMetaData;
// import com.mysql.jdbc.Statement;

public class PreprocessBase {
	@SuppressWarnings("unchecked")
	public static <T> int makeDataStruct (Connection conn, String query, Map <Integer, List<T>> baseColumn, 
			Map<Integer, Map<String, List<Integer>>> baseFreq, BiMap<Integer, String> columnName,
			Map<Integer, String> columnClass,
			Map<Integer, List<String>> rowStore,
			Map<List<Integer>, List<Integer>> partialRow //,
			//Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable
			) 
	{
		// //Map<List<Integer>, Map<Integer, Multimap<Integer, T>>> latticeTable ,
		Statement stmt = null;
		int cardinality = 0;
		try {
			stmt = (Statement) conn.createStatement();
			ResultSet rs = (ResultSet) stmt.executeQuery(query);
			ResultSetMetaData rsmd = (ResultSetMetaData) rs.getMetaData();
			int columnsNumber = rsmd.getColumnCount();
			List<Integer> allColumn = new ArrayList<>();
	        for (int i=1 ; i <= columnsNumber ; ++i){
	        	columnName.put(i, rsmd.getColumnLabel(i));
	        	columnClass.put(i, rsmd.getColumnClassName(i));
	        	allColumn.add(i);
	        }
			Integer[] columnArray = allColumn.toArray(new Integer[allColumn.size()]);
			ArrayList<List<Integer>> treeCom = new ArrayList<>();
			for (int i=1; i <= columnArray.length ; ++i) {
				treeCom.addAll(combination(columnArray, i));	
			}
			//@@
			/**for (List<Integer> treenode : treeCom) {
				Map<List<String>, Multimap<Integer, T>> reallyMeteor = new HashMap<>();
				latticeTable.put(treenode, reallyMeteor);
			} */
			//@@
	        for (int i=1 ; i <= columnsNumber ; ++i) {
	        	baseColumn.put(i, new ArrayList<T>());
	        	Map<String, List<Integer>> meteor = new HashMap<>();
	        	baseFreq.put(i, meteor);
	        }
	        
	        
	        int size = 0;
	        while (rs.next()) {
	        	for (List<Integer> treenode : treeCom) {
	        		List<Integer> outskirt = new ArrayList<>(allColumn);
	        		outskirt.removeAll(treenode);
	        		if (outskirt.isEmpty()) {
	        			partialRow.put(treenode, null);
	        		} else {
	        			partialRow.put(treenode, outskirt);
	        		}
	        	}
	        	//@@
	        	/**for (List<Integer> treenode : treeCom) {
	        		List<T> tname = new ArrayList<>();
	        		for (int t : treenode) { 
	        			tname.add((T) rs.getObject(t).toString());
	        		}
	        		List<String> treename = (List<String>) tname;
	        		List<Integer> outskirt = new ArrayList<>(allColumn);
	        		outskirt.removeAll(treenode);
	        		//System.out.println(outskirt);
	        		Multimap<Integer, T> other = ArrayListMultimap.create();
	        		
	        		if (outskirt.isEmpty()) {
	        			Map<List<String>, Multimap<Integer, T>> setnull = latticeTable.get(treenode);
	        			setnull.put(treename, null);
	        		} else {
	        			
	        			for (int j : outskirt) {
	        				other.put(j, (T) rs.getObject(j));
	        			}
	        			if (latticeTable.get(treenode).isEmpty() || !latticeTable.get(treenode).containsKey(treename)) {
	        				latticeTable.get(treenode).put(treename, other);
	        			} else if (latticeTable.get(treenode).containsKey(treename)) {
	        				latticeTable.get(treenode).get(treename).putAll(other);
	        			} 
	        		}
	        	} */
	        	//@@
	        	List<String> rowList = new ArrayList<>();
	        	for (int i=1 ; i <= columnsNumber ; ++i){
	        		baseColumn.get(i).add((T) rs.getObject(i));
	        		String str = rs.getObject(i).toString();
	        		rowList.add(str);
	        		
	        		Map<String, List<Integer>> there = baseFreq.get(i);
	        		if (there.isEmpty()) {
	        			List<Integer> tupleID = new ArrayList<>();
	        			tupleID.add(size);
						there.put(str, tupleID); // when there is no any word inside the structure
	        		} else if (!there.isEmpty() && !there.containsKey(str)) {
	        			List<Integer> tupleID = new ArrayList<>();
	        			tupleID.add(size);
	        			there.put(str, tupleID); // update the words
	        		} else {
	        			List<Integer> tupleID = there.get(str);
	        			tupleID.add(size);
	        			there.put(str, tupleID); // update the indexes
	        		}
	        	}
	        	rowStore.put(size, rowList);
	        	size++;
	        }
	        cardinality = rowStore.size(); //int rowCount = rs.last() ? rs.getRow() : 0;
	         
		} catch (SQLException e ) { 
			System.err.println("Pretree.restable Problem reading properties file ");
	        e.printStackTrace();
		}
		return cardinality;
	}
	
	@SuppressWarnings("unchecked")
	public static <T> void mathematics (Map<Integer, String> baseClass, Map<Integer, List<T>> baseColumn,
			Map<Integer, MathClass<Integer>> inmath, Map<Integer, MathClass<BigDecimal>> domath) {
		for (int i : baseClass.keySet()) {
			if ( baseClass.get(i).contains("Integer")) {
				MathClass<Integer> mathin = new MathClass<>();
				List<Integer> type = (List<Integer>) baseColumn.get(i);
				Iterator<Integer> incom = (Iterator<Integer>) baseColumn.get(i).iterator();
				int inpos = 0; int inneg = 0;
				while (incom.hasNext()) {
					int com = incom.next();
					if (com-0 >= 0) {
						inpos += com;
					} else {
						inneg += com;
					}
				}
				mathin.setTotpos(inpos);
				mathin.setTotneg(inneg);
				int max = Collections.max(type);
				int min = Collections.min(type);
				mathin.setMax(max);
				mathin.setMin(min);
				inmath.put(i, mathin);
				
			} else if ( baseClass.get(i).contains("BigDecimal")) {
				List<BigDecimal> type = (List<BigDecimal>) baseColumn.get(i);
				
				BigDecimal max = Collections.max(type);
				BigDecimal min = Collections.min(type);
				Iterator<BigDecimal> docom = (Iterator<BigDecimal>) baseColumn.get(i).iterator();
				Double dopos = 0.0; BigDecimal bdpos = new BigDecimal(dopos);
				Double doneg = 0.0; BigDecimal bdneg = new BigDecimal(doneg);
				BigDecimal zero = new BigDecimal(0);
				while (docom.hasNext()) {
					BigDecimal com = docom.next();
					if (com.compareTo(zero)>= 0) {
						bdpos = bdpos.add(com);
					} else {
						bdneg = bdneg.add(com);
					}
				}
				MathClass<BigDecimal> mathdo = new MathClass<BigDecimal>(min, max, bdpos, bdneg);
				domath.put(i, mathdo);
			} else if (baseClass.get(i).contains("Long")) {
				MathClass<Integer> mathin = new MathClass<>();
				List<Long> type = (List<Long>) baseColumn.get(i);
				List<Integer> convert = new ArrayList<>();
				for (int k=0; k<type.size(); k++) {
					convert.add(type.get(k).intValue());
				}
				Iterator<Integer> incom = (Iterator<Integer>) convert.iterator();
				int inpos = 0; int inneg = 0;
				while (incom.hasNext()) {
					int com = incom.next();
					if (com-0 >= 0) {
						inpos += com;
					} else {
						inneg += com;
					}
				}
				mathin.setTotpos(inpos);
				mathin.setTotneg(inneg);
				int max = Collections.max(convert);
				int min = Collections.min(convert);
				mathin.setMax(max);
				mathin.setMin(min);
				inmath.put(i, mathin);
			}
		}
	}
	
	private static <T> Map<String, List<Integer>> getMapping(List<? extends T> list) {
		Map<String, List<Integer>> objPair = new HashMap<>();
		for (int i=0; i<list.size(); i++){
			if (objPair.containsKey(list.get(i))){
				List<Integer> v = objPair.get(list.get(i));
				v.add(i);
				objPair.put(list.get(i).toString(), v);
			}
			else {
				ArrayList<Integer> value = new ArrayList<>();
				value.add(i);
				objPair.put(list.get(i).toString(), value);
			}
		}
		//System.out.println("PreproBase.getMapping " + objPair);
		return objPair;
	}
	
	public static ArrayList<ArrayList<Integer>> combination(Integer[] oriArray, int K) {
		// TODO Auto-generated method stub
		ArrayList<ArrayList<Integer>> oriCombination = new ArrayList<>();
		int N = oriArray.length;
		if(K > N){
			System.out.println("PreproBase.combination Invalid input, K > N");
			return null;
		}
		c(N,K);
		int combination[] = new int[K];
		int r = 0;
		int index = 0;
		while(r >= 0){
			if(index <= (N + (r - K))){
				combination[r] = index;
				if (r == K-1){
					ArrayList<Integer> oriCombine = print(combination, oriArray);
					oriCombination.add(oriCombine);
					index++;
				}else{
					index = combination[r]+1;
					r++;
				}
			}else{
				r--;
				if(r > 0)
					index = combination[r]+1;
				else
					index = combination[0]+1;
			}
		}
		return oriCombination;
	}
	
	public static int c(int n, int r){
		int nf = fact(n);
		int rf = fact(r);
		int nrf = fact(n-r);
		if(nrf == 0) return 1;
		int npr = nf/nrf;
		int ncr = npr/rf;
		//System.out.println("PreproBase.c C(" + n + "," + r +") = " + ncr);
		return ncr;
	}
	
	public static int fact(int n){
		if(n == 0)
			return 1;
		else
			return n*fact(n-1);
	}
	
	public static ArrayList<Integer> print(int[] combination, Integer[] oriArray){
		ArrayList<Integer> oriCombine = new ArrayList<>();
		for(int i=0; i < combination.length; i++){
			oriCombine.add(oriArray[combination[i]]);
		}
		return oriCombine;
	}
	
	public static Set<Integer> commonListIndices(List<List<Integer>> getAllLists) {
		// TODO Auto-generated method stub
		if (getAllLists.size() == 1) {
			return Sets.newHashSet(getAllLists.get(0));
		} else {
			//reverseSizeOfLists(getAllLists);
			Set<Integer> result = Sets.newHashSet(getAllLists.get(0));
			for (List<Integer> lists : getAllLists){
				result = Sets.intersection(result, Sets.newHashSet(lists));
			}
			//System.out.println("PreproBase.commonListIndices " + result);
			return result;
		}	
	}
	
	private static <T> List<? extends List<T>> reverseSizeOfLists(List<? extends List<T>> getAllLists) {
		Collections.sort(getAllLists, new Comparator<List<T>>() {
			public int compare(List<T> o1, List<T> o2) {
				return Integer.compare(o2.size(), o1.size());
			}
		});
		return getAllLists;
	}
}
