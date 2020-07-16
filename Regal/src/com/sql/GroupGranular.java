package com.sql;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.BiMap;
import com.google.common.collect.Multimap;

public class GroupGranular {
	public static <T extends Number & Comparable<T>> 
	Map<Map<String, Integer>, Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>>> granularity 
	(Map<Map<String, Integer>, List<CombNode>> groupbynodes, 
			List<Pair<Integer, Multimap<Object, Integer>>> aggregation, 
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase, //@@
			BiMap<Integer, String> columnName,
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable, //@@
			Map<Integer, List<T>> resultColumn,
			Map<Integer, List<String>> baseRowStore, Map<List<Integer>, List<Integer>> basePartialRow,
			Map<Integer, List<String>> rowStore, Map<List<Integer>, List<Integer>> partialRow,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> querySPGA,
			Map<Integer, List<T>> baseColumn, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<Integer, Map<String, List<Integer>>> resultFreq) 
	{
		Map<Map<String,Integer>, Map<CombNode, List<Pair<Integer, Multimap<Object,Integer>>>>> whichtree = 
				new HashMap<>();
		Map<CombNode, List<Pair<Integer, Multimap<Object,Integer>>>> whichnode = new HashMap<>();
		for (Map<String, Integer> treenode : groupbynodes.keySet()) {
			whichnode = new HashMap<>();
			List<CombNode> combnode = groupbynodes.get(treenode);
			whichTree(treenode, combnode, aggregation, columnName, whichnode, resultColumn, 
					baseRowStore, basePartialRow, rowStore, partialRow, baseSPGA, querySPGA,
					baseColumn, baseFreq, resultFreq);
			//@@    System.out.println("GroupGranular.granularity.whichtree_whichnode " + whichnode);
			if (!whichnode.isEmpty())
				whichtree.put(treenode, whichnode); /**update whichnode into whichtree */
		}
		
		Map<Map<String, Integer>, Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>>> groupaggregations = 
				new HashMap<>();
		List<Map<String, Integer>> listOfTree = new ArrayList<>(whichtree.keySet());
		
		for (Map<String,Integer> sgtree : listOfTree) {
			Map<CombNode, List<Pair<Integer, Multimap<Object, Integer>>>> whosenode = whichtree.get(sgtree);
			
			/**long multi_aggre_start = System.currentTimeMillis(); */
			Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>> replacenode = new HashMap<>();
			for (CombNode combnode : whosenode.keySet()) {
				System.out.println("Phase2.combnode " + combnode.attr + 
						" \naggregation(list of pair) " + whosenode.get(combnode));
				List<Integer> aggreattrs = new ArrayList<>(); 
				List<Multimap<Object, Integer>> aggrefunctions = new ArrayList<>();
				List<Pair<Integer, Multimap<Object, Integer>>> whoseaggre = whosenode.get(combnode);
				for (Pair<Integer, Multimap<Object, Integer>> peraggre : whoseaggre) {
					aggreattrs.add(peraggre.getFirst());
					aggrefunctions.add(peraggre.getSecond());
				}
				List<List<Pair<Object, Integer>>> totalcart = new ArrayList<>();
				for (int i = 0; i < aggrefunctions.size(); i++) {
					List<Pair<Object, Integer>> cartesianjuga = new ArrayList<>();
					Multimap<Object, Integer> perattr = aggrefunctions.get(i);
					for (Object fun : perattr.keySet()) {
						List<Integer> bids = (List<Integer>) perattr.get(fun);
						for (int b : bids) {
							Pair<Object, Integer> agg = new Pair<>(fun, b);
							cartesianjuga.add(agg);
						}
					}
					totalcart.add(cartesianjuga);
				}
				List<List<Pair<Object, Integer>>> hasilcart = cartesianProduct(totalcart);
				//System.out.println("AggreCoulmns_" + aggreattrs + "_num_of_carts_" + hasilcart.size());
				
				List<Map<Integer, Pair<Object, Integer>>> replaceaggre = new ArrayList<>();
				for (List<Pair<Object, Integer>> hasil : hasilcart) {
					Map<Integer, Pair<Object, Integer>> colaggre = new HashMap<>();
					List<Pair<Object, Integer>> respository = new ArrayList<>();
					endFunction:
					if (aggreattrs.size() > 1) {
						for (int m=0; m<aggreattrs.size()-1; m++) {
							for (int n=m+1; n<aggreattrs.size(); n++) {
								if (hasil.get(m).getSecond() == hasil.get(n).getSecond()) {
									if (hasil.get(m).getFirst().toString() == hasil.get(n).getFirst().toString()) {
										// System.out.println("crash aggregation " + hasil.get(m)); 
										break endFunction;
									} else {
									
										if (!multiConstraint(hasil.get(n).getFirst(), aggreattrs.get(n), 
												hasil.get(m).getFirst(), aggreattrs.get(m), resultColumn)) {
											// System.out.println("multi-constraint " + hasil.get(n) + " " + hasil.get(m)); 
											break endFunction;
										} 
									}
								}
							}
							respository.add(hasil.get(m));
						}
						if (!respository.isEmpty()) {
							respository.add(hasil.get(hasil.size()-1));
						}
					} else {
						respository.add(hasil.get(0));
					}
					//System.out.println(respository);
					if (respository.size() == aggreattrs.size()) {
						//restore it 
						for (int a = 0; a < aggreattrs.size(); a++) {
							colaggre.put(aggreattrs.get(a), respository.get(a));
						}
					}
					if (!colaggre.isEmpty())
						replaceaggre.add(colaggre);
				}
				if (!replaceaggre.isEmpty())
					replacenode.put(combnode, replaceaggre);
			}
			/**long multi_aggre_end = System.currentTimeMillis();
			long diff = multi_aggre_end - multi_aggre_start;
			System.out.println("time for multi_aggregation " + diff); */
			groupaggregations.put(sgtree, replacenode);
		}
		
		return groupaggregations;
	}
	
	public static <T extends Number & Comparable<T>> void whichTree (Map<String, Integer> treenode, 
			List<CombNode> combnode, List<Pair<Integer, Multimap<Object, Integer>>> aggregation, 
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase, //@@ 
			BiMap<Integer, String> columnName,
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable, //@@
			Map<CombNode, List<Pair<Integer, Multimap<Object,Integer>>>> whichnode, 
			Map<Integer, List<T>> resultColumn,
			Map<Integer, List<String>> baseRowStore, Map<List<Integer>, List<Integer>> basePartialRow,
			Map<Integer, List<String>> rowStore, Map<List<Integer>, List<Integer>> partialRow,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> querySPGA,
			Map<Integer, List<T>> baseColumn, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<Integer, Map<String, List<Integer>>> resultFreq) 
	{
		//Set<Integer> treeBaseID = new HashSet<Integer>(treenode.values());
		
		//Generate resultIndex---the query table columns that are possibly aggregated
		BiMap<String, Integer> columnIndex = columnName.inverse();
		List<Integer> resultIndex = new ArrayList<Integer>(columnName.keySet());
		Collections.sort(resultIndex);
		for (Pair<Integer, Multimap<Object, Integer>> checkrid : aggregation) {
			if (checkrid.getSecond().isEmpty()) {
				resultIndex.remove(checkrid.getFirst());
			}
		}
		//System.out.println("GroupGranular.whichTree.resultIndex " + resultIndex);
		Iterator<Integer> loop = resultIndex.iterator();
		Set<CombNode> deleteNode = new HashSet<>();
		while (loop.hasNext() && !combnode.isEmpty()) {
			int t = loop.next();
			String columnname = columnName.get(t);
			//System.out.println("GroupGranular.whichTree_columnname " + columnname);
			
			//@@
			whichNode(columnname, combnode, aggregation, treenode, 
					columnIndex, resultColumn, resultIndex, whichnode, deleteNode,
					baseRowStore, basePartialRow, rowStore, partialRow, baseSPGA, querySPGA,
					baseColumn, baseFreq, resultFreq);
			
			//System.out.println("GroupGranular.whichTree_whichnode " + whichnode);
		}
	}
	
	public static <T extends Number & Comparable<T>> void whichNode (String columnname, List<CombNode> combnode, 
			List<Pair<Integer, Multimap<Object, Integer>>> aggregation,
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase, //@@
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable, //@@ 
			Map<String, Integer> treenode,
			BiMap<String, Integer> columnIndex, Map<Integer, List<T>> resultColumn, List<Integer> resultIndex,
			Map<CombNode, List<Pair<Integer, Multimap<Object,Integer>>>> whichnode, Set<CombNode> deleteNode,
			Map<Integer, List<String>> baseRowStore, Map<List<Integer>, List<Integer>> basePartialRow,
			Map<Integer, List<String>> rowStore, Map<List<Integer>, List<Integer>> partialRow,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> querySPGA,
			Map<Integer, List<T>> baseColumn, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<Integer, Map<String, List<Integer>>> resultFreq) 
	{
		List<CombNode> candnode = new ArrayList<>();
		if (!combnode.isEmpty()) {
			for (CombNode cn : combnode) {
				if (!cn.getAttr().contains(columnname)) {
					candnode.add(cn);
				}
			}
		}
		
		for (Iterator<CombNode> w = candnode.iterator(); w.hasNext();) {
			CombNode cn = w.next();
			// System.out.println("deleteNode_" + deleteNode + " currentNode_" + candnode + " runNode_" + cn);	
			if (deleteNode.isEmpty() || !deleteNode.contains(cn)) {
				//@@
				whichnodecont(columnname, cn, aggregation,
						treenode, columnIndex, resultColumn, resultIndex, whichnode, deleteNode, combnode,
						baseRowStore, basePartialRow, rowStore, partialRow, baseSPGA, querySPGA,
						baseColumn, baseFreq, resultFreq);
			} else {
				// System.out.println("delete_node_" + cn);
				combnode.remove(cn);
				w.remove();
			}
		}
	}
	
	public static <T extends Number & Comparable<T>> void whichnodecont (String columnname, CombNode cn, 
			List<Pair<Integer, Multimap<Object, Integer>>> aggregation,
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase, //@@
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable, //@@
			Map<String, Integer> treenode,
			BiMap<String, Integer> columnIndex, Map<Integer, List<T>> resultColumn, List<Integer> resultIndex,
			Map<CombNode, List<Pair<Integer, Multimap<Object,Integer>>>> whichnode, Set<CombNode> deleteNode, 
			List<CombNode> combnode,
			Map<Integer, List<String>> baseRowStore, Map<List<Integer>, List<Integer>> basePartialRow,
			Map<Integer, List<String>> rowStore, Map<List<Integer>, List<Integer>> partialRow,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> querySPGA,
			Map<Integer, List<T>> baseColumn, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<Integer, Map<String, List<Integer>>> resultFreq) 
	{
		Set<Integer> removeKey = new HashSet<Integer>(resultIndex);
		
		ArrayList<Integer> combinaRID = new ArrayList<>();
		ArrayList<Integer> combinaBID = new ArrayList<>();
					
		ArrayList<String> cname = cn.getAttr();
		for (String s : cname) {
			combinaBID.add(treenode.get(s));
			combinaRID.add(columnIndex.get(s));
			removeKey.remove(columnIndex.get(s));
		}
					
		Collections.sort(combinaBID);
		System.out.println("Grouping_" + cname + "_aggreColumns_" + removeKey);
		long s = System.currentTimeMillis();
		if (!querySPGA.keySet().contains(combinaRID)) {
			List<Integer> getQueryCols = partialRow.get(combinaRID);
			//System.out.println("getQueryCols " + getQueryCols);
			ReadDatabase.readDatabase(querySPGA, rowStore, combinaRID, getQueryCols, resultColumn, resultFreq);
		}
		Map<List<String>, Multimap<Integer, T>> mapRID = querySPGA.get(combinaRID);
		//reorder the mapRID to complement mapBID			
		Map<List<String>, Multimap<Integer, T>> reorderRID = new HashMap<>();
		for (List<String> ruple : mapRID.keySet()) {
			List<String> retuple = new ArrayList<>();
			for (int i=0; i<cn.attr.size(); i++) {
				String r = cn.attr.get(i);
				int y = treenode.get(r);
				int z = combinaBID.indexOf(y);
				retuple.add(ruple.get(z));
			}
			reorderRID.put(retuple, mapRID.get(ruple));
		}
		List<List<String>> skipBase = new ArrayList<>(reorderRID.keySet());
		if (!baseSPGA.keySet().contains(combinaBID)) {
			List<Integer> getBaseCols = basePartialRow.get(combinaBID); //@@@@
			//List<Integer> getBaseCols = new ArrayList<>(removeKey);
			//System.out.println("getBaseCols " +getBaseCols);
			ReadDatabase.readSkipDatabase(baseSPGA, baseRowStore, combinaBID, 
					getBaseCols, baseColumn, baseFreq, skipBase);
		}
		Map<List<String>, Multimap<Integer, T>> mapBID = baseSPGA.get(combinaBID);
		
		long e = System.currentTimeMillis();
		double d = (e-s)/1e0;
		System.out.println("Read_partial_data_" + d + "_ms");
		//System.out.println(mapRID);
		
		/////Map<List<String>, Multimap<Integer, T>> mapBID = latticeBase.get(combinaBID);
		//scan base table for rows
		/////Map<List<String>, Multimap<Integer, T>> mapRID = latticeTable.get(combinaRID); 
		//scan query table for rows
					
		//System.out.println("GroupGranular.whichnode_reorderRID " + reorderRID.keySet());
		
		List<Pair<Integer, Multimap<Object, Integer>>> listOfNewPair = new ArrayList<>();
		Map<Object, Pair<Integer, Integer>> sumAvgCount = new HashMap<>();
		outer:
		for (Pair<Integer, Multimap<Object, Integer>> perColumn : aggregation) {
			long findstart = System.currentTimeMillis();
			if (removeKey.contains(perColumn.getFirst())) {
				int rid = perColumn.getFirst();
				Multimap<Object, Integer> getFunc = perColumn.getSecond();
				System.out.println("Column_id_" + rid + " Aggregations_" + getFunc);
				Multimap<Object, Integer> newMulti = ArrayListMultimap.create();
				for (Object fun : getFunc.keySet()) {
					//endFunction:
					for (int bid : getFunc.get(fun)) {
						endFunction:
						if (!listOfNewPair.isEmpty()) {
							for (Pair<Integer, Multimap<Object, Integer>> subNewPair : listOfNewPair) {
								if (subNewPair.getSecond().size() == 1 && subNewPair.getSecond().containsValue(bid)){
									List<Object> getFunction = new ArrayList<Object>(subNewPair.getSecond().keySet());
									
									if (fun.toString() == getFunction.get(0).toString()) {
										System.out.println("overlap_aggregations_" + fun.toString() + " " + bid);
										break endFunction;
									} else {
										if (!multiConstraint(fun, rid, getFunction.get(0), subNewPair.getFirst(), 
												resultColumn))
											System.out.println("multi-constraint_" + fun + " " + 
												getFunction.get(0) + " " + bid);
											break endFunction;
									}
								}
							}
						} 
						//@@
						if (!switchDemo(fun, reorderRID, mapBID, bid, rid, cn, sumAvgCount)) { 
							
							newMulti.put(fun, bid);
						}
					}
				}
				if (newMulti.isEmpty()) {
					deleteNode.addAll(cn.parents);
					combnode.remove(cn);
					break outer;
				} else {
					Pair<Integer, Multimap<Object, Integer>> newPair = 
							new Pair<Integer, Multimap<Object, Integer>>(rid, newMulti);
					listOfNewPair.add(newPair);
					if (newMulti.size() == 1 && (newMulti.keySet().contains("avg") || 
							newMulti.keySet().contains("sum") || 
							newMulti.keySet().contains("count"))) 
					{
									//sumAvgCount.add(newPair);
						List<Object> sac = new ArrayList<Object>(newMulti.keySet());
						List<Integer> bid = (List<Integer>) newMulti.get(sac.get(0));
						Pair<Integer, Integer> ridbid = new Pair<>(rid, bid.get(0)); 
						sumAvgCount.put(sac.get(0), ridbid);
					}
				}
			}
			long findend = System.currentTimeMillis();
			double diff = (findend - findstart)/1e0;
			//System.out.println("diff_time" + diff + "_current_column_" + perColumn);
		}
		int table = listOfNewPair.size();
		if (table == removeKey.size()) {
			whichnode.put(cn, listOfNewPair);
						//System.out.println("GroupGranular.whichnode " + cn.attr + " " + listOfNewPair);	
		} else {
			//deleteNode.addAll(cn.parents);
			//System.out.println("deleteNode " + deleteNode);
		}
		combnode.remove(cn);
	}
		

	private static <T extends Number & Comparable<T>> boolean multiConstraint(Object fun, int rid, Object object,
			int first, Map<Integer, List<T>> resultColumn) {
		/**long multi_aggre_start = System.currentTimeMillis(); */
		boolean valid = true;
		Object[] positiveSequence = new Object[] {"min", "avg", "max", "sum" };
		ArrayList<Object> positiveDirection = new ArrayList<Object>(Arrays.asList(positiveSequence));
		List<T> current = resultColumn.get(rid);
		List<T> history = resultColumn.get(first);
		Iterator<T> curr = current.iterator();
		Iterator<T> hist = history.iterator();
		if (positiveDirection.indexOf(fun) > positiveDirection.indexOf(object)) {
			while (curr.hasNext()) {
				T c = curr.next();
				T h = hist.next();
				//System.out.println(c + " " + h);
				if (c.compareTo(h) < 0) 
					valid = false;
					break;
			}
		} else {
			while (curr.hasNext()) {
				T c = curr.next();
				T h = hist.next();
				if (c.compareTo(h) > 0)
					valid = false;
					break;
			}
		}
		/**long multi_aggre_end = System.currentTimeMillis();
		long diff = multi_aggre_end - multi_aggre_start;
		System.out.println("multi_aggre_time " + diff); */
		return valid;
	}

	private static <T extends Number & Comparable<T>> boolean switchDemo(Object fun,
			Map<List<String>, Multimap<Integer, T>> mapRID, Map<List<String>, Multimap<Integer, T>> mapBID,
			int bid, int rid, CombNode cn, Map<Object, Pair<Integer, Integer>> sumAvgCount) {
		// TODO Auto-generated method stub
		boolean bool = false;
		switch((String) fun) {
		case "count":
			if (!gotoCount(mapRID, mapBID, rid, cn, sumAvgCount))  
				bool = true;
			break;
		case "max":
			if (!gotoMax(mapRID, mapBID, rid, cn, bid, sumAvgCount))
				bool = true;
			break;
		case "min":
			if (!gotoMin(mapRID, mapBID, rid, cn, bid, sumAvgCount))
				bool = true;
			break;
		case "sum":
			if (!gotoSum(mapRID, mapBID, rid, cn, bid, sumAvgCount))
				bool = true;
			break;
		case "avg":
			if (!gotoAvg(mapRID, mapBID, rid, cn, bid, sumAvgCount))
				bool = true;
			break;
		case "median":
			if (!gotoAvg(mapRID, mapBID, rid, cn, bid, sumAvgCount))
				bool = true;
			break;
		default:
			System.out.println("GroupGranular invalid");
			break;
		}
		return bool;
	}
	
	private static <T extends Number & Comparable<T>> boolean gotoAvg(Map<List<String>, Multimap<Integer, T>> mapRID, 
			Map<List<String>, Multimap<Integer, T>> mapBID,
			int rid, CombNode cn, int bid, Map<Object, Pair<Integer, Integer>> sumAvgCount) {
		// TODO Auto-generated method stub
		boolean valid = true;
		ArrayList<List<String>> inferFreq = new ArrayList<>();
		boolean statment = false;
		if (sumAvgCount.size() > 1 && sumAvgCount.keySet().contains("avg") && sumAvgCount.keySet().contains("sum")) {
			if (sumAvgCount.get("avg").getSecond() == sumAvgCount.get("sum").getSecond()) {
				statment = true;
			}
		}
		
		List<List<String>> listRID = new ArrayList<>(mapRID.keySet());
		Iterator<List<String>> iterListRID = listRID.iterator();
		outer:
		while (iterListRID.hasNext()) {
			List<String> tupleRID = iterListRID.next();
			//System.out.println(tupleRID);
			if (mapBID.keySet().contains(tupleRID) && mapBID.get(tupleRID)!= null) {
				List<T> ridValue = (List<T>) mapRID.get(tupleRID).get(rid);
				T ridval = ridValue.get(0);
				List<T> bidValue = (List<T>) mapBID.get(tupleRID).get(bid);
				if (bidValue.size() == 0) { valid = false; break outer; }
				double groupsum = 0.0;
				for (int i = 0; i < bidValue.size(); i++) {
					groupsum += bidValue.get(i).doubleValue();
				}
				double groupavg = groupsum / bidValue.size();
			//System.out.println("SwitchDemo.gotoAvg_groupavg "+ bidValue.size()+" " + groupavg + " compare " +
			//	ridval + " " + (int) Math.round(groupavg-ridval.doubleValue()));
			//System.out.println("SwitchDemo.gotoAvg " + cn.getInferFrequency());
			//int zero = (int) Math.round(groupavg-ridval.doubleValue());
				BigDecimal value = new BigDecimal(groupavg-ridval.doubleValue());
				value = value.setScale(2, RoundingMode.HALF_DOWN);
				if (statment == true) {
					int avid = sumAvgCount.get("avg").getFirst();
					int suid = sumAvgCount.get("sum").getFirst();
					List<T> avidValue = (List<T>) mapRID.get(tupleRID).get(avid);
					List<T> suidValue = (List<T>) mapRID.get(tupleRID).get(suid);
					BigDecimal avidval = (BigDecimal) avidValue.get(0);
					BigDecimal suidval = (BigDecimal) suidValue.get(0);
					int ans = (int) (Math.round(suidval.doubleValue() / avidval.doubleValue()));
					//System.out.println("gotoAvg_supposecount " + ans + " BaseGroupSize " + bidValue.size() + " " + cn);
					if (ans > bidValue.size()) {  //havent count filter, so BaseTable should be bigger or equal 
						valid = false;
						break outer;
					} else if (ans == bidValue.size()) {
						if (value.compareTo(BigDecimal.ZERO) !=0) {
							valid = false;
							break outer;
						} else {
							inferFreq.add(tupleRID);
							cn.isFrequency = true;
							cn.setInferFrequency(inferFreq);
						}
					}
				} else {
					if (cn.isFrequency == true) {
						if (cn.getInferFrequency().contains(tupleRID) && value.compareTo(BigDecimal.ZERO) != 0) {
							valid = false;
							break outer;
						} /** else if (value.compareTo(BigDecimal.ZERO) == 0) {
							inferFreq.add(tupleRID);
						} */
					} else if (Collections.min(bidValue).compareTo(ridval) > 0 || 
							ridval.compareTo(Collections.max(bidValue)) > 0) 
					{
						valid = false;
						break outer;
					}
					
					/***
					else if (!cn.getInferFrequency().contains(tupleRID) && value.compareTo(BigDecimal.ZERO) == 0) {
						inferFreq.add(tupleRID);
					} ***/
				} 
			} else {
				valid = false;
				break outer;
			}
		}
		/**if (valid == true) {
			cn.isFrequency = true;
			cn.setInferFrequency(inferFreq);
		}**/ 
		return valid;
	}

	private static <T extends Number & Comparable<T>> boolean gotoSum(Map<List<String>, Multimap<Integer, T>> mapRID, 
			Map<List<String>, Multimap<Integer, T>> mapBID,
			int rid, CombNode cn, int bid, Map<Object, Pair<Integer, Integer>> sumAvgCount) {
		
		boolean valid = true;
		ArrayList<List<String>> inferFreq = new ArrayList<>();
		boolean statment = false;
		if (sumAvgCount.size() > 1 && sumAvgCount.keySet().contains("avg") && sumAvgCount.keySet().contains("sum")) {
			if (sumAvgCount.get("avg").getSecond() == sumAvgCount.get("sum").getSecond()) {
				statment = true;
			}
		}
		
		List<List<String>> listRID = new ArrayList<>(mapRID.keySet());
		Iterator<List<String>> iterListRID = listRID.iterator();
		outer:
		while (iterListRID.hasNext()) {
			List<String> tupleRID = iterListRID.next();
			if (mapBID.keySet().contains(tupleRID) && mapBID.get(tupleRID) != null) {
				List<T> ridValue = (List<T>) mapRID.get(tupleRID).get(rid);
				T ridval = ridValue.get(0);
				List<T> bidValue = (List<T>) mapBID.get(tupleRID).get(bid);
				
				if (bidValue.size() == 0) { valid = false; break outer; }
			
				boolean allpositiveBor = true;
				double tempsum = 0.0;
				for (int i = 0; i < bidValue.size(); i++) {
					tempsum += bidValue.get(i).doubleValue();
					if (bidValue.get(i).equals(0)) {
						allpositiveBor = false;
					}
				}
			
				//int zero = (int) Math.round(tempsum-ridval.doubleValue());
				BigDecimal value = new BigDecimal(tempsum-ridval.doubleValue());
				value = value.setScale(2, RoundingMode.HALF_DOWN);
				
				if (statment == true) {
					int avid = sumAvgCount.get("avg").getFirst();
					int suid = sumAvgCount.get("sum").getFirst();
					List<T> avidValue = (List<T>) mapRID.get(tupleRID).get(avid);
					List<T> suidValue = (List<T>) mapRID.get(tupleRID).get(suid);
					BigDecimal avidval = (BigDecimal) avidValue.get(0);
					BigDecimal suidval = (BigDecimal) suidValue.get(0);
					
					int ans = (int) (Math.round(suidval.doubleValue() / avidval.doubleValue()));
					//System.out.println("gotoSum_supposecount "+ans+"<=BaseGroupSize"+bidValue.size()+" "+cn);
					if (ans > bidValue.size()) {  //havent count filter, so BaseTable should be bigger or equal 
						valid = false;
						break outer;
					} else if (ans == bidValue.size()) {
						if (value.compareTo(BigDecimal.ZERO) != 0) {
							valid = false;
							break outer;
						} else {
							inferFreq.add(tupleRID);
							//cn.isFrequency = true;
							//cn.setInferFrequency(inferFreq);
						}
					}
				}
				else {
					//System.out.println("SwitchDemo.gotoSum_tempsum"+tempsum+" compare "+ridval.getClass()+ " " +value);
					if (cn.isFrequency == true) {
						if (cn.getInferFrequency().contains(tupleRID) && value.compareTo(BigDecimal.ZERO) != 0) {
							valid = false;
							break outer;
						} else if (value.compareTo(BigDecimal.ZERO) == 0) {
							inferFreq.add(tupleRID);
							//cn.isFrequency = true;
							//cn.setInferFrequency(inferFreq);
						}
					} else if (ridval.getClass().getSimpleName().contains("Long")) {
						int change = ridval.intValue();
						int test = (Integer) Collections.min(bidValue);
						if (test > change || value.compareTo(BigDecimal.ZERO) < 0) {
							valid = false;
							break outer;
						}
					} else if ( bidValue.getClass().getSimpleName().contains("BigDecimal") && 
							(Collections.min(bidValue).compareTo(ridval) > 0 || value.compareTo(BigDecimal.ZERO) < 0)) 
					{
						valid = false;
						break outer;
					} else if ( bidValue.getClass().getSimpleName().contains("Integer") && 
							(Collections.min(bidValue).compareTo(ridval) > 0 || value.compareTo(BigDecimal.ZERO) < 0)) 
					{
						valid = false;
						break outer;
					} else if (!cn.getInferFrequency().contains(tupleRID) && value.compareTo(BigDecimal.ZERO) == 0 
							&& allpositiveBor) 
					{
						inferFreq.add(tupleRID);
						//cn.isFrequency = true;
						//cn.setInferFrequency(inferFreq);
					} 
				} 
			} else {
				valid = false;
				break outer;
			}
		} 
		if (valid == true) {
			cn.isFrequency = true;
			cn.setInferFrequency(inferFreq);
		}
		return valid;
	}

	private static <T extends Number & Comparable<T>> boolean gotoMin(Map<List<String>, Multimap<Integer, T>> mapRID, 
			Map<List<String>, Multimap<Integer, T>> mapBID,
			int rid, CombNode cn, int bid, Map<Object, Pair<Integer, Integer>> sumAvgCount) 
	{
		// TODO Auto-generated method stub
		boolean valid = true;
		
		boolean statment = false;
		if (sumAvgCount.size() > 1 && sumAvgCount.keySet().contains("avg") && sumAvgCount.keySet().contains("sum")) {
			if (sumAvgCount.get("avg").getSecond() == sumAvgCount.get("sum").getSecond()) {
				statment = true;
			}
		}
		
		List<List<String>> listRID = new ArrayList<>(mapRID.keySet());
		//System.out.println(listRID);
		Iterator<List<String>> iterListRID = listRID.iterator();
		outer:
		while (iterListRID.hasNext()) {
			List<String> tupleRID = iterListRID.next();
			if (mapBID.containsKey(tupleRID) && mapBID.get(tupleRID) != null) {
				List<T> ridValue = (List<T>) mapRID.get(tupleRID).get(rid);
				T ridval = ridValue.get(0);
				
				List<T> bidValue = (List<T>) mapBID.get(tupleRID).get(bid);
				
				
				//List<T> bidValue = (List<T>) mapBID.get(tupleRID).get(bid);
				//System.out.println( mapBID.get(tupleRID));
				
				if (statment == true) {
					int avid = sumAvgCount.get("avg").getFirst();
					int suid = sumAvgCount.get("sum").getFirst();
					List<T> avidValue = (List<T>) mapRID.get(tupleRID).get(avid);
					List<T> suidValue = (List<T>) mapRID.get(tupleRID).get(suid);
					BigDecimal avidval = (BigDecimal) avidValue.get(0);
					BigDecimal suidval = (BigDecimal) suidValue.get(0);
					int ans = (int) (Math.round(suidval.doubleValue() / avidval.doubleValue()));
					//System.out.println("gotoMin_supposecount " + ans + " BaseGroupSize " + bidValue.size() + " " + cn);
					if (ans > bidValue.size()) {  //havent count filter, so BaseTable should be bigger or equal 
						valid = false;
						break outer;
					} else if (ans == bidValue.size() && Collections.min(bidValue).compareTo(ridval)!=0) {
						valid = false;
						break outer;
					}
				}
				//System.out.println("SwitchDemo.gotoMin_min " + Collections.min(bidValue) + " compare " + ridval);
				else if (cn.isFrequency == true) {
					if (cn.getInferFrequency().contains(tupleRID) && Collections.min(bidValue).compareTo(ridval)!=0) {
						valid = false;
						break outer;
					} 
				} 
				else if (!bidValue.contains(ridval) || Collections.min(bidValue).compareTo(ridval) > 0) {
					valid = false;
					break outer;
				}
			} else {
				valid = false;
				break outer;
			}
		}
		return valid;
	}

	private static <T extends Number & Comparable<T>> boolean gotoMax(Map<List<String>, Multimap<Integer, T>> mapRID, 
			Map<List<String>, Multimap<Integer, T>> mapBID,
			int rid, CombNode cn, int bid, Map<Object, Pair<Integer, Integer>> sumAvgCount) 
	{
		// TODO Auto-generated method stub
		boolean valid = true;
		
		boolean statment = false;
		if (sumAvgCount.size() > 1 && sumAvgCount.keySet().contains("avg") && sumAvgCount.keySet().contains("sum")) {
			if (sumAvgCount.get("avg").getSecond() == sumAvgCount.get("sum").getSecond()) {
				statment = true;
			}
		}
		
		List<List<String>> listRID = new ArrayList<>(mapRID.keySet());
		Iterator<List<String>> iterListRID = listRID.iterator();
		outer:
		while (iterListRID.hasNext()) {
			List<String> tupleRID = iterListRID.next();
			if (mapBID.keySet().contains(tupleRID) && mapBID.get(tupleRID)!=null) {
				List<T> ridValue = (List<T>) mapRID.get(tupleRID).get(rid);
				T ridval = ridValue.get(0);
				List<T> bidValue = (List<T>) mapBID.get(tupleRID).get(bid);
				
				if (statment == true) {
					int avid = sumAvgCount.get("avg").getFirst();
					int suid = sumAvgCount.get("sum").getFirst();
					List<T> avidValue = (List<T>) mapRID.get(tupleRID).get(avid);
					List<T> suidValue = (List<T>) mapRID.get(tupleRID).get(suid);
					BigDecimal avidval = (BigDecimal) avidValue.get(0);
					BigDecimal suidval = (BigDecimal) suidValue.get(0);
					int ans = (int) (Math.round(suidval.doubleValue() / avidval.doubleValue()));
		//System.out.println("gotoMax_supposecount " + ans + " BaseGroupSize " + bidValue.size() + " " + cn);
					if (ans > bidValue.size()) {
						valid = false;
						break outer;
					} else if (ans == bidValue.size() && Collections.max(bidValue).compareTo(ridval)!=0) {
						valid = false;
						break outer;
					}
				}
				
		//System.out.println("SwitchDemo.gotoMax_max "+tupleRID+" "+Collections.max(bidValue)+" compare "+ridval);
				else if (cn.isFrequency == true) {
					if (cn.getInferFrequency().contains(tupleRID) && Collections.max(bidValue).compareTo(ridval) != 0) {
						//System.out.println("why");
						valid = false;
						break outer;
					}
				} else if (!bidValue.contains(ridval) || Collections.max(bidValue).compareTo(ridval) < 0) {
					valid = false;
					break outer;
				}
			} else {
				valid = false;
				break outer;
			}
			
		}
		return valid;
	}

	private static <T> boolean gotoCount(Map<List<String>, Multimap<Integer, T>> mapRID, 
			Map<List<String>, Multimap<Integer, T>> mapBID,
			int rid, CombNode cn, Map<Object, Pair<Integer, Integer>> sumAvgCount) {
		boolean valid = true;
		
		ArrayList<List<String>> inferFreq = new ArrayList<>();
		
		//check the sumAvgCount size > 1 and contains "avg" and "sum" on same BID
		//if statement true, 
		boolean statment = false;
		if (sumAvgCount.size() > 1 && sumAvgCount.keySet().contains("avg") && sumAvgCount.keySet().contains("sum")) {
			if (sumAvgCount.get("avg").getSecond() == sumAvgCount.get("sum").getSecond()) {
				//use arid and srid to retrieve average and summation to compute count
				//compare with ridval
				statment = true;
			}
		}
		
		List<List<String>> listRID = new ArrayList<>(mapRID.keySet());
		Iterator<List<String>> iterListRID = listRID.iterator();
		outer:
		while (iterListRID.hasNext()) {
			List<String> tupleRID = iterListRID.next();
			if (mapBID.keySet().contains(tupleRID)) {
				List<T> ridValue = (List<T>) mapRID.get(tupleRID).get(rid);
				//System.out.println(ridValue.get(0).getClass().getSimpleName() + " " + rid);
				int ridval;
				if (ridValue.get(0).getClass().getSimpleName().contains("Long")) {
					Long ridvalu = (Long) ridValue.get(0);
					ridval = ridvalu.intValue(); //_________________________error
				} else {
					ridval = (Integer) ridValue.get(0);
				}
				if (statment == true) {
					int avid = sumAvgCount.get("avg").getFirst();
					int suid = sumAvgCount.get("sum").getFirst();
					List<T> avidValue = (List<T>) mapRID.get(tupleRID).get(avid);
					List<T> suidValue = (List<T>) mapRID.get(tupleRID).get(suid);
					BigDecimal avidval = (BigDecimal) avidValue.get(0);
					BigDecimal suidval = (BigDecimal) suidValue.get(0);
					int ans = (int) (Math.round(suidval.doubleValue() / avidval.doubleValue()));
					if (ans != ridval) {
						valid = false;
						break outer;
					}
				}
				//System.out.println("SwitchDemo.gotoCount_count " + mapBID + " compare " + ridval);
				if (mapBID.get(tupleRID) != null) {
					List<Integer> bidSize = new ArrayList<Integer>(mapBID.get(tupleRID).keySet());
					if (mapBID.get(tupleRID).get(bidSize.get(0)).size() < ridval) {
						valid = false;
						break outer;
					}
					else if (mapBID.get(tupleRID).get(bidSize.get(0)).size() == ridval) {
					//refer frequency
					//System.out.println(tupleRID);
						inferFreq.add(tupleRID);
					}
				} else {
					if (ridval > 1) {
						valid = false;
						break outer;
					} else if (ridval == 1) {
						inferFreq.add(tupleRID);
					}
				}

			}
			
		}
		if (valid == true) {
			//set frequency and update tuple count interference only when count equal aggregate value
			cn.isFrequency = true;
			cn.setInferFrequency(inferFreq);
		}
		return valid;
		
	}
	
	protected static <T> List<List<T>> cartesianProduct(List<List<T>> lists) {
	    List<List<T>> resultLists = new ArrayList<List<T>>();
	    if (lists.size() == 0) {
	        resultLists.add(new ArrayList<T>());
	        return resultLists;
	    } else {
	        List<T> firstList = lists.get(0);
	        List<List<T>> remainingLists = cartesianProduct(lists.subList(1, lists.size()));
	        for (T condition : firstList) {
	            for (List<T> remainingList : remainingLists) {
	                ArrayList<T> resultList = new ArrayList<T>();
	                resultList.add(condition);
	                resultList.addAll(remainingList);
	                resultLists.add(resultList);
	            }
	        }
	    }
	    return resultLists;
	}
}
