package com.sql;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class ReadDatabase {

	public static <T> void readDatabase(Map<List<Integer>, 
			Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<Integer, List<String>> baseRowStore, ArrayList<Integer> combinaBID,
			List<Integer> getBaseCols, Map<Integer, List<T>> baseColumn,
			Map<Integer, Map<String, List<Integer>>> baseFreq) 
	{
		//Map<List<String>, Multimap<Integer, T>> baseRowSet = new HashMap<>();
		
		List<List<String>> baseSetWords = new ArrayList<>();
		// List<List<Integer>> findCommon = new ArrayList<>();
		for (int j : combinaBID) {
			List<String> words = new ArrayList<>(baseFreq.get(j).keySet());
			baseSetWords.add(words);
		}
		
		//cartesian product for keys
		//grab common indexes
		//load into bigmap
		List<List<String>> cartKeys = MapBoth.cartesianProduct(baseSetWords);
		Map<List<String>, Multimap<Integer, T>> multivector = new HashMap<>();
		for (List<String> cart : cartKeys) {
			List<List<Integer>> indexes = new ArrayList<>();
			for (int i = 0; i < cart.size(); i++) {
				int d = combinaBID.get(i);
				List<Integer> grab = baseFreq.get(d).get(cart.get(i));
				indexes.add(grab);
			}
			
			Set<Integer> setCommonIndex = PreprocessBase.commonListIndices(indexes);
			List<Integer> commonIndex = new ArrayList<>(setCommonIndex);
			//List<Integer> commonIndex = intersect(indexes);
			if (!commonIndex.isEmpty()) {
				Multimap<Integer, T> vector = ArrayListMultimap.create();
				for (int b : getBaseCols) {			
					for (int c : commonIndex) {
						vector.put(b, baseColumn.get(b).get(c));
					}		
				}
				multivector.put(cart, vector);
			}
		}
		baseSPGA.put(combinaBID, multivector);
		
		/***
		for (int i : baseRowStore.keySet()) {
			List<String> tuplet = new ArrayList<>();
			List<List<Integer>> findCommonIndex = new ArrayList<>();
			//System.out.println("baseFeq " + baseFreq.keySet() + "combinaBID " + combinaBID);
			for (int j : combinaBID) {
				String str = baseRowStore.get(i).get(j-1);
				
				tuplet.add(str);
				List<Integer> index = baseFreq.get(j).get(str);
				findCommonIndex.add(index);
			}
			
			Multimap<Integer, T> getColValue= ArrayListMultimap.create();
			Set<Integer> setCommonIndex = PreprocessBase.commonListIndices(findCommonIndex);
			List<Integer> commonIndex = new ArrayList<>(setCommonIndex);
			for (int k : getBaseCols) {
				List<T> colvalues = baseColumn.get(k);
				for (int l : commonIndex) {
					getColValue.put(k, colvalues.get(l));
				}
			}
			baseRowSet.put(tuplet, getColValue);	
		}
		baseSPGA.put(combinaBID, baseRowSet);
		*/
	}
	/**
	public static List<Integer> intersect(List<List<Integer>> lists) {
		
		List<Integer> commons = new ArrayList<Integer>();
		//size must be greater than one
		//if (lists.size() > 1) {
			commons.addAll(lists.get(0));
			for (ListIterator<List<Integer>> iter = lists.listIterator(0); iter.hasNext(); ) {
				commons.retainAll(iter.next());
			}
		//} else {
		//	commons.addAll(lists.get(0));
		//}
		return commons;
	}
	*/
	public static <T> void readSkipDatabase(Map<List<Integer>, 
			Map<List<String>, Multimap<Integer, T>>> baseSPGA,
			Map<Integer, List<String>> baseRowStore, ArrayList<Integer> combinaBID,
			List<Integer> getBaseCols, Map<Integer, List<T>> baseColumn,
			Map<Integer, Map<String, List<Integer>>> baseFreq,
			List<List<String>> skipBase ) 
	{
		
		Map<List<String>, Multimap<Integer, T>> multivector = new HashMap<>();
		for (List<String> cart : skipBase) 
		{
			List<List<Integer>> indexes = new ArrayList<>();
			for (int i = 0; i < cart.size(); i++) {
				int d = combinaBID.get(i);
				List<Integer> grab = baseFreq.get(d).get(cart.get(i));
				indexes.add(grab);
			}
			
			Set<Integer> setCommonIndex = PreprocessBase.commonListIndices(indexes);
			List<Integer> commonIndex = new ArrayList<>(setCommonIndex);
			
			if (!commonIndex.isEmpty()) {
				Multimap<Integer, T> vector = ArrayListMultimap.create();
				for (int b : getBaseCols) {			
					for (int c : commonIndex) {
						vector.put(b, baseColumn.get(b).get(c));
					}		
				}
				multivector.put(cart, vector);
			}
		}
		baseSPGA.put(combinaBID, multivector);
	}
}
