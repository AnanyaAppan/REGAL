package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Map.Entry;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class MapBoth {
	public static Multimap<Integer, Integer> pairwise (Map<Integer, Map<String, List<Integer>>> oriFreMap,
			Map<Integer, Map<String, List<Integer>>> resFreMap, Map<Integer, String> baseClass, Map<Integer, String> columnClass) {
		Multimap<Integer, Integer> multiMap = ArrayListMultimap.create();
		for (Map<String, List<Integer>> hashRes : resFreMap.values()){
			for (Map<String, List<Integer>> hashOri : oriFreMap.values()){
				if (hashOri.keySet().containsAll(hashRes.keySet())){
					//if (checkBothFrequencies(hashOri, hashRes) == true){
						int resPair = getKeyByValue(resFreMap, hashRes);
						int oriPair = getKeyByValue(oriFreMap, hashOri);
						if (columnClass.get(resPair).equals(baseClass.get(oriPair)))
							multiMap.put(resPair, oriPair);
					//}
				}
			}
		}
		return multiMap;	
	}
	private static Integer getKeyByValue(
			Map<Integer, Map<String, List<Integer>>> map,
			Map<String, List<Integer>> value) {
		for (Entry<Integer, Map<String, List<Integer>>> entryMap : map.entrySet()){
			if (Objects.equals(value, entryMap.getValue())){
				return entryMap.getKey();
			}
		}
		return null;
	}

	public static List<List<Integer>> productRoot(Multimap<Integer, Integer> resTobase) {
		List<List<Integer>> resByres = new ArrayList<>();
		List<Integer> keys = new ArrayList<Integer>(resTobase.keySet());
		Collections.sort(keys);
		for (int k : keys) {
			resByres.add((List<Integer>) resTobase.get(k));
		}
		List<List<Integer>> rootProduct = cartesianProduct(resByres);
		return rootProduct;
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
	
	/*
	private static boolean checkBothFrequencies(Map<String, List<Integer>> hashOri,
			Map<String, List<Integer>> hashRes) {
		int count = 0;
		List<String> obKey = new ArrayList<>(hashRes.keySet());
		for (String oK: obKey)
		{
			if (hashOri.get(oK).size() >= hashRes.get(oK).size())
				count++;
		}
		if (count == hashRes.size())
			return true;
		else return false;
	}
	*/
}
