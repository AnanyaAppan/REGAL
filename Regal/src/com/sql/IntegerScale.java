package com.sql;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;

public class IntegerScale {
	
	public static <T extends Comparable<T>> Map<Integer, Integer> temProduceInteger (Object function, 
			ListMultimap<Integer, Integer> rangeId,
			Map<Integer, T> comidElem, List<Integer> count, 
			int ridvalue, List<Integer> pinpoint) 
	{
		Map<Integer, Integer> unitMapInteger = new HashMap<>();
		for (int x=0; x<rangeId.keySet().size(); x++) {
			List<Integer> rangeIdKey = new ArrayList<>(rangeId.keySet());
			List<Integer> tid = rangeId.get(rangeIdKey.get(x));
			if (!tid.contains(null)) {
				if (function == "sum" || function == "avg") {
					int temp = 0;
					for (int t : tid) {
						temp += Integer.parseInt(comidElem.get(t).toString());
					}
					count.add(tid.size());
					unitMapInteger.put(x, temp);
					/**if (function == "sum")
						unitMapInteger.put(x, temp);
					else if (function == "avg") {
						int temavg = (int) Math.round(temp/tid.size());
						unitMapInteger.put(x, temavg);
					}*/
				} else if (function == "max" || function == "min" ) {
					List<T> units = new ArrayList<T>();
					for (int t : tid) {
						units.add(comidElem.get(t));
					}
					count.add(tid.size());
					if (function == "max") {
						int turn = Integer.parseInt(Collections.max(units).toString());
						unitMapInteger.put(x, turn);
						if (turn == ridvalue) {
							pinpoint.add(x);
						}
					}
					else {
						int turn = Integer.parseInt(Collections.min(units).toString());
						unitMapInteger.put(x, turn);
						if (turn == ridvalue) {
							pinpoint.add(x);
						}
					}
				} else if (function == "count") {
					unitMapInteger.put(x, tid.size());
				}
			}
		}
		return unitMapInteger;
	}
	
	public static <T extends Comparable<T>> Map<Integer, Integer> temProduceCount (Object function, 
			ListMultimap<Integer, Integer> rangeId,
			Map<Integer, T> comidElem) {
		Map<Integer, Integer> unitMapInteger = new HashMap<>();
		for (int x=0; x<rangeId.keySet().size(); x++) {
			List<Integer> rangeIdKey = new ArrayList<>(rangeId.keySet());
			List<Integer> tid = rangeId.get(rangeIdKey.get(x));
			if (!tid.contains(null)) {
				if (function == "count") {
					unitMapInteger.put(x, tid.size());
				}
			}
		}
		return unitMapInteger;	
	}
	
	public static <T extends Comparable<T>> Map<Integer, BigDecimal> temProduceDecimal (Object function, 
			ListMultimap<Integer, Integer> rangeId, 
			Map<Integer, T> comidElem, List<Integer> count, 
			BigDecimal ridvalue, List<Integer> pinpoint) 
	{
		Map<Integer, BigDecimal> unitMapDecimal = new HashMap<>();
		//System.out.println(rangeId.size());
		//System.out.println(rangeId.keySet());
		//List<Integer> sortRange = new ArrayList<>(rangeId.keySet());
		//Collections.sort(sortRange);
		//System.out.println(sortRange);
		for (int x=0; x<rangeId.keySet().size(); x++) {
			List<Integer> rangeIdKey = new ArrayList<>(rangeId.keySet());
			List<Integer> tid = rangeId.get(rangeIdKey.get(x));
			//System.out.println("key " + x + " " + tid.size());
			if (!tid.contains(null)) {
				if (function == "sum" || function == "avg" ) {
					BigDecimal temp = new BigDecimal(0);
					for (int t : tid) {
						//System.out.println(t + " " + comidElem.get(t) + " " + comidElem.size());
						temp = temp.add(new BigDecimal(comidElem.get(t).toString()));
					}
					count.add(tid.size());
					unitMapDecimal.put(x, temp);
					/**
					if (function == "sum")
						unitMapDecimal.put(x, temp);
					else 
						unitMapDecimal.put(x, temp);
						//temp = temp.divide(new BigDecimal(tid.size()),2,RoundingMode.HALF_UP);
					*/		
				} else if (function == "max" || function == "min") {
					List<T> units = new ArrayList<T>();
					for (int t : tid) {
						units.add(comidElem.get(t));
					}
					if (function == "max") {
						BigDecimal turn = new BigDecimal(Collections.max(units).toString());
						unitMapDecimal.put(x, turn);
						if (turn.equals(ridvalue)) {
							pinpoint.add(x);
						}
					}
					else {
						BigDecimal turn = new BigDecimal(Collections.min(units).toString());
						unitMapDecimal.put(x, turn);
						if (turn.equals(ridvalue)) {
							pinpoint.add(x);
						}
					}
				}
			}
		}
		return unitMapDecimal;
	}

	public static <T> Multimap<Integer, BigDecimal> temProduceMulti(ListMultimap<Integer, Integer> rangeId,
			Map<Integer, T> comidElem, BigDecimal result, List<Integer> tnouc) {
		// TODO Auto-generated method stub
		//System.out.println(comidElem);
		Multimap<Integer, BigDecimal> unitMultiDecimal = ArrayListMultimap.create();
		List<Integer> rangeIdKey = new ArrayList<>(rangeId.keySet());
		for (int x=0; x<rangeId.keySet().size(); x++) {
			List<Integer> tid = rangeId.get(rangeIdKey.get(x)); //got problem
			if (!tid.contains(null)) {
				for (int t : tid) {
					unitMultiDecimal.put(rangeIdKey.get(x), new BigDecimal(comidElem.get(t).toString()) );
				}
				tnouc.add(tid.size());
			}
		}
		//System.out.println(unitMultiDecimal);
		return unitMultiDecimal;
	}
}

/**
public static <T extends Comparable<T>> List<Integer> produceUnit (Object function, 
	ListMultimap<Integer, Integer> rangeId, 
	Map<Integer, T> comidElem) 
{
	List<Integer> unitInteger = new ArrayList<>();
	for (int x=0; x<rangeId.keySet().size(); x++) {
		List<Integer> tid = rangeId.get(x);
		if (!tid.contains(null)) {
			if (function == "sum" || function == "avg" ) {
				int temp = 0;
				for (int t : tid) {
					temp += Integer.parseInt(comidElem.get(t).toString());
				}
				if (function == "sum")
					unitInteger.add(temp);
				else unitInteger.add((int) Math.round(temp/tid.size()));
			} else if (function == "max" || function == "min") {
				List<T> units = new ArrayList<T>();
				for (int t : tid) {
					units.add(comidElem.get(t));
				}
				if (function == "max")
					unitInteger.add(Integer.parseInt(Collections.max(units).toString()));
				else unitInteger.add(Integer.parseInt(Collections.min(units).toString()));
			}
		} else {
			if (function == "sum" || function == "avg") {
				unitInteger.add((int) 0);
			} else if (function == "min") {
				unitInteger.add(Integer.MAX_VALUE);
			} else if (function == "max") {
				unitInteger.add(Integer.MIN_VALUE);
			}
		}	
	}
	return unitInteger;
}*/
/**
public static <T extends Comparable<T>> List<BigDecimal> produceDecimal (Object function, 
		Map<Integer, List<Integer>> rangeId, Map<Integer, T> comidElem) 
{
	List<BigDecimal> unitDecimal = new ArrayList<>();
	for (int x=0; x<rangeId.size(); x++) {
		List<Integer> tid = rangeId.get(x);
		if (!tid.contains(null)) {
			if (function == "sum" || function == "avg" ) {
				BigDecimal temp = new BigDecimal(0);
				for (int t : tid) {
					temp = temp.add(new BigDecimal(comidElem.get(t).toString()));
				}
				if (function == "sum")
					unitDecimal.add(temp);
				else { //Non-terminating decimal expansion; no exact representable decimal result.
					temp = temp.divide(new BigDecimal(tid.size()), 2, RoundingMode.HALF_UP);
					unitDecimal.add(temp);
				}
			} else if (function == "max" || function == "min") {
				List<T> units = new ArrayList<T>();
				for (int t : tid) {
					units.add(comidElem.get(t));
				}
				if (function == "max")
					unitDecimal.add(new BigDecimal(Collections.max(units).toString()));
				else unitDecimal.add(new BigDecimal(Collections.min(units).toString()));
			}
		} else {
			if (function == "sum" || function == "avg") {
				unitDecimal.add(new BigDecimal(0));
			} else if (function == "max" || function == "min") {
				unitDecimal.add(null);
			}
		}
	}
	//System.out.println("unit "+ unitDecimal.size()+ " " + unitDecimal);
	return unitDecimal;
}
*/