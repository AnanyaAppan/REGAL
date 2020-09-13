package com.sql;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class SingleConstraints {
	public static <T> List<Pair<Integer, Multimap<Object, Integer>>> aggregation (Map<Integer, 
			Map<String, List<Integer>>> baseFreq, Map<Integer, Map<String, List<Integer>>> resultFreq, 
			Map<Integer, String> baseClass, Map<Integer, String> columnClass, Map<Integer, MathClass<Integer>> inmath, 
			Map<Integer, MathClass<BigDecimal>> domath, Map<Integer, MathClass<Integer>> inresult, 
			Map<Integer, MathClass<BigDecimal>> doresult, Multimap<Integer, Integer> resTobase, int baseCardinality) {
		
		List<Pair<Integer, Multimap<Object, Integer>>> availableFunction = new ArrayList<>();
		List<Integer> resultkeys = new ArrayList<>(columnClass.keySet());
		List<Integer> basekeys = new ArrayList<>(baseClass.keySet());
		
		for (int i : resultkeys) {
			
			Multimap<Object, Integer> function = ArrayListMultimap.create();
			if (columnClass.get(i).contains("Integer") || columnClass.get(i).contains("BigDecimal") 
					|| columnClass.get(i).contains("Long")) {
				// System.out.println("rule_1_" + i + "_" + resTobase.get(i));
				for (int j : resTobase.get(i)) {
					if (checkBothFrequencies(resultFreq.get(i), baseFreq.get(j)) == true) {
						function.put("max", j);
						function.put("min", j);
						function.put("avg", j);
						//function.put("median", j);
					}
				}
				if (columnClass.get(i).contains("Integer") || columnClass.get(i).contains("Long")) {
					if (preprocessingCount(inresult.get(i), baseCardinality, resultFreq.get(i).keySet()))
						function.put("count", 0);
				}
				//System.out.println("current_candidates1_" + i + "_" + function);
				
				for (int k: basekeys) {
					//System.out.println("base_" + k + "_" + baseClass.get(k) + "_result_" + i + "_" + columnClass.get(i));
					if ( //(columnClass.get(i).contains("Integer") || columnClass.get(i).contains("Long")) 
							columnClass.get(i).contains("Integer")
							&& baseClass.get(k).contains("Integer")) {
						if (!resTobase.get(i).contains(k)) {
							if (preprocessingAverage(inresult.get(i), inmath.get(k), baseFreq.get(k), resultFreq.get(i)))
								function.put("avg", k);
						}
						if (preprocessingSum(inresult.get(i), inmath.get(k), baseFreq.get(k), resultFreq.get(i)) ) {
							function.put("sum", k); 
						}
							
					} else if (columnClass.get(i).contains("BigDecimal") && baseClass.get(k).contains("BigDecimal")) {
						if (!resTobase.get(i).contains(k)) {
							if (preprocessingAverage(doresult.get(i), domath.get(k), baseFreq.get(k), resultFreq.get(i))) {
								function.put("avg", k);
								//function.put("median", k);
							}
						}
						if (preprocessingSum(doresult.get(i), domath.get(k), baseFreq.get(k), resultFreq.get(i)) ) 
							function.put("sum", k); 
							
					}
				}
				//System.out.println("current_candidates2_" + i + "_" + function);

			}
			/** else if (columnClass.get(i).contains("String") ) {
				//only min max
			} 
			*/
			
			Pair<Integer, Multimap<Object, Integer>> eachSingle = new Pair<>(i, function);
			availableFunction.add(eachSingle);
		}
		
		for (int i = 1 ; i < availableFunction.size(); i++) {
			int len = availableFunction.get(i).getSecond().size();
			Pair<Integer, Multimap<Object, Integer>> function = availableFunction.get(i);
			int j = i - 1;
			while (j >= 0 && availableFunction.get(j).getSecond().size() > len) {
				availableFunction.set(j+1, availableFunction.get(j));
				j = j-1;
				availableFunction.set(j+1, function);
				len = availableFunction.get(j+1).getSecond().size();
			}
		}
		return availableFunction;
	}
	
	private static <T extends Number & Comparable<T>> boolean preprocessingSum(MathClass<T> mathResult,
			MathClass<T> mathBase, Map<String, List<Integer>> mapbase,
			Map<String, List<Integer>> mapresult) {
		/**long start_agg = System.currentTimeMillis(); */
		boolean cond = false;
		//System.out.println(mathResult.totpos.doubleValue() + " " + mathBase.totpos.doubleValue()); 
		double totpos = mathResult.totpos.doubleValue(); double sumpos = mathBase.totpos.doubleValue();
		double totneg = mathResult.totneg.doubleValue(); double sumneg = mathBase.totneg.doubleValue();
		if (totpos > 0 && totneg < 0) {
			cond = (totneg >= sumneg && sumpos >= totpos) ? true : false;
		} else if (totneg == 0) {
			cond = (mathResult.min.compareTo(mathBase.min) >= 0 && sumpos >= mathResult.max.doubleValue()) ? true : false;
			//System.out.println("SingleConstraints.preprocessingSum_cond " + cond);
			if (cond) {
				//first cond
				for (String restr : mapresult.keySet()) {
					if (restr.equals(mathBase.min.toString())) {
						if ( mapresult.get(restr).size() > mapbase.get(mathBase.min.toString()).size() ) {
								cond = false;
								break;
						}
					}
				}
			}//System.out.println("SingleConstraints.preprocessingSum_second_cond " + cond);
			//second cond
			/**if(cond) {
				double smallersum = 0.0;
				for (String num : mapbase.keySet()) {
					
					double db = Double.parseDouble(num);
					if (db <= mathResult.min.doubleValue()) {
						smallersum += db * mapbase.get(num).size();
						//System.out.println("SingleConstrains.preporcessingSum " + smallersum + " >= " + mathResult.min);
						if (smallersum >= mathResult.min.doubleValue()) {
							cond = true;
							break;
						}
					}
				}
				//System.out.println("SingleConstrains.preporcessingSum " + smallersum + " < " + mathResult.min);
				if (smallersum < mathResult.min.doubleValue())
					cond = false;
			} */
 		}//System.out.println("SingleConstraints.preprocessingSum_final_cond " + cond);
		/**long end_agg = System.currentTimeMillis();
		long diff = end_agg - start_agg;
		System.out.println("Suspect " + diff); */
		return cond;
	
	}

	private static <T extends Number & Comparable<T>> boolean preprocessingAverage(MathClass<T> mathResult,
			MathClass<T> mathBase, Map<String, List<Integer>> mapbase,
			Map<String, List<Integer>> mapresult) {
		
		if (mathBase.getMax().doubleValue() - mathResult.getMax().doubleValue() >= 0 && 
				mathResult.getMin().doubleValue() - mathBase.getMin().doubleValue() >= 0) {
			if (mathBase.getMax().doubleValue() == mathResult.getMax().doubleValue()) {
				if (mapbase.get(mathBase.getMax().toString()).size() >= mapresult.get(mathResult.getMax().toString()).size())
					return true;
				else return false;
			} else if (mathBase.getMin().doubleValue() == mathResult.getMin().doubleValue()) {
				if (mapbase.get(mathBase.getMin().toString()).size() >= mapresult.get(mathResult.getMin().toString()).size())
					return true;
				else return false;
			}
			else return true;
		}
		else return false;
	}

	private static boolean preprocessingCount(
			MathClass<Integer> mathClass, int baseCardinality, Set<String> set) {
		boolean b = true;
		if (mathClass.totneg < 0 || set.contains("0")) b = false;
		if (b) b = (baseCardinality >= mathClass.getMax().intValue() && baseCardinality >= mathClass.totpos.intValue()) 
				? true : false;
		//System.out.println("SingleConstraints.preprocessingCount " + b);
		return b;
	}

	private static boolean checkBothFrequencies(Map<String, List<Integer>> resmap,
			Map<String, List<Integer>> orimap) {
		int count = 0;
		for (String ele: resmap.keySet()) {
			//System.out.println("checkBothFre " + ele + " " + orimap.get(ele).size() + " " + resmap.get(ele).size());
			if (orimap.get(ele).size() >= resmap.get(ele).size())
				count++;
		}
		if (count == resmap.size())
			return true;
		else return false;
	}
}
