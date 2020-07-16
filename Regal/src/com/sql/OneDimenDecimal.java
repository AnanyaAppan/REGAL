package com.sql;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Multimap;

public class OneDimenDecimal {
	public static void sumop(Map<Integer, BigDecimal> unitMapDecimal,
			BigDecimal result, int rid, List<Integer> unitKey, int startLen, int unitLen, 
			Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int jj, BigDecimal totalsum) 
	{
		BigDecimal temp = totalsum; 
		BigDecimal slup = new BigDecimal(0);
					
		for (int i=0; i<unitMapDecimal.size(); i++) {
						
			if (i-1 >= 0) {
				slup = slup.add(unitMapDecimal.get(unitKey.get(i-1)));
				temp = totalsum.subtract(slup);
			}
			int j = unitMapDecimal.size()-1;		
			while (j > i && temp.compareTo(result) > 0) 
			{
				temp = temp.subtract(unitMapDecimal.get(unitKey.get(j))); 		
				j--;
			}
			if (temp.compareTo(result) == 0) {
				Pair<Integer, Integer> fuzzyMin = new Pair<>(unitKey.get(i),unitKey.get(j));
				int newinit = 
						(Collections.min(unitKey) != unitKey.get(i)) ? 
								unitKey.get(unitKey.indexOf(unitKey.get(i))-1) : startLen;
				int newj = 
						(Collections.max(unitKey) != unitKey.get(j)) ? 
								unitKey.get(unitKey.indexOf(unitKey.get(j))+1) : unitLen-1;
				Pair<Integer,Integer> fuzzyMax = new Pair<>(newinit, newj);
				DoubleBox<Integer> onebox = new DoubleBox<>(fuzzyMin, fuzzyMax);
				multiFunBox.put(rid, onebox);
			}			
		}
	} 
	public static void avgop(Map<Integer, BigDecimal> unitMapDecimal,
			BigDecimal result, int rid, List<Integer> unitKey, int startLen, int unitLen, 
			Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int jj, 
			BigDecimal totalsum, List<Integer> count, int times) 
	{
		BigDecimal temp = totalsum;
		//System.out.println("times " + times);
		BigDecimal timesresult = new BigDecimal(0); 
		BigDecimal slup = new BigDecimal(0);
		int dda = 0;
						
		for (int i=0; i<unitMapDecimal.size(); i++) {
			int timing = times;			
			if (i-1 >= 0) {
				slup = slup.add(unitMapDecimal.get(unitKey.get(i-1)));
				dda += count.get(i-1);
				temp = totalsum.subtract(slup);
				timing -= dda;
			}
				
			timesresult = result.multiply(new BigDecimal(timing));
			//System.out.println("outimesresult_old " + timesresult);
			timesresult = timesresult.subtract(temp);
			//System.out.println("outimesresult_new " + timesresult);
			int j = unitMapDecimal.size()-1;
			while (j > i && timesresult.intValue() != 0) 
			{
				temp = temp.subtract(unitMapDecimal.get(unitKey.get(j))); 
				timing -= count.get(j);
					
				timesresult = result.multiply(new BigDecimal(timing));
				//System.out.println("intimesresult_old " + timesresult);
				timesresult = timesresult.subtract(temp);
				//System.out.println("intimesresult_new " + timesresult);
				j--;
			}
			if (timesresult.intValue() == 0) {
				Pair<Integer, Integer> fuzzyMin = new Pair<>(unitKey.get(i),unitKey.get(j));
				int newinit = 
					(Collections.min(unitKey) != unitKey.get(i)) ? 
							unitKey.get(unitKey.indexOf(unitKey.get(i))-1) : startLen;
				int newj = 
					(Collections.max(unitKey) != unitKey.get(j)) ? 
							unitKey.get(unitKey.indexOf(unitKey.get(j))+1) : unitLen-1;
				Pair<Integer,Integer> fuzzyMax = new Pair<>(newinit, newj);
				DoubleBox<Integer> onebox = new DoubleBox<>(fuzzyMin, fuzzyMax);
				multiFunBox.put(rid, onebox);
			}			
		}
	}
	//expansion for mono, pick THE point and expand widely
	public static void betterexpand(Map<Integer, BigDecimal> unitMapDecimal, 
			BigDecimal result, int rid, List<Integer> unitKey, int startLen, int unitLen, 
			Multimap<Integer, DoubleBox<Integer>> multiFunBox, 
			Object function, int j, List<Integer> pinpoint) 
	{
		//System.out.println(unitKey.size() + " " + unitKey);
		int xmin = unitKey.get(0); int xmax = unitKey.get(unitKey.size()-1);
		//System.out.println(xmin + " " + pinpoint + " " + xmax);
		for (int xloc : pinpoint) {
			if (xloc >= startLen && xloc < unitLen ) {
				int xpoint = unitKey.indexOf(xloc);
				int gotoleft = unitKey.get(0); 
				int gotoright = unitKey.get(unitKey.size()-1);
				for (int i = xpoint-1; i >= 0; i--) {
					gotoleft = unitKey.get(i);
					BigDecimal cellunit = unitMapDecimal.get(gotoleft);
					if (function == "max" && cellunit.compareTo(result) > 0) {
						//System.out.println(cellunit + " " + gotoleft);
						break;
					} else if (function == "min" && cellunit.compareTo(result) < 0) {
						//System.out.println(cellunit + " " + gotoleft);
						break;
					}
				}
				for (int k = xpoint+1; k < unitKey.size(); k++) {
					gotoright = unitKey.get(k);
					BigDecimal cellunit = unitMapDecimal.get(gotoright);
					if (function == "min" && cellunit.compareTo(result) < 0) {
						//System.out.println(cellunit + " " + gotoright);
						break;
					} else if (function == "max" && cellunit.compareTo(result) > 0) {
						//System.out.println(cellunit + " " + gotoright);
						break;
					}
				}
				int kiri = (gotoleft == xmin) ? startLen : gotoleft; 
				//unitKey.get(unitKey.indexOf(gotoleft)-1);
				int kanan = (gotoright == xmax) ? unitLen-1 : gotoright; 
				//unitKey.get(unitKey.indexOf(gotoright)+1);
				Pair<Integer, Integer> maxfuzz = new Pair<>(kiri, kanan);
				Pair<Integer, Integer> minfuzz = new Pair<>(xloc, xloc);
				DoubleBox<Integer> fuzzy = new DoubleBox<Integer>(minfuzz, maxfuzz);
				multiFunBox.put(rid, fuzzy);

					/**
					int left = xloc; int right = xloc;
					expandleft(left, xmin, unitKey, unitMapDecimal, result, function);
					expandright(right, xmax, unitKey, unitMapDecimal, result, function);
					int kiri = (left == xmin) ? startLen : unitKey.get(unitKey.indexOf(left)-1);
					int kanan = (right == xmax) ? unitLen-1 : unitKey.get(unitKey.indexOf(right)+1);
					System.out.println(kiri + " " + left + " " + right + " " + kanan);
					Pair<Integer, Integer> maxfuzz = new Pair<>(kiri, kanan);
					Pair<Integer, Integer> minfuzz = new Pair<>(xloc, xloc);
					DoubleBox<Integer> fuzzy = new DoubleBox<Integer>(minfuzz, maxfuzz);
					multiFunBox.put(rid, fuzzy);
					 */
			}
		}
	}
}
