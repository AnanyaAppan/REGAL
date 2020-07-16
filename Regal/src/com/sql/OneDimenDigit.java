package com.sql;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Multimap;

public class OneDimenDigit {
	public static void countop(Map<Integer, Integer> unitMapInteger,
		int result, int rid, List<Integer> unitKey, int startLen, int unitLen, 
		Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int jj, int totalcount) 
	{
		int temp = totalcount; 
		int slup = 0;
			
		for (int i=0; i<unitMapInteger.size(); i++) {
				
			if (i-1 >= 0) {
				slup += unitMapInteger.get(unitKey.get(i-1));
				temp = totalcount - slup;
			}
			int j = unitMapInteger.size()-1;		
			while (j > i && temp > result) 
			{
				temp -= unitMapInteger.get(unitKey.get(j)); 		
				j--;
			}
			if (temp == result) {
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
	
	//first choice: sum shrink optimization 	
	public static void sumpop(Map<Integer, Integer> unitMapInteger,
		int result, int rid, List<Integer> unitKey, int startLen, int unitLen, 
		Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int jj, int totalsum) 
	{
		int temp = totalsum; 
		int slup = 0;
					
		for (int i=0; i<unitMapInteger.size(); i++) {
						
			if (i-1 >= 0) {
				slup += unitMapInteger.get(unitKey.get(i-1));
				temp = totalsum - slup;
			}
			int j = unitMapInteger.size()-1;		
			while (j > i && temp > result) 
			{
				temp -= unitMapInteger.get(unitKey.get(j)); 		
				j--;
			}
			if (temp == result) {
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

	public static void avgop(Map<Integer, Integer> unitMapInteger, int result, int rid, List<Integer> unitKey,
			int startLen, int unitLen, Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int jj,
			int totalsum, List<Integer> count, int times) {
		// TODO Auto-generated method stub
		int temp = totalsum;
		//System.out.println("times " + times);
		int timesresult = 0; 
		int slup = 0;
		int dda = 0;
						
		for (int i=0; i<unitMapInteger.size(); i++) {
			int timing = times;			
			if (i-1 >= 0) {
				slup += unitMapInteger.get(unitKey.get(i-1));
				dda += count.get(i-1);
				temp = totalsum - slup;
				timing -= dda;
			}
				
			timesresult = result*timing;
			//System.out.println("outimesresult_old " + timesresult);
			timesresult = timesresult - temp;
			//System.out.println("outimesresult_new " + timesresult);
			int j = unitMapInteger.size()-1;
			while (j > i && timesresult != 0) 
			{
				temp = temp - unitMapInteger.get(unitKey.get(j)); 
				timing -= count.get(j);
					
				timesresult = result*timing;
				//System.out.println("intimesresult_old " + timesresult);
				timesresult = timesresult - temp;
				//System.out.println("intimesresult_new " + timesresult);
				j--;
			}
			if (timesresult == 0) {
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

	public static void betterexpand(Map<Integer, Integer> unitMapInteger, int result, int rid, List<Integer> unitKey,
			int startLen, int unitLen, Multimap<Integer, DoubleBox<Integer>> multiFunBox, Object function, int j,
			List<Integer> pinpoint) 
	{
		int xmin = unitKey.get(0); 
		int xmax = unitKey.get(unitKey.size()-1);
		for (int xloc : pinpoint) {
			if (xloc >= startLen && xloc < unitLen ) {
				int xpoint = unitKey.indexOf(xloc);
				int gotoleft = unitKey.get(0); 
				int gotoright = unitKey.get(unitKey.size()-1);
				for (int i = xpoint-1; i >= 0; i--) {
					gotoleft = unitKey.get(i);
					int cellunit = unitMapInteger.get(gotoleft);
					if (function == "max" && cellunit > result) {
						break;
					} else if (function == "min" && cellunit < result) {
						break;
					}
				}
				for (int k = xpoint+1; k < unitKey.size(); k++) {
					gotoright = unitKey.get(k);
					int cellunit = unitMapInteger.get(gotoright);
					if (function == "min" && cellunit < result) {
						break;
					} else if (function == "max" && cellunit > result) {
						break;
					}
				}
				int kiri = (gotoleft == xmin) ? startLen : gotoleft; 
				int kanan = (gotoright == xmax) ? unitLen-1 : gotoright; 
						
				Pair<Integer, Integer> maxfuzz = new Pair<>(kiri, kanan);
				Pair<Integer, Integer> minfuzz = new Pair<>(xloc, xloc);
				DoubleBox<Integer> fuzzy = new DoubleBox<Integer>(minfuzz, maxfuzz);
				multiFunBox.put(rid, fuzzy);

			}
		}
	} 	
}
