package com.sql;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;

public class Extrawork {
	public static <T extends Comparable<T>> List<DoubleBox<Integer>> crossValidation (
			DoubleBox<Integer> subplane, Map<Integer, T> collect, String baseClass,
			Pair<Object, Integer> fungsi, T ridValue, int jr, 
			ListMultimap<Integer, Integer> rangeId, List<T> wholeIndex) 
	{
		List<DoubleBox<Integer>> crossmulti = new ArrayList<>();
		Pair<Integer, Integer> maxplane = subplane.maxX;	 
		Pair<Integer, Integer> minplane = subplane.minX;
		BigDecimal compareBig = new BigDecimal(ridValue.toString());
		int compareDigit = compareBig.intValue();
		//int compareDigit = Integer.parseInt(ridValue.toString());
		int xa = maxplane.getFirst();
		int xb = maxplane.getSecond();	
		// System.out.println("Maximum bounding box_" + xa + " " + xb);
		// System.out.println("Function_" + fungsi.getFirst() + "::" + fungsi.getSecond());
		if (fungsi.getFirst() == "count") {
			Map<Integer, Integer> linecount = new HashMap<>();
			int changeint = Integer.parseInt(ridValue.toString());
			int total = 0;
			List<Integer> xbx = new ArrayList<>();
			for (int x=xa; x<= xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> f = rangeId.get(x);
				if (!f.contains(null)) {
					xbx.add(x);
					linecount.put(x, f.size());
					total += f.size();
				}
				}
			}
			if (changeint == total) {
				if (Collections.min(xbx) <= minplane.getFirst() && 
						minplane.getSecond() <= Collections.max(xbx)) 
				{
					Pair<Integer, Integer> minbxch = new Pair<>(Collections.min(xbx),Collections.max(xbx));
					DoubleBox<Integer> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane); //if the fuzzy min is smaller than original
				}
			} else if (total > changeint) {
				List<Integer> unitKey = new ArrayList<Integer>(linecount.keySet());
				Collections.sort(unitKey);
				Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
				
				OneDimenDigit.countop(linecount, changeint, jr, unitKey, xa, xb+1, multiFunBox, 
						fungsi.getFirst(), fungsi.getSecond(), total);
				//CountExpandShrink.countfilter(linecount, changeint, jr, unitKey, xa, xb+1, multiFunBox, 
				//		fungsi.getFirst(), fungsi.getSecond());
				
				for (int t : multiFunBox.keySet()) {
					List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
					for (DoubleBox<Integer> nb: newbox) {
						Pair<Integer, Integer> nubox = nb.minX;
						Pair<Integer, Integer> unbox = nb.maxX;
						int minl = 
								(minplane.getFirst() < nubox.getFirst()) 
								? minplane.getFirst() : nubox.getFirst();
						int minr = 
								(minplane.getSecond() > nubox.getSecond()) 
								? minplane.getSecond(): nubox.getSecond();
						Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
						if (unbox.getFirst() <= nubox.getFirst() && 
								unbox.getSecond() >= nubox.getSecond() ) {
							DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		else if (fungsi.getFirst() == "sum" && baseClass.contains("Integer")) 
		{
			Map<Integer, Integer> linedigit = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			int total = 0;
			List<Integer> count = new ArrayList<>();
			for (int x= xa; x <= xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> f = rangeId.get(x);
				if (!f.contains(null)) {
					int temp = 0;
					count.add(f.size());
					for (int sf : f) {
						temp += Integer.parseInt(collect.get(sf).toString());
						xbx.add(x);
					}		
					//xbx.add(x); 
					linedigit.put(x, temp);
					total += temp;
				}
				}
			}
			if (fungsi.getFirst() == "sum" && total == compareDigit) 
			{	
				if (Collections.min(xbx) <= minplane.getFirst() 
						&& minplane.getSecond() <= Collections.max(xbx)) 
				{
					Pair<Integer, Integer> minbxch = new Pair<>(Collections.min(xbx),Collections.max(xbx));
					DoubleBox<Integer> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane); //if the fuzzy min is smaller than original
				}
			} else if (fungsi.getFirst() == "sum" && total > compareDigit) 
			{
				List<Integer> unitKey = new ArrayList<Integer>(linedigit.keySet());
				Collections.sort(unitKey);
				Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
				
				OneDimenDigit.sumpop(linedigit, compareDigit, jr, unitKey, xa, xb+1, 
						multiFunBox, fungsi.getFirst(), fungsi.getSecond(), total);
				//OneDimenDigit.shrinkSumFilter(linedigit, compareDigit, jr, unitKey, xa, xb+1, 
				//		multiFunBox, fungsi.getFirst(), fungsi.getSecond(), total);
				
				for (int t : multiFunBox.keySet()) 
				{
					List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
					for (DoubleBox<Integer> nb: newbox) 
					{
						Pair<Integer, Integer> nubox = nb.minX;
						Pair<Integer, Integer> unbox = nb.maxX;
						int minl = 
								(minplane.getFirst() < nubox.getFirst()) 
								? minplane.getFirst() : nubox.getFirst();
						int minr = 
								(minplane.getSecond() > nubox.getSecond()) 
								? minplane.getSecond(): nubox.getSecond();
						Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
						if (unbox.getFirst() <= nubox.getFirst() && 
								unbox.getSecond() >= nubox.getSecond() ) 
						{
							DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		else if ((fungsi.getFirst() == "sum" || fungsi.getFirst() == "avg") 
				&& baseClass.contains("BigDecimal") ) 
		{
			Map<Integer, BigDecimal> linedecimal = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			BigDecimal total = new BigDecimal(0);
			List<Integer> count = new ArrayList<>();
			//List<Integer> sortRange = new ArrayList<>(rangeId.keySet()); /////@@@@@@@@@@@@@@@@@@@@@@
			//Collections.sort(sortRange);
			for (int x= xa; x <= xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> f = rangeId.get(x);
				if (!f.contains(null)) {
					BigDecimal temp = new BigDecimal(0);
					count.add(f.size());
					for (int sf : f) {
						temp = temp.add(new BigDecimal(collect.get(sf).toString()));
						xbx.add(x);
					}		
					//xbx.add(x); 
					linedecimal.put(x, temp);
					total = total.add(temp);
				}
				}
			}
			BigDecimal compareTotal = compareBig.multiply(new BigDecimal(xbx.size()));
			//System.out.println(linedecimal);
			//System.out.println("test " + total + " ? " + compareTotal + " " + count);
			//System.out.println(xbx);
			BigDecimal tolak = total.subtract(compareTotal);
			if (fungsi.getFirst() == "sum" && total.equals(compareBig)) {
				
				if (Collections.min(xbx) <= minplane.getFirst() && minplane.getSecond() <= Collections.max(xbx)) 
				{
					Pair<Integer, Integer> minbxch = new Pair<>(Collections.min(xbx),Collections.max(xbx));
					DoubleBox<Integer> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane); //if the fuzzy min is smaller than original
				}
			} else if (fungsi.getFirst() == "avg" && tolak.intValue() == 0) 
			{
				if (!xbx.isEmpty()) {
				if (Collections.min(xbx) <= minplane.getFirst() && minplane.getSecond() <= Collections.max(xbx)) 
				{
					Pair<Integer, Integer> minbxch = new Pair<>(Collections.min(xbx),Collections.max(xbx));
					DoubleBox<Integer> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane);
				}
				}
			} else if (fungsi.getFirst() == "sum" && total.compareTo(compareBig) > 0) 
			{
				List<Integer> unitKey = new ArrayList<Integer>(linedecimal.keySet());
				Collections.sort(unitKey);
				//System.out.println("unitKey " + unitKey);
				Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
				//FilterSelection.decimalSumFilter(linedecimal, compareBig, jr, unitKey, 
				//xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond());
				//SumExpandShrink.shrinkSumFilter(linedecimal, compareBig, jr, unitKey, 
				//		xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), total);
				OneDimenDecimal.sumop(linedecimal, compareBig, jr, unitKey, xa, xb+1, multiFunBox, 
						fungsi.getFirst(), fungsi.getSecond(), total);
				
				for (int t : multiFunBox.keySet()) {
					List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
					for (DoubleBox<Integer> nb: newbox) {
						Pair<Integer, Integer> nubox = nb.minX;
						Pair<Integer, Integer> unbox = nb.maxX;
						int minl = 
								(minplane.getFirst() < nubox.getFirst()) 
								? minplane.getFirst() : nubox.getFirst();
						int minr = 
								(minplane.getSecond() > nubox.getSecond()) 
								? minplane.getSecond(): nubox.getSecond();
						Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
						if (unbox.getFirst() <= nubox.getFirst() && 
								unbox.getSecond() >= nubox.getSecond() ) {
							DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
							crossmulti.add(crossbox);
						}
					}
				}
			} else if (fungsi.getFirst() == "avg" || total.compareTo(compareTotal) > 0) 
			{
				List<Integer> unitKey = new ArrayList<Integer>(linedecimal.keySet());
				Collections.sort(unitKey);
				//System.out.println(unitKey);
				Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
				//FilterSelection.shrinkAvgFilter(linedecimal, compareBig, jr, unitKey, xa, xb+1, 
				//multiFunBox, fungsi.getFirst(), fungsi.getSecond(), count, total);
				//int initial = 0;
				//while (initial < linedecimal.size() ) {
				//	SumExpandShrink.trialAverage(linedecimal, compareBig, jr, initial, xa, xb+1, unitKey, 
				//			multiFunBox, fungsi.getFirst(), fungsi.getSecond(), count);
				//	initial++;
				//}
				int times = 0;
				for (int u = 0; u < count.size(); u++) {
					times += count.get(u);
				}
				OneDimenDecimal.avgop(linedecimal, compareBig, jr, unitKey, xa, xb+1, multiFunBox, 
						fungsi.getFirst(), fungsi.getSecond(), total, count, times);
				
				for (int t : multiFunBox.keySet()) 
				{
					List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
					for (DoubleBox<Integer> nb: newbox) {
						Pair<Integer, Integer> nubox = nb.minX;
						Pair<Integer, Integer> unbox = nb.maxX;
						int minl = 
								(minplane.getFirst() < nubox.getFirst()) 
								? minplane.getFirst() : nubox.getFirst();
						int minr = 
								(minplane.getSecond() > nubox.getSecond()) 
								? minplane.getSecond(): nubox.getSecond();
						Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
						if (unbox.getFirst() <= nubox.getFirst() && unbox.getSecond() >= nubox.getSecond() ) 
						{
							DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
							crossmulti.add(crossbox);
						}
					}
				}
			} 
		}
		else if ((fungsi.getFirst() == "median") && baseClass.contains("BigDecimal")) 
		{
			Multimap<Integer, BigDecimal> unitMultiDecimal = ArrayListMultimap.create();
			List<Integer> tnuoc = new ArrayList<>();
			List<Integer> xbx = new ArrayList<>();
			for (int x = xa; x <= xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> tid = rangeId.get(x);
				if (!tid.contains(null)) {
					for (int t : tid) {
						unitMultiDecimal.put(x, new BigDecimal(collect.get(t).toString()) );
					}
					tnuoc.add(tid.size());
				}
				xbx.add(x);
				}
			}
			int times = 0;
			for (int u = 0; u < tnuoc.size(); u++) {
				times += tnuoc.get(u);
			}
			BigDecimal[] allDecimals = new BigDecimal[times];
			List<Integer> sortKey = new ArrayList<Integer>(unitMultiDecimal.keySet());
			Collections.sort(sortKey);
			int y = 0;
			for (int k : sortKey) {
				List<BigDecimal> values = (List<BigDecimal>) unitMultiDecimal.get(k);
				for (int z = 0; z < values.size(); z++) {
					allDecimals[z+y] = values.get(z);
				}
				y += values.size();
			}
			BigDecimal[] copy = Arrays.copyOfRange(allDecimals, 0, allDecimals.length); 
			BigDecimal tempMed = new BigDecimal(0);
			int middle = allDecimals.length/2;
			
			List<BigDecimal> tempList = new ArrayList<>(Arrays.asList(allDecimals));
			Collections.sort(tempList);
			BigDecimal[] copyright = tempList.toArray(new BigDecimal[0]);
			//System.out.println("middle__" + middle + "_" + tnuoc);
			if (allDecimals.length%2 == 1) {
				tempMed = copyright[middle];
			} else {
				tempMed = tempMed.add(copyright[middle-1]);
				tempMed = tempMed.add(copyright[middle]); 
				tempMed = tempMed.divide(new BigDecimal(2.0)); 
			}
			//System.out.println("every_II_" + 0 +"_"+(middle)+"_"+copyright.length + " " + tempMed + " vs " + compareBig);
			//System.out.println(Arrays.toString(copyright));
			if (tempMed.compareTo(compareBig) == 0) {
				if (!xbx.isEmpty()) {
					if (Collections.min(xbx) <= minplane.getFirst() && minplane.getSecond() <= Collections.max(xbx)) 
					{
						Pair<Integer, Integer> minbxch = new Pair<>(Collections.min(xbx),Collections.max(xbx));
						DoubleBox<Integer> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
						crossmulti.add(boundregion);
					} else {
						crossmulti.add(subplane);
					}
				}
			} else {
				List<Integer> unitKey = new ArrayList<Integer>(unitMultiDecimal.keySet());
				Collections.sort(unitKey);
				Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
				AdditionalUDF.searchMedian(unitMultiDecimal, compareBig, unitKey, multiFunBox, copy, tnuoc, 
						xa, xb+1, jr);
				//AdditionalUDF.decrementMedian(unitMultiDecimal, compareBig, unitKey, multiFunBox, tempList, tnuoc, 
				//		xa, xb+1, jr);
				for (int t : multiFunBox.keySet()) 
				{
					List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
					for (DoubleBox<Integer> nb: newbox) {
						Pair<Integer, Integer> nubox = nb.minX;
						Pair<Integer, Integer> unbox = nb.maxX;
						int minl = 
								(minplane.getFirst() < nubox.getFirst()) 
								? minplane.getFirst() : nubox.getFirst();
						int minr = 
								(minplane.getSecond() > nubox.getSecond()) 
								? minplane.getSecond(): nubox.getSecond();
						Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
						if (unbox.getFirst() <= nubox.getFirst() && unbox.getSecond() >= nubox.getSecond() ) 
						{
							DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		else if ((fungsi.getFirst() == "max" || fungsi.getFirst() == "min") && baseClass.contains("Integer")) 
		{
			List<Integer> linevalid = new ArrayList<>();
			List<Integer> pointindex = new ArrayList<>();
			Map<Integer, Integer> lineDecimal = new HashMap<>();
			for (int x=xa; x<=xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> tid = rangeId.get(x);
				if (!tid.contains(null)) {
					List<T> f = new ArrayList<>();
					for (int t: tid) {
						f.add(collect.get(t));
					}
					System.out.println(x + " " + f.size());
					if (fungsi.getFirst() == "max") {
						int sthg = new Integer(Collections.max(f).toString());
						linevalid.add(sthg);
						if (sthg == compareDigit) {
							pointindex.add(x);
						}
						lineDecimal.put(x, sthg);
						
					} else {
						int sthg = new Integer(Collections.min(f).toString());
						linevalid.add(sthg);
						if (sthg == compareDigit) {
							pointindex.add(x);
						}
						lineDecimal.put(x, sthg);
					}
				} else {
					linevalid.add(null);
				}	
			}
			}
			List<Integer> linenotvalid = new ArrayList<>(linevalid);
			linenotvalid.removeAll(Collections.singleton(null));
			//System.out.println("linevalid " + xa + " " + xb + " " + linevalid); 
			//System.out.println("insideFuzzy " + insideFuzzy);
			if (fungsi.getFirst() == "max") {
				if (!linenotvalid.isEmpty()) {
				int sthg = Collections.max(linenotvalid);
				if (sthg == compareDigit) {
					List<Integer> xs = new ArrayList<>();
					xs.add(subplane.minX.getFirst());
					xs.add(subplane.minX.getSecond());
					xs.addAll(pointindex);
					//System.out.println("min_fuzzy " + xs);
					Pair<Integer, Integer> leftRight = new Pair<>(Collections.min(xs), Collections.max(xs));
					DoubleBox<Integer> boundregion = new DoubleBox<>(leftRight, subplane.maxX);
					crossmulti.add(boundregion);
				} else if (linevalid.contains(compareDigit) && sthg > compareDigit ) 
				{
					Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
					List<Integer> unitKey = new ArrayList<>(lineDecimal.keySet());
					Collections.sort(unitKey);
					betterexpandInt(lineDecimal, compareDigit, jr, unitKey, xa, xb+1, 
							multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					//FilterSelection.shrinkMaxMin(lineDecimal, compareBig, jr, unitKey, 
					//xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					for (int t : multiFunBox.keySet()) 
					{
						List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
						for (DoubleBox<Integer> nb: newbox) 
						{
							Pair<Integer, Integer> nubox = nb.minX;
							Pair<Integer, Integer> unbox = nb.maxX;
							int minl = 
									(minplane.getFirst() <= nubox.getFirst()) 
									? minplane.getFirst() : nubox.getFirst();
							int minr = 
									(minplane.getSecond() >= nubox.getSecond()) 
									? minplane.getSecond(): nubox.getSecond();
							Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
							if (unbox.getFirst() <= nubox.getFirst() && 
									unbox.getSecond() >= nubox.getSecond() ) 
							{
								DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
								crossmulti.add(crossbox);
							}
						}
					}
				}
				}
			} else if (fungsi.getFirst() == "min") {
				if (!linenotvalid.isEmpty()) {
				int sthg = Collections.min(linenotvalid);
				if (sthg == compareDigit) {
					List<Integer> xs = new ArrayList<>();
					xs.add(minplane.getFirst());
					xs.add(minplane.getSecond());
					xs.addAll(pointindex);
					Pair<Integer, Integer> leftRight = new Pair<>(Collections.min(xs), Collections.max(xs));
					DoubleBox<Integer> boundregion = new DoubleBox<>(leftRight, subplane.maxX);
					crossmulti.add(boundregion);
				} else if (linevalid.contains(compareBig) && sthg < compareDigit) {
					Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
					//FilterSelection.decimalMinRange(0, linevalid.size(), linevalid, compareBig, jr, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), xa);
					
					List<Integer> unitKey = new ArrayList<>(lineDecimal.keySet());
					Collections.sort(unitKey);
					betterexpandInt(lineDecimal, compareDigit, jr, unitKey, xa, xb+1, 
							multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					//FilterSelection.betterexpand(lineDecimal, compareBig, jr, unitKey, xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					//FilterSelection.shrinkMaxMin(lineDecimal, compareBig, jr, unitKey, xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					
					for (int t : multiFunBox.keySet()) {
						List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
						for (DoubleBox<Integer> nb: newbox) {
							Pair<Integer, Integer> nubox = nb.minX;
							Pair<Integer, Integer> unbox = nb.maxX;
							int minl = (minplane.getFirst() < nubox.getFirst()) ? minplane.getFirst() : nubox.getFirst();
							int minr = (minplane.getSecond() > nubox.getSecond()) ? minplane.getSecond(): nubox.getSecond();
							Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
							if (unbox.getFirst() <= nubox.getFirst() && 
									unbox.getSecond() >= nubox.getSecond() ) {
								DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
								crossmulti.add(crossbox);
							}
						}
					}
				}
			}
		} 
		}
		else if ((fungsi.getFirst() == "max" || fungsi.getFirst() == "min") && baseClass.contains("BigDecimal")) 
		{
			List<BigDecimal> linevalid = new ArrayList<>();
			List<Integer> pointindex = new ArrayList<>();
			Map<Integer, BigDecimal> lineDecimal = new HashMap<>();
			for (int x=xa; x<=xb; x++) {
				if (rangeId.containsKey(x)) {
				List<Integer> tid = rangeId.get(x);
				if (!tid.contains(null)) {
					List<T> f = new ArrayList<>();
					for (int t: tid) {
						f.add(collect.get(t));
					}
					if (fungsi.getFirst() == "max") {
						BigDecimal sthg = new BigDecimal(Collections.max(f).toString());
						linevalid.add(sthg);
						if (sthg.equals(compareBig)) {
							pointindex.add(x);
						}
						lineDecimal.put(x, sthg);
						/*if (x >= minplane.getFirst() && x <= minplane.getSecond()) {
							insideFuzzy.add(sthg);	
						}*/
					} else {
						BigDecimal sthg = new BigDecimal(Collections.min(f).toString());
						linevalid.add(sthg);
						if (sthg.equals(compareBig)) {
							pointindex.add(x);
						}
						lineDecimal.put(x, sthg);
						/*if (x >= minplane.getFirst() && x <= minplane.getSecond()) {
							insideFuzzy.add(sthg);	
						}*/
					}
				} else {
					linevalid.add(null);
				}	
			}
			}
			List<BigDecimal> linenotvalid = new ArrayList<>(linevalid);
			linenotvalid.removeAll(Collections.singleton(null));
			//System.out.println("linevalid " + xa + " " + xb + " " + linevalid); 
			//System.out.println("insideFuzzy " + insideFuzzy);
			if (fungsi.getFirst() == "max") {
				if (!linenotvalid.isEmpty()) {
				BigDecimal sthg = Collections.max(linenotvalid);
				if (sthg.equals(compareBig)) {
					List<Integer> xs = new ArrayList<>();
					xs.add(subplane.minX.getFirst());
					xs.add(subplane.minX.getSecond());
					xs.addAll(pointindex);
					//System.out.println("min_fuzzy " + xs);
					Pair<Integer, Integer> leftRight = new Pair<>(Collections.min(xs), Collections.max(xs));
					DoubleBox<Integer> boundregion = new DoubleBox<>(leftRight, subplane.maxX);
					crossmulti.add(boundregion);
				} else if (linevalid.contains(compareBig) && sthg.compareTo(compareBig) > 0) {
					Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
					
					//FilterSelection.decimalMaxRange(0, linevalid.size(), linevalid, compareBig, 
					//jr, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), xa);
					List<Integer> unitKey = new ArrayList<>(lineDecimal.keySet());
					Collections.sort(unitKey);
					OneDimenDecimal.betterexpand(lineDecimal, compareBig, jr, unitKey, xa, xb+1, 
							multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					//FilterSelection.shrinkMaxMin(lineDecimal, compareBig, jr, unitKey, xa, xb+1, 
					//multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					for (int t : multiFunBox.keySet()) {
						List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
						for (DoubleBox<Integer> nb: newbox) {
							Pair<Integer, Integer> nubox = nb.minX;
							Pair<Integer, Integer> unbox = nb.maxX;
							int minl = 
									(minplane.getFirst() <= nubox.getFirst()) 
									? minplane.getFirst() : nubox.getFirst();
							int minr = 
									(minplane.getSecond() >= nubox.getSecond()) 
									? minplane.getSecond(): nubox.getSecond();
							Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
							if (unbox.getFirst() <= nubox.getFirst() && 
									unbox.getSecond() >= nubox.getSecond() ) {
								DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
								crossmulti.add(crossbox);
							}
						}
					}
				}
				}
			} else if (fungsi.getFirst() == "min") {
				if (!linenotvalid.isEmpty()) {
				BigDecimal sthg = Collections.min(linenotvalid);
				if (sthg.equals(compareBig)) {
					List<Integer> xs = new ArrayList<>();
					xs.add(minplane.getFirst());
					xs.add(minplane.getSecond());
					xs.addAll(pointindex);
					Pair<Integer, Integer> leftRight = new Pair<>(Collections.min(xs), Collections.max(xs));
					DoubleBox<Integer> boundregion = new DoubleBox<>(leftRight, subplane.maxX);
					crossmulti.add(boundregion);
				} else if (linevalid.contains(compareBig) && sthg.compareTo(compareBig) < 0) {
					Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
					//FilterSelection.decimalMinRange(0, linevalid.size(), linevalid, compareBig, jr, 
					//multiFunBox, fungsi.getFirst(), fungsi.getSecond(), xa);
					
					List<Integer> unitKey = new ArrayList<>(lineDecimal.keySet());
					Collections.sort(unitKey);
					OneDimenDecimal.betterexpand(lineDecimal, compareBig, jr, unitKey, 
							xa, xb+1, multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					//FilterSelection.shrinkMaxMin(lineDecimal, compareBig, jr, unitKey, xa, xb+1, 
					//multiFunBox, fungsi.getFirst(), fungsi.getSecond(), pointindex);
					
					for (int t : multiFunBox.keySet()) {
						List<DoubleBox<Integer>> newbox = (List<DoubleBox<Integer>>) multiFunBox.get(t);
						for (DoubleBox<Integer> nb: newbox) {
							Pair<Integer, Integer> nubox = nb.minX;
							Pair<Integer, Integer> unbox = nb.maxX;
							int minl = 
									(minplane.getFirst() < nubox.getFirst()) 
									? minplane.getFirst() : nubox.getFirst();
							int minr = 
									(minplane.getSecond() > nubox.getSecond()) 
									? minplane.getSecond(): nubox.getSecond();
							Pair<Integer, Integer> newnubox = new Pair<>(minl, minr);
							if (unbox.getFirst() <= nubox.getFirst() && 
									unbox.getSecond() >= nubox.getSecond() ) {
								DoubleBox<Integer> crossbox = new DoubleBox<>(newnubox, unbox);
								crossmulti.add(crossbox);
							}
						}
					}
				}
			}
		}
		}
		return crossmulti;
	}
	
	public static void betterexpandInt(Map<Integer, Integer> unitMapDecimal, int result, int rid,
			List<Integer> unitKey, int startLen, int unitLen, Multimap<Integer, DoubleBox<Integer>> multiFunBox, 
			Object function, int j, List<Integer> pinpoint) {
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
					int cellunit = unitMapDecimal.get(gotoleft);
					if (function == "max" && cellunit > result) {
						//System.out.println(cellunit + " " + gotoleft);
						break;
					} else if (function == "min" && cellunit < result) {
						//System.out.println(cellunit + " " + gotoleft);
						break;
					}
				}
				for (int k = xpoint+1; k < unitKey.size(); k++) {
					gotoright = unitKey.get(k);
					int cellunit = unitMapDecimal.get(gotoright);
					if (function == "min" && cellunit < result) {
						//System.out.println(cellunit + " " + gotoright);
						break;
					} else if (function == "max" && cellunit > result) {
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

			}
		}
	}
}
