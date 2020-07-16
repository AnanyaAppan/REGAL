package com.sql;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Multimap;

public class AdditionalUDF {

	public static void decrementMedian(Multimap<Integer, BigDecimal> unitMultiDecimal, BigDecimal result,
			List<Integer> sortKey, Multimap<Integer, DoubleBox<Integer>> multiFunBox, List<BigDecimal> tempList,
			List<Integer> tnuoc, int startLen, int unitLen, int rid) {
		// TODO Auto-generated method stub
		int up = 0;
		for (int i=0; i<tnuoc.size(); i++) {
			BigDecimal tempMed = new BigDecimal(0);
			if (i-1>=0) {
				up += tnuoc.get(i-1);
				List<BigDecimal> decrement = (List<BigDecimal>) unitMultiDecimal.get(i);
				tempList.removeAll(decrement); //
				/*for (BigDecimal de : decrement) {
					tempList.remove(de);
				}*/
			}
			int down = tempList.size();
			int middle = tempList.size()/2;
			if (tempList.size()%2 == 1) {
				tempMed = tempList.get(middle);
			} else {
				tempMed = tempMed.add(tempList.get(middle-1));
				tempMed = tempMed.add(tempList.get(middle));
				tempMed = tempMed.divide(new BigDecimal(2.0)); 
			}
			int j = tnuoc.size()-1;
			while (j > i && tempMed.compareTo(result) != 0) {
				tempMed = new BigDecimal(0);
				down -= tnuoc.get(j);
				List<BigDecimal> decrement = (List<BigDecimal>) unitMultiDecimal.get(j);
				for (BigDecimal de : decrement) {
					tempList.remove(de);
				}
				int mid = tempList.size()/2;
				if (tempList.size()%2 == 1) {
					tempMed = tempList.get(mid);
				} else {
					tempMed = tempMed.add(tempList.get(mid-1));
					tempMed = tempMed.add(tempList.get(mid));
					tempMed = tempMed.divide(new BigDecimal(2.0)); 
				}
				j--;
			}
			if (tempMed.compareTo(result) == 0 ) {
				//System.out.println(i + "---" + j + "___" + sortKey.get(i) + "---" + sortKey.get(j));
				Pair<Integer, Integer> fuzzyMin = new Pair<>(sortKey.get(i),sortKey.get(j));
				
				int newinit = 
					(Collections.min(sortKey) != sortKey.get(i)) ? 
							sortKey.get(sortKey.indexOf(sortKey.get(i))-1) : startLen;
							
				int newj = 
					(Collections.max(sortKey) != sortKey.get(j)) ? 
							sortKey.get(sortKey.indexOf(sortKey.get(j))+1) : unitLen-1;
							
				//System.out.println(newinit + "---" + newj + " " + sortKey);
				Pair<Integer,Integer> fuzzyMax = new Pair<>(newinit, newj);
				DoubleBox<Integer> onebox = new DoubleBox<>(fuzzyMin, fuzzyMax);
				multiFunBox.put(rid, onebox);
			}
		}	
	}
	
	public static void searchMedian(Multimap<Integer, BigDecimal> unitMultiDecimal, BigDecimal result,
			List<Integer> sortKey, Multimap<Integer, DoubleBox<Integer>> multiFunBox, 
			BigDecimal[] allDecimals, List<Integer> tnuoc, int startLen, int unitLen, int rid) {
		// TODO Auto-generated method stub
		//BigDecimal testMed = new BigDecimal(0); 
		int up = 0;
		//System.out.println("count_size_" + tnuoc);
		for (int i=0; i<tnuoc.size(); i++) {
			
			BigDecimal tempMed = new BigDecimal(0);
			
			if (i-1 >= 0) {
				up += tnuoc.get(i-1);
			}
			int down = allDecimals.length;
			BigDecimal[] copy = Arrays.copyOfRange(allDecimals, up, allDecimals.length); 
			int middle = copy.length/2;
			//System.out.println("middle_" + middle);
			List<BigDecimal> tempList = Arrays.asList(copy);
			Collections.sort(tempList);
			BigDecimal[] copyright = tempList.toArray(new BigDecimal[0]);
			if (copy.length%2 == 1) {
				tempMed = copyright[middle];
			} else {
				tempMed = tempMed.add(copyright[middle-1]);
				tempMed = tempMed.add(copyright[middle]); 
				tempMed = tempMed.divide(new BigDecimal(2.0)); 
			}
			//System.out.println("every_I_" +up+"_"+(up+middle)+"_"+down + " " + tempMed + " vs " + result);
			//System.out.println(Arrays.toString(copyright));
			/**
			if (tempMed.compareTo(result) == 0) {
				System.out.println("ok");
				break;
			} */
			//testMed = tempMed.subtract(result);
			
			int j = tnuoc.size()-1;
			
			//System.out.println("original size_" + down);
			while (j > i && tempMed.compareTo(result) != 0) 
			{
			
				tempMed = new BigDecimal(0);
				//System.out.println(tnuoc + " " + j);
				down -= tnuoc.get(j);
				//System.out.println(up + " " + down);
				BigDecimal[] copies = Arrays.copyOfRange(allDecimals, up, down); 
				int mid = copies.length/2;
				//System.out.println("mid_" + mid);
				List<BigDecimal> tempLists = Arrays.asList(copies);
				Collections.sort(tempLists);
				BigDecimal[] copyrights = tempLists.toArray(new BigDecimal[0]);
				if (copies.length%2 == 1) {
					tempMed = copyrights[mid];
				} else {
					tempMed = tempMed.add(copyrights[mid-1]);
					tempMed = tempMed.add(copyrights[mid]); 
					tempMed = tempMed.divide(new BigDecimal(2.0)); 
				}
				//System.out.println("every_J_" +up+"_"+(up+mid)+"_"+down + " " + tempMed + " vs " + result);
				//testMed = tempmed.subtract(result);
				
				j--;
			}
			if (tempMed.compareTo(result) == 0 ) {
				//System.out.println(i + "---" + j + "___" + sortKey.get(i) + "---" + sortKey.get(j));
				Pair<Integer, Integer> fuzzyMin = new Pair<>(sortKey.get(i),sortKey.get(j));
				
				int newinit = 
					(Collections.min(sortKey) != sortKey.get(i)) ? 
							sortKey.get(sortKey.indexOf(sortKey.get(i))-1) : startLen;
							
				int newj = 
					(Collections.max(sortKey) != sortKey.get(j)) ? 
							sortKey.get(sortKey.indexOf(sortKey.get(j))+1) : unitLen-1;
							
				//System.out.println(newinit + "---" + newj + " " + sortKey);
				Pair<Integer,Integer> fuzzyMax = new Pair<>(newinit, newj);
				DoubleBox<Integer> onebox = new DoubleBox<>(fuzzyMin, fuzzyMax);
				multiFunBox.put(rid, onebox);
			}
		}
	}
}
