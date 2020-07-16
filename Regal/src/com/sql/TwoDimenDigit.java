package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;

public class TwoDimenDigit {
	
	public static <T extends Comparable<T>> Multimap<Integer, DoubleBox<TwoD>> sumshrinking(
			Map<TwoD, Integer> planeDigit, Integer compareResult, Object function, Integer rid, Integer bid, 
			int xa, int xb, int ya, int yb, Set<Integer> xaxiset, Set<Integer> yaxiset) {
		Multimap<Integer, DoubleBox<TwoD>> multifunbox = ArrayListMultimap.create();
		
		List<Integer> xaxis = new ArrayList<Integer>(xaxiset);
		List<Integer> yaxis = new ArrayList<Integer>(yaxiset);
		Collections.sort(xaxis);
		Collections.sort(yaxis);
		
		int xmin = xaxis.get(0); int xmax = xaxis.get(xaxis.size()-1);
		int ymin = yaxis.get(0); int ymax = yaxis.get(yaxis.size()-1);
		
		TwoD first = new TwoD(xmin, ymin);
		TwoD second = new TwoD(xmax, ymax);
		
		List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound = new ArrayList<>();
		if (ymax > ymin) {
			int ry = yaxis.get(yaxis.indexOf(ymax)-1);
			TwoD cutdown = new TwoD(xmax, ry);
			Pair<TwoD, TwoD> shrinkdown = new Pair<>(first, cutdown);
			int tempsun = sumshrinkable(shrinkdown, function, planeDigit);
			int substract = tempsun - compareResult;
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(1);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkdown, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				sumrecursivelytype4(shrinkdown, compareResult, function, planeDigit, ymin, yaxis, recordmove, superbound);
			} 
		}
		
		if (ymin < ymax) {
			int ly = yaxis.get(yaxis.indexOf(ymin)+1);
			TwoD cutup = new TwoD(xmin, ly);
			Pair<TwoD, TwoD> shrinkup = new Pair<>(cutup, second);
			int tempsun = sumshrinkable(shrinkup, function, planeDigit);
			int substract = tempsun - compareResult;
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(2);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkup, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				sumrecursivelytype4(shrinkup, compareResult, function, planeDigit, ymax, yaxis, recordmove, superbound);
				sumrecursivelytype3(shrinkup, compareResult, function, planeDigit, ymin, ymax, yaxis, superbound, recordmove2);
			} 
		}
		
		if (xmax > xmin) {
			int rx = xaxis.get(xaxis.indexOf(xmax)-1);
			TwoD cutright = new TwoD(rx, ymax);
			Pair<TwoD, TwoD> shrinkright = new Pair<>(first, cutright);
			int tempsun = sumshrinkable(shrinkright, function, planeDigit);
			int substract = tempsun - compareResult;
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(3);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkright, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				sumrecursivelytype4(shrinkright, compareResult, function, planeDigit, ymax, yaxis, 
						recordmove, superbound);
				sumrecursivelytype3(shrinkright, compareResult, function, planeDigit, ymin, ymax, yaxis, 
						superbound, recordmove2);
				sumrecursivelytype2(shrinkright, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				
			} 
		}
		
		if (xmin < xmax) {
			int lx = xaxis.get(xaxis.indexOf(xmin)+1);
			TwoD cutleft = new TwoD(lx, ymin);
			Pair<TwoD, TwoD> shrinkleft = new Pair<>(cutleft, second);
			int tempsun = sumshrinkable(shrinkleft, function, planeDigit);
			int substract = tempsun - compareResult;
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(4);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			List<Integer> recordmove4 = new ArrayList<>(recordmove);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkleft, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				sumrecursivelytype4(shrinkleft, compareResult, function, planeDigit, ymax, yaxis, 
						recordmove, superbound);
				sumrecursivelytype3(shrinkleft, compareResult, function, planeDigit, ymin, ymax, yaxis, 
						superbound, recordmove2);
				sumrecursivelytype2(shrinkleft, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				sumrecursivelytype1(shrinkleft, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xmax, xaxis, 
						superbound, recordmove4);

			} 
		}
		
		List<Integer> depth = new ArrayList<>();
		for (Pair<Pair<TwoD, TwoD>, List<Integer>> not : superbound) {
			
			depth.add(not.getSecond().size());	
		}
		List<Pair<Pair<TwoD, TwoD>, List<Integer>>> totalbound = new ArrayList<>();
		if (!depth.isEmpty()) {
		int mindepth = Collections.min(depth);
		for (int d=0; d<depth.size(); d++) {
			if (depth.get(d) == mindepth) {
				totalbound.add(superbound.get(d));
			}
		}
		if (!totalbound.isEmpty()) {
			for (int d = 0; d < depth.size(); d++) {
				if (depth.get(d) > mindepth) {
					boolean store = true;
					for (Pair<Pair<TwoD,TwoD>, List<Integer>> bound : totalbound) {
						if (totalbound.get(d).getSecond().containsAll(bound.getSecond())) 
							store = false;
						
					}
					if (store) totalbound.add(superbound.get(d));
				}
			}
		}
		}
		
		for (Pair<Pair<TwoD, TwoD>, List<Integer>> bound : totalbound) {
			Pair<TwoD, TwoD> maxbx = bound.getFirst();
			if (maxbx.getFirst().x == xmin) {
				maxbx.getFirst().x = xa;
			}
			if (maxbx.getSecond().x == xmax) {
				maxbx.getSecond().x = xb;
			}
			if (maxbx.getFirst().y == ymin) {
				maxbx.getFirst().y = ya;
			}
			if (maxbx.getSecond().y == ymax) {
				maxbx.getSecond().y = yb;
			}
			List<Integer> bx = new ArrayList<>();
			List<Integer> by = new ArrayList<>();
			for (TwoD cell : planeDigit.keySet()) {
				if (maxbx.getFirst().x <= cell.x && cell.x <= maxbx.getSecond().x && 
						maxbx.getFirst().y <= cell.y && cell.y <= maxbx.getSecond().y) {
					bx.add(cell.x);
					by.add(cell.y);
				}
			}
			TwoD minp = new TwoD(Collections.min(bx), Collections.min(by));
			TwoD maxp = new TwoD(Collections.max(bx), Collections.max(by));
			Pair<TwoD, TwoD> minbx = new Pair<>(minp, maxp);
			DoubleBox<TwoD> minmax = new DoubleBox<>(minbx, maxbx);
			/**System.out.println("RID " + rid + "[" + minmax.minX.getFirst().x + " " 
			 		+ minmax.minX.getFirst().y + "] [" + 
					minmax.minX.getSecond().x + " " + minmax.minX.getSecond().y + "] [" + 
					minmax.maxX.getFirst().x + " " + minmax.maxX.getFirst().y + "] [" +
					minmax.maxX.getSecond().x + " " + minmax.maxX.getSecond().y + "]"); */
			
			multifunbox.put(rid, minmax);
		}
			
		return multifunbox;
	}
	
	private static <T extends Comparable<T>> int sumshrinkable(Pair<TwoD, TwoD> type, Object function, 
			Map<TwoD, Integer> planeDigit) {
		int temp = 0;
		for (TwoD cell : planeDigit.keySet()) {
			if (type.getFirst().x <= cell.x && cell.x <= type.getSecond().x && type.getFirst().y <= cell.y 
					&& cell.y <= type.getSecond().y) {
				if (function == "sum") {
					if (planeDigit.get(cell) != null) {
						temp += planeDigit.get(cell);
					}
				}
			}
		} 
		return temp;
	}
	
	private static void sumrecursivelytype4(Pair<TwoD, TwoD> shrink, Integer compareResult,
			Object function, Map<TwoD, Integer> planeDigit, int ymin, List<Integer> yaxis, List<Integer> recordmove, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound) {
		int tempy = shrink.getSecond().y;
		if (tempy > ymin) {
			int ry = yaxis.get(yaxis.indexOf(tempy)-1);
			TwoD newcutdown = new TwoD(shrink.getSecond().x, ry);
			Pair<TwoD, TwoD> newtype4 = new Pair<>(shrink.getFirst(), newcutdown);
			int tempsun = sumshrinkable(newtype4, function, planeDigit);
			int substract = tempsun - compareResult;
			recordmove.add(1);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype4, recordmove);
				superbound.add(sb);
			} else if (substract >0) {
				sumrecursivelytype4(newtype4, compareResult, function, planeDigit, ymin, yaxis, 
						recordmove, superbound);
			} 	
		}
	}
	
	public static <T extends Comparable<T>> Map<TwoD, Integer> producePlaneInteger(
			ListMultimap<Integer, Integer> fx, ListMultimap<Integer, Integer> fy, 
			Map<Integer, T> collect, Set<Integer> xaxiset, Set<Integer> yaxiset, Object function) {
		
		Map<TwoD, Integer> matrixmap = new HashMap<>();
		for (int x=0; x<fx.keySet().size(); x++) {
			for (int y=0; y<fy.keySet().size(); y++) {
				List<Integer> f = fx.get(x);
				List<Integer> g = fy.get(y);
				if (!f.contains(null) && !g.contains(null)) {
					List<Integer> c = TwoDimension.intersectArrays(f,g);
					//List<Integer> c = new ArrayList<>(f);
					//c.retainAll(g);
					if (!c.isEmpty()) {
						TwoD dua = new TwoD(x,y);
						xaxiset.add(x); yaxiset.add(y);
						if (function.equals("sum")) {
							int temp = 0;
							for (int cc : c) {
								temp += new Integer(collect.get(cc).toString());
							}
							matrixmap.put(dua, temp);
						} else if (function.equals("count")) {
							matrixmap.put(dua, c.size());
						}	
					}
				}
			}
		}
		return matrixmap;
	}
	
	private static void sumrecursivelytype3(Pair<TwoD, TwoD> shrink, int compareResult, Object function,
			Map<TwoD, Integer> planeDigit, int ymin, int ymax, List<Integer> yaxis, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound, List<Integer> recordmove) {
		int tempy = shrink.getFirst().y;
		if (tempy < ymax) {
			int ly = yaxis.get(yaxis.indexOf(tempy)+1);
			TwoD newcutup = new TwoD(shrink.getFirst().x, ly);
			Pair<TwoD, TwoD> newtype3 = new Pair<>(newcutup, shrink.getSecond());
			int tempsun = sumshrinkable(newtype3, function, planeDigit);
			int substract = tempsun - compareResult;
			recordmove.add(2);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype3, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				sumrecursivelytype4(newtype3, compareResult, function, planeDigit, ymin, yaxis, 
						recordmove1, superbound);
				sumrecursivelytype3(newtype3, compareResult, function, planeDigit, ymin, ymax, yaxis, 
						superbound, recordmove2);
			}
		}
	}
	
	private static void sumrecursivelytype2(Pair<TwoD, TwoD> shrink, Integer compareResult, Object function,
			Map<TwoD, Integer> planeDigit, int ymin, int ymax, List<Integer> yaxis, int xmin, List<Integer> xaxis,
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound, List<Integer> recordmove) {
		int tempx = shrink.getSecond().x;
		if (tempx > xmin) {
			int rx = xaxis.get(xaxis.indexOf(tempx)-1);
			TwoD newcutright = new TwoD(rx, shrink.getSecond().y);
			Pair<TwoD, TwoD> newtype2 = new Pair<>(shrink.getFirst(), newcutright);
			int tempsun = sumshrinkable(newtype2, function, planeDigit);
			int substract = tempsun - compareResult;
			recordmove.add(3);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype2, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
			
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				sumrecursivelytype4(newtype2, compareResult, function, planeDigit, ymin, yaxis, 
						recordmove1, superbound);
			
				sumrecursivelytype3(newtype2, compareResult, function, planeDigit, ymin, ymax, yaxis, 
						superbound, recordmove2);
				
				sumrecursivelytype2(newtype2, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				
			} 
		}
	}
	
	private static void sumrecursivelytype1(Pair<TwoD, TwoD> shrink, Integer compareResult, Object function,
			Map<TwoD, Integer> planeDigit, int ymin, int ymax, List<Integer> yaxis, int xmin, int xmax, 
			List<Integer> xaxis, List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound, List<Integer> recordmove) 
	{
		int tempx = shrink.getFirst().x;
		if (tempx < xmax) {
			int lx = xaxis.get(xaxis.indexOf(tempx)+1);
			TwoD newcutleft = new TwoD(lx, shrink.getFirst().y);
			Pair<TwoD, TwoD> newtype1 = new Pair<>(newcutleft, shrink.getSecond());
			int tempsun = sumshrinkable(newtype1, function, planeDigit);
			int substract = tempsun - compareResult;
			recordmove.add(4);
			if (tempsun == compareResult) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype1, recordmove);
				superbound.add(sb);
			} else if (substract > 0) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				sumrecursivelytype4(newtype1, compareResult, function, planeDigit, ymin, yaxis, 
						recordmove1, superbound);			
				sumrecursivelytype3(newtype1, compareResult, function, planeDigit, ymin, ymax, yaxis, 
						superbound, recordmove2);
				sumrecursivelytype2(newtype1, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				sumrecursivelytype1(newtype1, compareResult, function, planeDigit, ymin, ymax, yaxis, xmin, xmax, xaxis, 
						superbound, recordmove4);
			} 
		}	
	}
}
