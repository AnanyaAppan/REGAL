package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class TwoDimenCount {
	public static Multimap<Integer, DoubleBox<TwoD>> countshrinking(
			Map<TwoD, Integer> planeInteger, int countvalue, Integer rid,Integer bid, 
			int xa, int xb, int ya, int yb, Set<Integer> xaxiset, Set<Integer> yaxiset) 
	{
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
			int tempcount = countshrinkable(shrinkdown, countvalue, planeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(1);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkdown, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				cntrecursivelytype4(shrinkdown, countvalue, planeInteger, ymin, yaxis, recordmove, superbound);
			} 
		}
		if (ymin < ymax) {
			int ly = yaxis.get(yaxis.indexOf(ymin)+1);
			TwoD cutup = new TwoD(xmin, ly);
			Pair<TwoD, TwoD> shrinkup = new Pair<>(cutup, second);
			int tempcount = countshrinkable(shrinkup, countvalue, planeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(2);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkup, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				cntrecursivelytype4(shrinkup, countvalue, planeInteger, ymax, yaxis, recordmove, superbound);
				cntrecursivelytype3(shrinkup, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
			} 
		}
		if (xmax > xmin) {
			int rx = xaxis.get(xaxis.indexOf(xmax)-1);
			TwoD cutright = new TwoD(rx, ymax);
			Pair<TwoD, TwoD> shrinkright = new Pair<>(first, cutright);
			int tempcount = countshrinkable(shrinkright, countvalue, planeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(3);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkright, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				cntrecursivelytype4(shrinkright, countvalue, planeInteger, ymax, yaxis, recordmove, superbound);
				cntrecursivelytype3(shrinkright, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
				cntrecursivelytype2(shrinkright, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
			} 
		}
		if (xmin < xmax) {
			int lx = xaxis.get(xaxis.indexOf(xmin)+1);
			TwoD cutleft = new TwoD(lx, ymin);
			Pair<TwoD, TwoD> shrinkleft = new Pair<>(cutleft, second);
			int tempcount = countshrinkable(shrinkleft, countvalue, planeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(4);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			List<Integer> recordmove4 = new ArrayList<>(recordmove);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(shrinkleft, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				cntrecursivelytype4(shrinkleft, countvalue, planeInteger, ymax, yaxis, recordmove, superbound);
				cntrecursivelytype3(shrinkleft, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
				cntrecursivelytype2(shrinkleft, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				cntrecursivelytype1(shrinkleft, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xmax, xaxis, 
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
						if (superbound.get(d).getSecond().containsAll(bound.getSecond())) 
							store = false;
						
					}
					if (store) totalbound.add(superbound.get(d));
				}
			}
		}
		}
		/**Pair<Object, Integer> fungsi = new Pair<>(function, bid);*/
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
			for (TwoD cell : planeInteger.keySet()) {
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
			/**System.out.println("RID " + rid + "[" + minmax.minX.getFirst().x + " " + minmax.minX.getFirst().y + "] [" + 
					minmax.minX.getSecond().x + " " + minmax.minX.getSecond().y + "] [" + 
					minmax.maxX.getFirst().x + " " + minmax.maxX.getFirst().y + "] [" +
					minmax.maxX.getSecond().x + " " + minmax.maxX.getSecond().y + "]"); */
			
			multifunbox.put(rid, minmax);
		}
		return multifunbox;
	}
	
	private static void cntrecursivelytype1(Pair<TwoD, TwoD> shrink,
			int countvalue, Map<TwoD, Integer> planeInteger, int ymin,
			int ymax, List<Integer> yaxis, int xmin, int xmax,
			List<Integer> xaxis, List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound,
			List<Integer> recordmove) 
	{
		int tempx = shrink.getFirst().x;
		if (tempx < xmax) {
			int lx = xaxis.get(xaxis.indexOf(tempx)+1);
			TwoD newcutleft = new TwoD(lx, shrink.getFirst().y);
			Pair<TwoD, TwoD> newtype1 = new Pair<>(newcutleft, shrink.getSecond());
			int tempcount = countshrinkable(shrink, countvalue, planeInteger);
			//BigDecimal tempsun = sumshrinkable(newtype1, decimalResult, function, planeDecimal);
			//BigDecimal substract = tempsun.subtract(decimalResult);
			recordmove.add(4);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype1, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				cntrecursivelytype4(newtype1, countvalue, planeInteger, ymin, yaxis, recordmove1, superbound);			
				cntrecursivelytype3(newtype1, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
				cntrecursivelytype2(newtype1, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xaxis, 
						superbound, recordmove3);
				cntrecursivelytype1(newtype1, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xmax, xaxis, 
						superbound, recordmove4);
			
			} 
		}	
	}

	private static void cntrecursivelytype2(Pair<TwoD, TwoD> shrink,
			int countvalue, Map<TwoD, Integer> planeInteger, int ymin,
			int ymax, List<Integer> yaxis, int xmin, List<Integer> xaxis,
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound, List<Integer> recordmove) 
	{
		int tempx = shrink.getSecond().x;
		if (tempx > xmin) {
			int rx = xaxis.get(xaxis.indexOf(tempx)-1);
			TwoD newcutright = new TwoD(rx, shrink.getSecond().y);
			Pair<TwoD, TwoD> newtype2 = new Pair<>(shrink.getFirst(), newcutright);
			int tempcount = countshrinkable(shrink, countvalue, planeInteger);
			//BigDecimal tempsun = sumshrinkable(newtype2, decimalResult, function, planeDecimal);
			//BigDecimal substract = tempsun.subtract(decimalResult);
			recordmove.add(3);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sibeh = new Pair<>(newtype2, recordmove);
				superbound.add(sibeh);
			} else if (tempcount > countvalue) {
			
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				cntrecursivelytype4(newtype2, countvalue, planeInteger, ymin, yaxis, recordmove1, superbound);
				cntrecursivelytype3(newtype2, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
				cntrecursivelytype2(newtype2, countvalue, planeInteger, ymin, ymax, yaxis, xmin, xaxis, superbound, recordmove3);	
			} 
		}	
	}

	private static void cntrecursivelytype3(Pair<TwoD, TwoD> shrink,
			int countvalue, Map<TwoD, Integer> planeInteger, int ymin,
			int ymax, List<Integer> yaxis, List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound,
			List<Integer> recordmove) 
	{
		int tempy = shrink.getFirst().y;
		if (tempy < ymax) {
			int ly = yaxis.get(yaxis.indexOf(tempy)+1);
			TwoD newcutup = new TwoD(shrink.getFirst().x, ly);
			Pair<TwoD, TwoD> newtype3 = new Pair<>(newcutup, shrink.getSecond());
			int tempcount = countshrinkable(shrink, countvalue, planeInteger);
			//BigDecimal tempsun = sumshrinkable(newtype3, decimalResult, function, planeDecimal);
			//BigDecimal substract = tempsun.subtract(decimalResult);
			recordmove.add(2);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype3, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				cntrecursivelytype4(newtype3, countvalue, planeInteger, ymin, yaxis, recordmove1, superbound);
				cntrecursivelytype3(newtype3, countvalue, planeInteger, ymin, ymax, yaxis, superbound, recordmove2);
			}
		}
	}

	private static void cntrecursivelytype4(Pair<TwoD, TwoD> shrink,
			int countvalue, Map<TwoD, Integer> planeInteger, int ymin,
			List<Integer> yaxis, List<Integer> recordmove,
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> superbound) {
		int tempy = shrink.getSecond().y;
		if (tempy > ymin) {
			int ry = yaxis.get(yaxis.indexOf(tempy)-1);
			TwoD newcutdown = new TwoD(shrink.getSecond().x, ry);
			Pair<TwoD, TwoD> newtype4 = new Pair<>(shrink.getFirst(), newcutdown);
			int tempcount = countshrinkable(shrink, countvalue, planeInteger);
			//BigDecimal tempsun = sumshrinkable(newtype4, decimalResult, function, planeDecimal);
			//BigDecimal substract = tempsun.subtract(decimalResult);
			recordmove.add(1);
			if (tempcount == countvalue) {
				Pair<Pair<TwoD, TwoD>, List<Integer>> sb = new Pair<>(newtype4, recordmove);
				superbound.add(sb);
			} else if (tempcount > countvalue) {
				cntrecursivelytype4(newtype4, countvalue, planeInteger, ymin, yaxis, recordmove, superbound);
			} 	
		}	
	}

	
	private static <T extends Comparable<T>> int countshrinkable(Pair<TwoD, TwoD> type, int compareResult, 
			Map<TwoD, Integer> planeInteger) {
		int temp = 0;
		for (TwoD cell : planeInteger.keySet()) {
			if (type.getFirst().x <= cell.x && cell.x <= type.getSecond().x 
					&& type.getFirst().y <= cell.y && cell.y <= type.getSecond().y) {
				if (planeInteger.get(cell) != null) {
					temp += planeInteger.get(cell);
				}
			}
		} 
		return temp;
	}
}
