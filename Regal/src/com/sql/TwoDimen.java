package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class TwoDimen {

	public static <T extends Comparable<T>> Multimap<Integer, DoubleBox<TwoD>> expansionMinMax ( Map<TwoD, T> cellxy, 
			List<TwoD> expandPoint, T compareResult, Object function, Integer rid, Integer bid, 
			int xa, int xb, int ya, int yb, Set<Integer> xaxiset, Set<Integer> yaxiset) 
	{
		
		Multimap<Integer, DoubleBox<TwoD>> multifunbox = ArrayListMultimap.create();
		
		List<Integer> xaxis = new ArrayList<Integer>(xaxiset);
		List<Integer> yaxis = new ArrayList<Integer>(yaxiset);
		Collections.sort(xaxis);
		Collections.sort(yaxis);
		
		int xmin = xaxis.get(0); int xmax = xaxis.get(xaxis.size()-1);
		int ymin = yaxis.get(0); int ymax = yaxis.get(yaxis.size()-1);
		
		List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound = new ArrayList<>();
		
		for (TwoD ep : expandPoint) {
			int x = ep.x;
			int y = ep.y;
			
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound_down = new ArrayList<>();
			
			if (y < ymax) {
				int ry = yaxis.get(yaxis.indexOf(y)+1);
				TwoD expdown = new TwoD(x,ry);
				List<Integer> recordmove = new ArrayList<Integer>();
				recordmove.add(1);
				Pair<TwoD, TwoD> godown = new Pair<>(ep, expdown);
				if (extendable(godown, compareResult, function, cellxy)) {
					recursivelytype4(godown, compareResult, function, cellxy, ymax, yaxis, recordmove, bound_down, bound);
				}
			}
			
			if (y > ymin) {
				int ly = yaxis.get(yaxis.indexOf(y)-1);
				TwoD expup = new TwoD(x,ly);
				List<Integer> recordmove = new ArrayList<Integer>();
				recordmove.add(2);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				Pair<TwoD, TwoD> goup = new Pair<>(expup, ep);
				if (extendable(goup, compareResult, function, cellxy)) {
					recursivelytype4(goup, compareResult, function, cellxy, ymax, yaxis, recordmove, bound_down, bound);
					recursivelytype3(goup, compareResult, function, cellxy, ymax, ymin, yaxis, 
							bound_down, recordmove2, bound);
				}
			}
			
			if (x < xmax) {
				int rx = xaxis.get(xaxis.indexOf(x)+1);
				TwoD expright = new TwoD(rx,y);
				List<Integer> recordmove = new ArrayList<Integer>();
				recordmove.add(3);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				Pair<TwoD, TwoD> goright = new Pair<>(ep, expright);
				if (extendable(goright, compareResult, function, cellxy)) {
					recursivelytype4(goright, compareResult, function, cellxy, ymax, yaxis, recordmove, bound_down, bound);
					recursivelytype3(goright, compareResult, function, cellxy, ymax, ymin, yaxis, 
							bound_down, recordmove2, bound);
					recursivelytype2(goright, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xaxis, 
							bound_down, recordmove3, bound);
				}
			}
			
			if (x > xmin) {
				int lx = xaxis.get(xaxis.indexOf(x)-1);
				TwoD expleft = new TwoD(lx,y);
				Pair<TwoD, TwoD> goleft = new Pair<>(expleft, ep);
				List<Integer> recordmove = new ArrayList<Integer>();
				recordmove.add(4);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				if (extendable(goleft, compareResult, function, cellxy)) {
					recursivelytype4(goleft, compareResult, function, cellxy, ymax, yaxis, 
							recordmove, bound_down, bound);
					recursivelytype3(goleft, compareResult, function, cellxy, ymax, ymin, yaxis, bound_down, 
							recordmove2, bound);
					recursivelytype2(goleft, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xaxis, 
							bound_down, recordmove3, bound);
					recursivelytype1(goleft, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xmin, xaxis, 
							bound_down, recordmove4, bound);
				}
			}
			
			if (!bound_down.isEmpty()) {
				List<Integer> depth = new ArrayList<>();
				for(Pair<Pair<TwoD,TwoD>, List<Integer>> not : bound_down) {
					depth.add(not.getSecond().size());
				}
				int maxdepth = Collections.max(depth);
				for (int d=0; d < depth.size(); d++) {
					if (depth.get(d) == maxdepth) {
						bound.add(bound_down.get(d));
					}
				}
				if (!bound.isEmpty()) {
					for (int d = 0; d < depth.size(); d++) {
						if (depth.get(d) < maxdepth) {
							boolean store = true;
							for (Pair<Pair<TwoD,TwoD>, List<Integer>> not : bound) {
								if (not.getSecond().containsAll(bound_down.get(d).getSecond()))
									store = false;
							}
							if (store) bound.add(bound_down.get(d));
						}
					}
				}
			}	
		}
		
		for(Pair<Pair<TwoD,TwoD>, List<Integer>> not : bound) {
			
			if (not.getFirst().getFirst().x == xmin) {
				not.getFirst().getFirst().x = xa;
			}
			if (not.getFirst().getSecond().x == xmax) {
				not.getFirst().getSecond().x = xb;
			}
			if (not.getFirst().getFirst().y == ymin) {
				not.getFirst().getFirst().y = ya;
			}
			if (not.getFirst().getSecond().y == ymax) {
				not.getFirst().getSecond().y = yb;
			}
			
			for (TwoD poin : expandPoint) {
				if (poin.x <= not.getFirst().getSecond().x && poin.x >= not.getFirst().getFirst().x && 
						poin.y <= not.getFirst().getSecond().y && poin.y >= not.getFirst().getFirst().y) {
					Pair<TwoD, TwoD> minmin = new Pair<>(poin, poin);
					DoubleBox<TwoD> minmax = new DoubleBox<TwoD>(minmin, not.getFirst());
					
					multifunbox.put(rid, minmax);
				}
			}	
		}
		return multifunbox;
	}
	
	private static <T extends Comparable<T>> void recursivelytype1(Pair<TwoD, TwoD> type1, T compareResult, 
			Object function, Map<TwoD, T> cellxy, int ymax, int ymin, List<Integer> yaxis, 
			int xmax, int xmin, List<Integer> xaxis,
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound_down, List<Integer> recordmove, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound) {
		int tempx = type1.getFirst().x;
		if (tempx > xmin) {
			int lx = xaxis.get(xaxis.indexOf(tempx)-1);
			TwoD newexpleft = new TwoD(lx, type1.getFirst().y);
			Pair<TwoD, TwoD> newtype1 = new Pair<>(newexpleft, type1.getSecond());
			if (extendable(newtype1, compareResult, function, cellxy)) {
				recordmove.add(4);
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				recursivelytype4(newtype1, compareResult, function, cellxy, ymax, yaxis, 
						recordmove1, bound_down, bound);
				recursivelytype3(newtype1, compareResult, function, cellxy, ymax, ymin, yaxis, 
						bound_down, recordmove2, bound);	
				recursivelytype2(newtype1, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xaxis, 
						bound_down, recordmove3, bound);					
				recursivelytype1(newtype1, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xmin, xaxis, 
						bound_down, recordmove4, bound);
				
			}
		}
	}
	
	private static <T extends Comparable<T>> void recursivelytype2(Pair<TwoD, TwoD> type2, T compareResult,
			Object function, Map<TwoD, T> cellxy, int ymax, int ymin, List<Integer> yaxis, int xmax, List<Integer> xaxis, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound_down, List<Integer> recordmove, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound) {
		int tempx = type2.getSecond().x;
		if (tempx < xmax) {
			int rx = xaxis.get(xaxis.indexOf(tempx)+1);
			TwoD newexpright = new TwoD(rx, type2.getSecond().y);
			Pair<TwoD, TwoD> newtype2 = new Pair<>(type2.getFirst(), newexpright);
			
			if (extendable(newtype2, compareResult, function, cellxy)) {
				recordmove.add(3);
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				recursivelytype4(newtype2, compareResult, function, cellxy, ymax, yaxis, recordmove1, 
						bound_down, bound);
				recursivelytype3(newtype2, compareResult, function, cellxy, ymax, ymin, yaxis, 
						bound_down, recordmove2, bound);
				recursivelytype2(newtype2, compareResult, function, cellxy, ymax, ymin, yaxis, xmax, xaxis, 
						bound_down, recordmove3, bound);	
			}
		}
	}
	
	private static <T extends Comparable<T>> void recursivelytype3(Pair<TwoD, TwoD> type3, T compareResult,
			Object function, Map<TwoD, T> cellxy, int ymax, int ymin, List<Integer> yaxis, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound_down, 
			List<Integer> recordmove, List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound) {
		int tempy = type3.getFirst().y;
		if (tempy > ymin) {
			int ly = yaxis.get(yaxis.indexOf(tempy)-1);
			TwoD newexpup = new TwoD(type3.getFirst().x, ly);
			Pair<TwoD, TwoD> newtype3 = new Pair<>(newexpup, type3.getSecond());
			//Pair<TwoD, TwoD> duptype3 = newtype3;
			if (extendable(newtype3, compareResult, function, cellxy)) {
				recordmove.add(2);
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				recursivelytype4(newtype3, compareResult, function, cellxy, ymax, yaxis, recordmove1, bound_down, bound);				
				recursivelytype3(newtype3, compareResult, function, cellxy, ymax, ymin, yaxis, 
						bound_down, recordmove2, bound);
				
			} 
		} 
	}
	
	private static <T extends Comparable<T>> void recursivelytype4(Pair<TwoD, TwoD> type4, T compareResult, 
			Object function, Map<TwoD, T> cellxy, int ymax, List<Integer> yaxis, List<Integer> recordmove, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound_down, 
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound) {
		int tempy = type4.getSecond().y;
		
		if (tempy < ymax) {
			int ry = yaxis.get(yaxis.indexOf(tempy)+1);
			TwoD newexpdown = new TwoD(type4.getSecond().x, ry);
			
			Pair<TwoD, TwoD> newtype4 = new Pair<>(type4.getFirst(), newexpdown);
			boolean extend = extendable(newtype4, compareResult, function, cellxy);
			
			if (extend == true) {
				recordmove.add(1);
				recursivelytype4(newtype4, compareResult, function, cellxy, ymax, yaxis, recordmove, bound_down, bound);
			} else {
				Pair<Pair<TwoD, TwoD>, List<Integer>> type4new = new Pair<>(type4, recordmove);
				if (storenewpoint(bound, type4new))
					bound_down.add(type4new);
			}
		} else {
			Pair<Pair<TwoD, TwoD>, List<Integer>> type4new = new Pair<>(type4, recordmove);
			if (storenewpoint(bound, type4new))
				bound_down.add(type4new);
		}
		
	}
	
	private static boolean storenewpoint(
			List<Pair<Pair<TwoD, TwoD>, List<Integer>>> bound,
			Pair<Pair<TwoD, TwoD>, List<Integer>> type4new) {
		boolean store = true;
		if (!bound.isEmpty()) {
			for (Pair<Pair<TwoD, TwoD>, List<Integer>> bd: bound) {
				Pair<TwoD, TwoD> startend = bd.getFirst();
				if (type4new.getFirst().getFirst().x >= startend.getFirst().x 
						&& type4new.getFirst().getSecond().x  <= startend.getSecond().x &&
						type4new.getFirst().getFirst().y >= startend.getFirst().y 
						&& type4new.getFirst().getSecond().y <= startend.getSecond().y) {
					store = false;
					return store;
				}
			}
		}
		return store;		
	}
	
	private static <T extends Comparable<T>> boolean extendable(Pair<TwoD, TwoD> type, 
			T compareResult, Object function, Map<TwoD, T> cellxy) {
		boolean extendable = true;
		for (TwoD cell : cellxy.keySet()) {
			if (type.getFirst().x <= cell.x && cell.x <= type.getSecond().x 
					&& type.getFirst().y <= cell.y && cell.y <= type.getSecond().y) {
				if (function == "max") {
					if (cellxy.get(cell) != null && cellxy.get(cell).compareTo(compareResult) > 0) {
						//box generated
						extendable = false;
						break;
					}
				} else if (function == "min") {
					if (cellxy.get(cell) != null && cellxy.get(cell).compareTo(compareResult) < 0) {
						//box generated
						extendable = false;
						break;
					}
				}
			}
		}
		return extendable;
	}
}
