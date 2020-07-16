package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class ThreeDimenCount {
	public static Multimap<Integer, DoubleBox<ThreeD>> countshrinking(
			Map<ThreeD, Integer> cubeInteger, int intResult, Object function,
			Integer rid, Integer bid, int xa, int xb, int ya, int yb, int za, int zb,
			Set<Integer> xaxiset, Set<Integer> yaxiset, Set<Integer> zaxiset) {
		Multimap<Integer, DoubleBox<ThreeD>> multifunbox = ArrayListMultimap.create();
		List<Integer> xaxis = new ArrayList<>(xaxiset);
		List<Integer> yaxis = new ArrayList<>(yaxiset);
		List<Integer> zaxis = new ArrayList<>(zaxiset);
		Collections.sort(xaxis);
		Collections.sort(yaxis);
		Collections.sort(zaxis);
		int xmin = xaxis.get(0);	int xmax = xaxis.get(xaxis.size()-1);
		int ymin = yaxis.get(0);	int ymax = yaxis.get(yaxis.size()-1);
		int zmin = zaxis.get(0);	int zmax = zaxis.get(zaxis.size()-1);
		ThreeD head = new ThreeD(xmin, ymin, zmin);
		ThreeD tail = new ThreeD(xmax, ymax, zmax);
		List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound = new ArrayList<>();
		if (zmax > zmin) {
			int rz = zaxis.get(zaxis.indexOf(zmax)-1);
			ThreeD cutback = new ThreeD(xmax, ymax, rz);
			Pair<ThreeD, ThreeD> shrinkback = new Pair<>(head, cutback);
			int tempsum = cntshrinkable(shrinkback, intResult, function, cubeInteger);
			//BigDecimal substract = tempsum.subtract(compareBig);
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(1);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkback, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkback, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
			}
		}
		if (zmin < zmax) {
			int lz = zaxis.get(zaxis.indexOf(zmin)+1);
			ThreeD cutfront = new ThreeD(xmin, ymin, lz);
			Pair<ThreeD, ThreeD> shrinkfront = new Pair<>(cutfront, tail);
			int tempsum = cntshrinkable(shrinkfront, intResult, function, cubeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(2);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkfront, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkfront, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
				cntrecursivelytype5(shrinkfront, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
			}
		}
		if (ymax > ymin) {
			int ry = yaxis.get(yaxis.indexOf(ymax)-1);
			ThreeD cutdown = new ThreeD(xmax, ry, zmax);
			Pair<ThreeD, ThreeD> shrinkdown = new Pair<>(head, cutdown);
			int tempsum = cntshrinkable(shrinkdown, intResult, function, cubeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(3);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkdown, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkdown, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
				cntrecursivelytype5(shrinkdown, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(shrinkdown, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
			}
		}
		if (ymin < ymax) {
			int ly = yaxis.get(yaxis.indexOf(ymin)+1);
			ThreeD cutup = new ThreeD(xmin, ly, zmin);
			Pair<ThreeD, ThreeD> shrinkup = new Pair<>(cutup, tail);
			int tempsum = cntshrinkable(shrinkup, intResult, function, cubeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(4);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			List<Integer> recordmove4 = new ArrayList<>(recordmove);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkup, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkup, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
				cntrecursivelytype5(shrinkup, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(shrinkup, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(shrinkup, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
			}
		}
		if (xmax > xmin) {
			int rx = xaxis.get(xaxis.indexOf(xmax)-1);
			ThreeD cutright = new ThreeD(rx, ymax, zmax);
			Pair<ThreeD, ThreeD> shrinkright = new Pair<>(head, cutright);
			int tempsum = cntshrinkable(shrinkright, intResult, function, cubeInteger);
			
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(5);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			List<Integer> recordmove4 = new ArrayList<>(recordmove);
			List<Integer> recordmove5 = new ArrayList<>(recordmove);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkright, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkright, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
				cntrecursivelytype5(shrinkright, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(shrinkright, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(shrinkright, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
				cntrecursivelytype2(shrinkright, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xaxis, recordmove5, superbound);
			}
		}
		if (xmin < xmax) {
			int lx = xaxis.get(xaxis.indexOf(xmin)+1);
			ThreeD cutleft = new ThreeD(lx, ymin, zmin);
			Pair<ThreeD, ThreeD> shrinkleft = new Pair<>(cutleft, tail);
			int tempsum = cntshrinkable(shrinkleft, intResult, function, cubeInteger);
			List<Integer> recordmove = new ArrayList<>();
			recordmove.add(6);
			List<Integer> recordmove2 = new ArrayList<>(recordmove);
			List<Integer> recordmove3 = new ArrayList<>(recordmove);
			List<Integer> recordmove4 = new ArrayList<>(recordmove);
			List<Integer> recordmove5 = new ArrayList<>(recordmove);
			List<Integer> recordmove6 = new ArrayList<>(recordmove);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(shrinkleft, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(shrinkleft, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
				cntrecursivelytype5(shrinkleft, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(shrinkleft, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(shrinkleft, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
				cntrecursivelytype2(shrinkleft, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xaxis, recordmove5, superbound);
				cntrecursivelytype1(shrinkleft, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xmax, xaxis, recordmove6, superbound);
			}
		}
		List<Integer> depth = new ArrayList<>();
		for (Pair<Pair<ThreeD, ThreeD>, List<Integer>> not : superbound) {
			depth.add(not.getSecond().size());	
		}
		List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> totalbound = new ArrayList<>();
		if (!depth.isEmpty()) {
			System.out.println(depth);
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
						for (Pair<Pair<ThreeD,ThreeD>, List<Integer>> bound : totalbound) {
							//@@@@
							if (superbound.get(d).getSecond().containsAll(bound.getSecond()) )  
								store = false;
							
						}
						if (store) totalbound.add(superbound.get(d));
					}
				}
			}
		}
		for (Pair<Pair<ThreeD, ThreeD>, List<Integer>> bound : totalbound) {
			Pair<ThreeD, ThreeD> maxbx = bound.getFirst();
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
			if (maxbx.getFirst().z == zmin) {
				maxbx.getFirst().z = za;
			}
			if (maxbx.getSecond().z == zmax) {
				maxbx.getSecond().z = zb;
			}
			List<Integer> bx = new ArrayList<>();
			List<Integer> by = new ArrayList<>();
			List<Integer> bz = new ArrayList<>();
			for (ThreeD cell : cubeInteger.keySet()) {
				if (maxbx.getFirst().x <= cell.x && cell.x <= maxbx.getSecond().x && 
						maxbx.getFirst().y <= cell.y && cell.y <= maxbx.getSecond().y &&
						maxbx.getFirst().z <= cell.z && cell.z <= maxbx.getSecond().z) {
					bx.add(cell.x);
					by.add(cell.y);
					bz.add(cell.z);
				}
			}
			ThreeD minp = new ThreeD(Collections.min(bx), Collections.min(by), Collections.min(bz));
			ThreeD maxp = new ThreeD(Collections.max(bx), Collections.max(by), Collections.max(bz));
			Pair<ThreeD, ThreeD> minbx = new Pair<>(minp, maxp);
			DoubleBox<ThreeD> minmax = new DoubleBox<>(minbx, maxbx);
			/**System.out.println("RID " + rid + "[" + minmax.minX.getFirst().x + " " + minmax.minX.getFirst().y + "] [" + 
					minmax.minX.getSecond().x + " " + minmax.minX.getSecond().y + "] [" + 
					minmax.maxX.getFirst().x + " " + minmax.maxX.getFirst().y + "] [" +
					minmax.maxX.getSecond().x + " " + minmax.maxX.getSecond().y + "]"); */
			
			multifunbox.put(rid, minmax);
		}
		return multifunbox;
	}
	
	private static int cntshrinkable(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger) {
		int temp = 0;
		for (ThreeD cell : cubeInteger.keySet()) {
			if (shrink.getFirst().x <= cell.x && cell.x <= shrink.getSecond().x &&
					shrink.getFirst().y <= cell.y && cell.y <= shrink.getSecond().y &&
					shrink.getFirst().z <= cell.z && cell.z <= shrink.getSecond().z) {
				if (function == "count") {
					if (cubeInteger.get(cell) != null) {
						temp += cubeInteger.get(cell);
					}
				}
			}
		}
		return temp;
	}
	
	private static void cntrecursivelytype1(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, int zmax, List<Integer> zaxis, int ymin, int ymax,
			List<Integer> yaxis, int xmin, int xmax, List<Integer> xaxis,
			List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempx = shrink.getFirst().x;
		if (tempx < xmax) {
			int lx = xaxis.get(xaxis.indexOf(tempx)+1);
			ThreeD newcutleft = new ThreeD(lx, shrink.getFirst().y, shrink.getFirst().z);
			Pair<ThreeD, ThreeD> newtype1 = new Pair<>(newcutleft, shrink.getSecond());
			int tempsum = cntshrinkable(newtype1, intResult, function, cubeInteger);
			
			recordmove.add(6);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype1, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				List<Integer> recordmove5 = new ArrayList<>(recordmove);
				List<Integer> recordmove6 = new ArrayList<>(recordmove);
				cntrecursivelytype6(newtype1, intResult, function, cubeInteger, zmin, zaxis, recordmove1, superbound);
				cntrecursivelytype5(newtype1, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(newtype1, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(newtype1, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
				cntrecursivelytype2(newtype1, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xaxis, recordmove5, superbound);
				cntrecursivelytype1(newtype1, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xmax, xaxis, recordmove6, superbound);
			} 
		}
		
	}
	
	private static void cntrecursivelytype2(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, int zmax, List<Integer> zaxis, int ymin, int ymax,
			List<Integer> yaxis, int xmin, List<Integer> xaxis,
			List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempx = shrink.getSecond().x;
		if (tempx > xmin) {
			int rx = xaxis.get(xaxis.indexOf(tempx)-1);
			ThreeD newcutright = new ThreeD(rx, shrink.getSecond().y, shrink.getSecond().z);
			Pair<ThreeD, ThreeD> newtype2 = new Pair<>(shrink.getFirst(), newcutright);
			int tempsum = cntshrinkable(newtype2, intResult, function, cubeInteger);
			recordmove.add(5);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype2, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
			
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				List<Integer> recordmove5 = new ArrayList<>(recordmove);
				cntrecursivelytype6(newtype2, intResult, function, cubeInteger, zmin, zaxis, recordmove1, superbound);
				cntrecursivelytype5(newtype2, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(newtype2, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(newtype2, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
				cntrecursivelytype2(newtype2, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						xmin, xaxis, recordmove5, superbound);
			} 
		}
	}
	
	private static void cntrecursivelytype3(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, int zmax, List<Integer> zaxis, int ymin, int ymax,
			List<Integer> yaxis, List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempy = shrink.getFirst().y;
		if (tempy < ymax) {
			int ly = yaxis.get(yaxis.indexOf(tempy)+1);
			ThreeD newcutup = new ThreeD(shrink.getFirst().x, ly, shrink.getFirst().z);
			Pair<ThreeD, ThreeD> newtype3 = new Pair<>(newcutup, shrink.getSecond());
			int tempsum = cntshrinkable(newtype3, intResult, function, cubeInteger);
			recordmove.add(4);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype3, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				List<Integer> recordmove4 = new ArrayList<>(recordmove);
				cntrecursivelytype6(newtype3, intResult, function, cubeInteger, zmin, zaxis, recordmove1, superbound);
				cntrecursivelytype5(newtype3, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(newtype3, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
				cntrecursivelytype3(newtype3, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, ymax, yaxis, 
						recordmove4, superbound);
			}
		}
	}

	private static void cntrecursivelytype4(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, int zmax, List<Integer> zaxis, int ymin,
			List<Integer> yaxis, List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempy = shrink.getSecond().y;
		if (tempy > ymin) {
			int ry = yaxis.get(yaxis.indexOf(tempy)-1);
			ThreeD newcutdown = new ThreeD(shrink.getSecond().x, ry, shrink.getSecond().z);
			Pair<ThreeD, ThreeD> newtype4 = new Pair<>(shrink.getFirst(), newcutdown);
			int tempsum = cntshrinkable(newtype4, intResult, function, cubeInteger);
			recordmove.add(3);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype4, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				List<Integer> recordmove3 = new ArrayList<>(recordmove);
				cntrecursivelytype6(newtype4, intResult, function, cubeInteger, zmin, zaxis, recordmove1, superbound);
				cntrecursivelytype5(newtype4, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
				cntrecursivelytype4(newtype4, intResult, function, cubeInteger, zmin, zmax, zaxis, ymin, yaxis, 
						recordmove3, superbound);
			} 	
		}	
	}

	private static void cntrecursivelytype5(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, int zmax, List<Integer> zaxis, List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempz = shrink.getFirst().z;
		if (tempz < zmax) {
			int lz = zaxis.get(zaxis.indexOf(tempz)+1);
			ThreeD newcutfront = new ThreeD(shrink.getFirst().x, shrink.getFirst().y, lz);
			Pair<ThreeD, ThreeD> newtype5 = new Pair<>(newcutfront, shrink.getSecond());
			int tempsum = cntshrinkable(newtype5, intResult, function, cubeInteger);
			recordmove.add(2);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype5, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				List<Integer> recordmove1 = new ArrayList<>(recordmove);
				List<Integer> recordmove2 = new ArrayList<>(recordmove);
				cntrecursivelytype6(newtype5, intResult, function, cubeInteger, zmin, zaxis, recordmove1, superbound);
				cntrecursivelytype5(newtype5, intResult, function, cubeInteger, zmin, zmax, zaxis, recordmove2, superbound);
			}
		}
	}
	
	private static void cntrecursivelytype6(Pair<ThreeD, ThreeD> shrink,
			int intResult, Object function, Map<ThreeD, Integer> cubeInteger,
			int zmin, List<Integer> zaxis, List<Integer> recordmove,
			List<Pair<Pair<ThreeD, ThreeD>, List<Integer>>> superbound) {
		int tempz = shrink.getSecond().z;
		if (tempz > zmin) {
			int rz = zaxis.get(zaxis.indexOf(tempz)-1);
			ThreeD newcutback = new ThreeD(shrink.getSecond().x, shrink.getSecond().y, rz);
			Pair<ThreeD, ThreeD> newtype6 = new Pair<>(shrink.getFirst(), newcutback);
			int tempsum = cntshrinkable(newtype6, intResult, function, cubeInteger);
			
			recordmove.add(1);
			if (tempsum == intResult) {
				Pair<Pair<ThreeD, ThreeD>, List<Integer>> sb = new Pair<>(newtype6, recordmove);
				superbound.add(sb);
			} else if (tempsum > intResult) {
				cntrecursivelytype6(newtype6, intResult, function, cubeInteger, zmin, zaxis, recordmove, superbound);
			} 	
		}
		
	}

}
