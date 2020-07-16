package com.sql;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;

public class ThreeDimension {
	public static <T extends Comparable<T>> 
	Map<Pair<ArrayList<Integer>, DoubleBox<ThreeD>>, List<Pair<Object, Integer>>> cubeCut(
			List<Integer> nongroupdimensi, 
			Multimap<List<String>, DimenIndex<T>> ridTupleIndex,
			Map<Integer, String> baseClass, 
			List<Map<Integer, Pair<Object, Integer>>> whichaggre, List<Integer> ridorder){

	Map<Pair<ArrayList<Integer>, DoubleBox<ThreeD>>, List<Pair<Object, Integer>>> whereTiga = new HashMap<>();
	Integer[] pairDimension = nongroupdimensi.toArray(new Integer[nongroupdimensi.size()]);
	ArrayList<ArrayList<Integer>> dimenThrice = PreprocessBase.combination(pairDimension, 3);	
	for(ArrayList<Integer> anyCube: dimenThrice) {
		List<T> firstAxis = new ArrayList<>();
		List<T> secondAxis = new ArrayList<>();
		List<T> thirdAxis = new ArrayList<>();
		
		for (Map<Integer, Pair<Object, Integer>> aggpernode : whichaggre) {
			List<Pair<Object, Integer>> functioname = new ArrayList<>(aggpernode.values());
			List<DoubleBox<ThreeD>> updateCube = new ArrayList<>(); //for cross vaildation all
			outer:
			for (List<String> ridTuple : ridTupleIndex.keySet()) {
				Collection<DimenIndex<T>> valueIndex = ridTupleIndex.get(ridTuple);
				ListMultimap<Integer, Integer> fx = ArrayListMultimap.create();
				ListMultimap<Integer, Integer> fy = ArrayListMultimap.create(); 
				ListMultimap<Integer, Integer> fz = ArrayListMultimap.create();
				Map<Integer, Map<Integer, T>> collect = new HashMap<>();
				Map<Integer, T> ridValue = new HashMap<>();
				for (DimenIndex<T> vi : valueIndex) {
					collect.put(vi.index, vi.indexMatch);
					if (vi.index == anyCube.get(0)) {
						ridValue = vi.ridValue;
						firstAxis = vi.letT;
						fx = vi.newBuildD;
					} else if (vi.index == anyCube.get(1)) {
						secondAxis = vi.letT;
						fy = vi.newBuildD;
					} else if (vi.index == anyCube.get(2)) {
						thirdAxis = vi.letT;
						fz = vi.newBuildD;
					}
				}
				for (int jr: ridorder) {
					Pair<Object, Integer> fungsi = aggpernode.get(jr);
					System.out.println(anyCube + " " + functioname + " " + ridTuple + " " + fungsi + " " + ridValue.get(jr) ); 
					
					if (updateCube.isEmpty()) { 
						Multimap<Integer, DoubleBox<ThreeD>> forone = 
								cubeConstruction(fx, fy, fz, collect.get(fungsi.getSecond()), firstAxis, secondAxis, thirdAxis, 
								fungsi, ridValue.get(jr), baseClass.get(fungsi.getSecond()), fungsi.getSecond(), jr);
					
						if (forone.isEmpty()) {
							System.out.println("this aggregate returns empty");
							break outer;
						} else {
							updateCube.addAll(new ArrayList<>(forone.get(jr))); 
							System.out.println("Created bounding box " + updateCube.size());
							for (DoubleBox<ThreeD> up : updateCube) {
								System.out.println("Created bounding box ["+up.minX.getFirst().x+" "+up.minX.getFirst().y+" "+
										up.minX.getFirst().z+"] ["+
										up.minX.getSecond().x+" "+up.minX.getSecond().y+" "+up.minX.getSecond().z+"] ["+
										up.maxX.getFirst().x+" "+up.maxX.getFirst().y+" "+up.maxX.getFirst().z+"] ["+
										up.maxX.getSecond().x+" "+up.maxX.getSecond().y+" "+up.maxX.getSecond().z+"]");
							}
						}
					} else {
						//if updateBox only has one box
						if (updateCube.size() > 1) {
							for (int u=0 ; u < updateCube.size()-1 ; u++) {
								DoubleBox<ThreeD> subplane = updateCube.get(u);
								for (int w=u+1; w < updateCube.size(); w++) {
									DoubleBox<ThreeD> dubplane = updateCube.get(w);
									if (dubplane.maxX.getFirst().equals(subplane.maxX.getFirst()) && 
											dubplane.maxX.getSecond().equals(subplane.maxX.getSecond()) ) {
										updateCube.remove(dubplane);
									}
								}
							}
						}
						List<DoubleBox<ThreeD>> repeatupdate = new ArrayList<>();
						for (int r=0; r<updateCube.size(); r++) {
							DoubleBox<ThreeD> subplane = updateCube.get(r);
							List<DoubleBox<ThreeD>> crossmulti = 
									crossvalidation(subplane, collect.get(fungsi.getSecond()), fx, fy, fz, 
											baseClass.get(fungsi.getSecond()), 
											fungsi, ridValue.get(jr), jr);
							repeatupdate.addAll(crossmulti);
						}
						if (repeatupdate.isEmpty()) {
							System.out.println("cross aggregate no valid");
							updateCube.clear();
							break outer;
						} else {
							updateCube.clear();
							updateCube = repeatupdate;
							System.out.println("Updated bounding box " + updateCube.size());
							for (DoubleBox<ThreeD> up : updateCube) {
								System.out.println("Updated bounding box ["+up.minX.getFirst().x+" "+up.minX.getFirst().y+" "
										+up.minX.getFirst().z+"] ["+
										up.minX.getSecond().x+" "+up.minX.getSecond().y+" "+up.minX.getSecond().z+"] ["+
										up.maxX.getFirst().x+" "+up.maxX.getFirst().y+" "+up.maxX.getFirst().z+"] ["+
										up.maxX.getSecond().x+" "+up.maxX.getSecond().y+" "+up.maxX.getSecond().z+"]");
							}
						}
					}
				}
			}
			if (!updateCube.isEmpty()) {
				for (DoubleBox<ThreeD> cross : updateCube) {
					System.out.println("Stored ["+cross.minX.getFirst().x+" "+cross.minX.getFirst().y+" "+
							cross.minX.getFirst().z+"] ["+
							cross.minX.getSecond().x+" "+cross.minX.getSecond().y+" "+cross.minX.getSecond().z+"] ["+
							cross.maxX.getFirst().x+" "+cross.maxX.getFirst().y+" "+cross.maxX.getFirst().z+"] ["+
							cross.maxX.getSecond().x+" "+cross.maxX.getSecond().y+" "+cross.maxX.getSecond().z+"]");
				
					Pair<ArrayList<Integer>, DoubleBox<ThreeD>> whichTiga = new Pair<>(anyCube, cross);
					whereTiga.put(whichTiga, functioname);
				}
			} else {
				System.out.println("not any box returns");
			}
		}
	}	
	return whereTiga;	
	}
	
	public static <T extends Comparable<T>> Multimap<Integer, DoubleBox<ThreeD>> cubeConstruction (
			ListMultimap<Integer, Integer> fx, ListMultimap<Integer, Integer> fy, ListMultimap<Integer, Integer> fz, 
			Map<Integer, T> collect, List<T> firstAxis, List<T> secondAxis, List<T> thirdAxis, 
			Pair<Object, Integer> fungsi, T ridValue, 
			String baseClass, Integer bid, Integer jr) {
		Multimap<Integer, DoubleBox<ThreeD>> totalmulti = ArrayListMultimap.create();
		if (fungsi.getFirst() == "sum") {
			if (baseClass.contains("BigDecimal")) {
				BigDecimal decimalResult = new BigDecimal(ridValue.toString());
				Set<Integer> xaxiset = new HashSet<>();
				Set<Integer> yaxiset = new HashSet<>();
				Set<Integer> zaxiset = new HashSet<>();
				long produceTime = System.currentTimeMillis();
				Map<ThreeD, BigDecimal> cubeDecimal = produceCubeDecimal(fx, fy, fz, collect, xaxiset, yaxiset, zaxiset); 
				long produceEnd = System.currentTimeMillis();
				double diff = (produceEnd - produceTime)/1e0;
				System.out.println("-------------------------------------cube building-------------------------" + diff + " ms");
				Multimap<Integer, DoubleBox<ThreeD>> multimm = 
					ThreeDimenDecimal.sumshrinking(cubeDecimal, decimalResult, fungsi.getFirst(), jr, bid, 
							0, firstAxis.size()-1, 0, secondAxis.size()-1, 0, thirdAxis.size()-1, xaxiset, yaxiset, zaxiset); 
				totalmulti.putAll(multimm);
			}
		}
		else if (fungsi.getFirst() == "count") {
			//to be continued ................................................................................................
			int intResult = new Integer(ridValue.toString());
			Set<Integer> xaxiset = new HashSet<>();
			Set<Integer> yaxiset = new HashSet<>();
			Set<Integer> zaxiset = new HashSet<>();
			Map<ThreeD, Integer> cubeInteger = produceCube(fx, fy,fz, collect, xaxiset, yaxiset, zaxiset);
			Multimap<Integer, DoubleBox<ThreeD>> multimm = 
				ThreeDimenCount.countshrinking(cubeInteger, intResult, fungsi.getFirst(), jr, bid, 0, firstAxis.size()-1, 0, 
						secondAxis.size()-1, 0, thirdAxis.size()-1, xaxiset, yaxiset, zaxiset);
			totalmulti.putAll(multimm);
		}
		return totalmulti;
	}

	private static <T extends Comparable<T>> Map<ThreeD, Integer> produceCube (
			ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy, ListMultimap<Integer, Integer> fz, 
			Map<Integer, T> collect, Set<Integer> xaxiset, 
			Set<Integer> yaxiset, Set<Integer> zaxiset) {
		Map<ThreeD, Integer> matrixmap = new HashMap<>();
		for (int x=0; x<fx.keySet().size(); x++) {
			for (int y=0; y<fy.keySet().size(); y++) {
				for (int z=0; z<fz.keySet().size(); z++) {
					List<Integer> ax = fx.get(x);
					List<Integer> by = fy.get(y);
					List<Integer> cz = fz.get(z);
					if (!ax.contains(null) && !by.contains(null) && !cz.contains(null)) {
						List<Integer> comm = intersectArrays(ax,by,cz);
						if (!comm.isEmpty()) {
							//int temp = 0;
							//for (int c : comm) {
							//	temp += new Integer(collect.get(c).toString());
							//}
							ThreeD tiga = new ThreeD(x,y,z);
							xaxiset.add(x); yaxiset.add(y); zaxiset.add(z);
							matrixmap.put(tiga, comm.size());
						}
					}
				}
			}
		}
		return matrixmap;
	}
	
	private static <T extends Comparable<T>> Map<ThreeD, BigDecimal> produceCubeDecimal (
			ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy, ListMultimap<Integer, Integer> fz, 
			Map<Integer, T> collect, Set<Integer> xaxiset, 
			Set<Integer> yaxiset, Set<Integer> zaxiset) {
		Map<ThreeD, BigDecimal> matrixmap = new HashMap<>();
		//List<Integer> sortX = new ArrayList<>(fx.keySet());
		//List<Integer> sortY = new ArrayList<>(fy.keySet());
		//List<Integer> sortZ = new ArrayList<>(fz.keySet());
		//Collections.sort(sortX);
		//Collections.sort(sortY);
		//Collections.sort(sortZ);
		for (int x=0; x<fx.keySet().size(); x++) {
			for (int y=0; y<fy.keySet().size(); y++) {
				for (int z=0; z<fz.keySet().size(); z++) {
					List<Integer> ax = fx.get(x);
					List<Integer> by = fy.get(y);
					List<Integer> cz = fz.get(z);
					if (!ax.contains(null) && !by.contains(null) && !cz.contains(null)) {
						List<Integer> comm = intersectArrays(ax,by,cz);
						if (!comm.isEmpty()) {
							BigDecimal temp = new BigDecimal(0);
							for (int c : comm) {
								temp = temp.add(new BigDecimal(collect.get(c).toString()));
							}
							ThreeD tiga = new ThreeD(x,y,z);
							xaxiset.add(x); yaxiset.add(y); zaxiset.add(z);
							matrixmap.put(tiga, temp);
						}
					}
				}
			}
		}
		return matrixmap;
	}

	private static <T> List<T> intersectArrays(List<T> ax,
			List<T> by, List<T> cz) {
		Map<T, Long> intersection = new HashMap<>();
		List<T> returnList = new LinkedList<T>();
		for (T elem : ax) {
			Long count = intersection.get(elem);
			if (count != null) {
				intersection.put(elem, count+1);
			} else {
				intersection.put(elem, 1L);
			}
		}
		for (T elem : by) {
			Long count = intersection.get(elem);
			if (count != null ) {
				intersection.put(elem, count+1);
			} else {
				intersection.put(elem, 1L);
			}
		}
		for (T elem : cz) {
			Long count = intersection.get(elem);
			if (count != null) {
				intersection.put(elem,  count+1);
			} else {
				intersection.put(elem, 1L);
			}
		}
		for (T elem : intersection.keySet()) {
			Long count = intersection.get(elem);
			if (count >= 3) {
				returnList.add(elem);
			}
		}
		return returnList;
	}
	
	private static <T extends Comparable<T>> List<DoubleBox<ThreeD>> crossvalidation (
			DoubleBox<ThreeD> subplane, Map<Integer, T> collect, ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy, ListMultimap<Integer, Integer> fz, String baseClass, 
			Pair<Object, Integer> fungsi, T ridvalue, int rid) {
		List<DoubleBox<ThreeD>> crossmulti = new ArrayList<>();
		Pair<ThreeD, ThreeD> maxplane = subplane.maxX;
		int xa = maxplane.getFirst().x;
		int ya = maxplane.getFirst().y;
		int za = maxplane.getFirst().z;
		int xb = maxplane.getSecond().x;
		int yb = maxplane.getSecond().y;
		int zb = maxplane.getSecond().z;
		if (fungsi.getFirst() == "count") {
			Map<ThreeD, Integer> matrixint = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			List<Integer> ybx = new ArrayList<>();
			List<Integer> zbx = new ArrayList<>();
			int total = 0;
			for (int x=xa; x<=xb; x++) {
				for (int y=ya; y<=yb; y++) {
					for (int z=za; z<=zb; z++) {
						List<Integer> ax = fx.get(x);
						List<Integer> by = fy.get(y);
						List<Integer> cz = fz.get(z);
						if (!ax.contains(null) && !by.contains(null) && !cz.contains(null)) {
							List<Integer> comm = intersectArrays(ax, by, cz);
							if (!comm.isEmpty()) {
									//int temp = 0;
									//for (int c : comm) {
									//	temp += new Integer(collect.get(c).toString());
									//}
									ThreeD tiga = new ThreeD(x,y,z);
									matrixint.put(tiga, comm.size());
									xbx.add(x);
									ybx.add(y);
									zbx.add(z);
									total += comm.size();
							}
						}
					}
				}
			}
			int compareBig = new Integer(ridvalue.toString());
			if (total == compareBig) {
				if (Collections.min(xbx) <= subplane.minX.getSecond().x && subplane.minX.getFirst().x <= Collections.max(xbx) &&
						Collections.min(ybx) <= subplane.minX.getSecond().y && subplane.minX.getFirst().y <= Collections.max(ybx) &&
						Collections.min(zbx) <= subplane.minX.getSecond().z && subplane.minX.getFirst().z <= Collections.max(zbx)) {
					ThreeD xch = new ThreeD(Collections.min(xbx),Collections.min(ybx),Collections.min(zbx));
					ThreeD ych = new ThreeD(Collections.max(xbx),Collections.max(ybx),Collections.max(zbx));
					Pair<ThreeD, ThreeD> minbxch = new Pair<>(xch, ych);
					DoubleBox<ThreeD> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane);
				}
			} else if (total > compareBig){
				Set<Integer> xaxiset = new HashSet<>(xbx);
				Set<Integer> yaxiset = new HashSet<>(ybx);
				Set<Integer> zaxiset = new HashSet<>(zbx);
				Pair<ThreeD, ThreeD> minplane = subplane.minX;
				xaxiset.add(minplane.getFirst().x); xaxiset.add(minplane.getSecond().x);
				yaxiset.add(minplane.getFirst().y); yaxiset.add(minplane.getSecond().y);
				zaxiset.add(minplane.getFirst().z); zaxiset.add(minplane.getSecond().z);
				Multimap<Integer, DoubleBox<ThreeD>> multimm = 
						ThreeDimenCount.countshrinking(matrixint, compareBig, fungsi.getFirst(), rid, fungsi.getSecond(), 
								xa, xb, ya, yb, za, zb, xaxiset, yaxiset, zaxiset);
				for (int t : multimm.keySet()) {
					List<DoubleBox<ThreeD>> newbox = (List<DoubleBox<ThreeD>>) multimm.get(t);
					for (DoubleBox<ThreeD> nb: newbox) {
						Pair<ThreeD, ThreeD> nubox = nb.minX;
						Pair<ThreeD, ThreeD> unbox = nb.maxX;
						if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
						if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
						if (minplane.getFirst().z < nubox.getFirst().z) nubox.getFirst().z = minplane.getFirst().z;
						if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
						if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
						if (minplane.getSecond().z > nubox.getSecond().z) nubox.getSecond().z = minplane.getSecond().z;
						if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
								unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y &&
								unbox.getFirst().z <= nubox.getFirst().x && unbox.getSecond().z >= nubox.getSecond().z) {
							DoubleBox<ThreeD> crossbox = new DoubleBox<>(nubox, unbox);
							
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		else if (fungsi.getFirst() == "sum" && baseClass.contains("BigDecimal")) {
			Map<ThreeD, BigDecimal> matrixdecimal = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			List<Integer> ybx = new ArrayList<>();
			List<Integer> zbx = new ArrayList<>();
			BigDecimal total = new BigDecimal(0);
			for (int x=xa; x<=xb; x++) {
				for (int y=ya; y<=yb; y++) {
					for (int z=za; z<=zb; z++) {
						List<Integer> ax = fx.get(x);
						List<Integer> by = fy.get(y);
						List<Integer> cz = fz.get(z);
						if (!ax.contains(null) && !by.contains(null) && !cz.contains(null)) {
							List<Integer> comm = intersectArrays(ax, by, cz);
							if (!comm.isEmpty()) {
								if (fungsi.getFirst() == "sum" && baseClass.contains("BigDecimal")) {
									BigDecimal temp = new BigDecimal(0);
									for (int c : comm) {
										temp = temp.add(new BigDecimal(collect.get(c).toString()));
									}
									ThreeD tiga = new ThreeD(x,y,z);
									matrixdecimal.put(tiga, temp);
									xbx.add(x);
									ybx.add(y);
									zbx.add(z);
									total = total.add(temp);
								}
							}
						}
					}
				}
			}
			BigDecimal compareBig = new BigDecimal(ridvalue.toString());
			if (total.equals(compareBig)) {
				if (Collections.min(xbx) <= subplane.minX.getSecond().x && subplane.minX.getFirst().x <= Collections.max(xbx) &&
						Collections.min(ybx) <= subplane.minX.getSecond().y && subplane.minX.getFirst().y <= Collections.max(ybx) &&
						Collections.min(zbx) <= subplane.minX.getSecond().z && subplane.minX.getFirst().z <= Collections.max(zbx)) {
					ThreeD xch = new ThreeD(Collections.min(xbx),Collections.min(ybx),Collections.min(zbx));
					ThreeD ych = new ThreeD(Collections.max(xbx),Collections.max(ybx),Collections.max(zbx));
					Pair<ThreeD, ThreeD> minbxch = new Pair<>(xch, ych);
					DoubleBox<ThreeD> boundregion = new DoubleBox<>(minbxch, subplane.maxX);
					crossmulti.add(boundregion);
				} else {
					crossmulti.add(subplane);
				}
			} else if (total.compareTo(compareBig) > 0){
				Set<Integer> xaxiset = new HashSet<>(xbx);
				Set<Integer> yaxiset = new HashSet<>(ybx);
				Set<Integer> zaxiset = new HashSet<>(zbx);
				Pair<ThreeD, ThreeD> minplane = subplane.minX;
				xaxiset.add(minplane.getFirst().x); xaxiset.add(minplane.getSecond().x);
				yaxiset.add(minplane.getFirst().y); yaxiset.add(minplane.getSecond().y);
				zaxiset.add(minplane.getFirst().z); zaxiset.add(minplane.getSecond().z);
				Multimap<Integer, DoubleBox<ThreeD>> multimm = 
						ThreeDimenDecimal.sumshrinking(matrixdecimal, compareBig, fungsi.getFirst(), rid, fungsi.getSecond(), 
								xa, xb, ya, yb, za, zb, xaxiset, yaxiset, zaxiset);
				for (int t : multimm.keySet()) {
					List<DoubleBox<ThreeD>> newbox = (List<DoubleBox<ThreeD>>) multimm.get(t);
					for (DoubleBox<ThreeD> nb: newbox) {
						Pair<ThreeD, ThreeD> nubox = nb.minX;
						Pair<ThreeD, ThreeD> unbox = nb.maxX;
						if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
						if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
						if (minplane.getFirst().z < nubox.getFirst().z) nubox.getFirst().z = minplane.getFirst().z;
						if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
						if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
						if (minplane.getSecond().z > nubox.getSecond().z) nubox.getSecond().z = minplane.getSecond().z;
						if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
								unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y &&
								unbox.getFirst().z <= nubox.getFirst().x && unbox.getSecond().z >= nubox.getSecond().z) {
							DoubleBox<ThreeD> crossbox = new DoubleBox<>(nubox, unbox);
							
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		return crossmulti;	
	}
}
