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

public class TwoDimension {
	public static <T extends Comparable<T>> 
	Map<Pair<ArrayList<Integer>, DoubleBox<TwoD>>, List<Pair<Object, Integer>>> regionCut(
			List<Integer> nongroupdimensi, 
			Multimap<List<String>, DimenIndex<T>> ridTupleIndex,
			Map<Integer, String> baseClass, 
			List<Map<Integer, Pair<Object, Integer>>> whichaggre, List<Integer> ridorder)
	{
		
		Map<Pair<ArrayList<Integer>, DoubleBox<TwoD>>, List<Pair<Object, Integer>>> whereDuaBiru = 
				new HashMap<>();
		
		Integer[] pairDimension = nongroupdimensi.toArray(new Integer[nongroupdimensi.size()]);
		ArrayList<ArrayList<Integer>> dimensionPair = PreprocessBase.combination(pairDimension, 2);
		
		for(ArrayList<Integer> anyPair: dimensionPair) {
			List<T> firstAxis = new ArrayList<>();
			List<T> secondAxis = new ArrayList<>();
			
			//outer:
			for (Map<Integer, Pair<Object, Integer>> aggpernode : whichaggre) {
				List<Pair<Object, Integer>> functioname = new ArrayList<>(aggpernode.values());
				//repository for run filter function
				List<DoubleBox<TwoD>> updateBox = new ArrayList<>(); //for cross validation all
				outer:
				for (List<String> ridTuple : ridTupleIndex.keySet()) {
					
					Collection<DimenIndex<T>> valueIndex = ridTupleIndex.get(ridTuple);
					ListMultimap<Integer, Integer> fx = ArrayListMultimap.create();
					ListMultimap<Integer, Integer> fy = ArrayListMultimap.create();
					Map<Integer, Map<Integer, T>> collect = new HashMap<>();
					Map<Integer, T> ridValue = new HashMap<>();
					for (DimenIndex<T> vi : valueIndex) {
						collect.put(vi.index, vi.indexMatch);
						if (vi.index == anyPair.get(0)) {
							ridValue = vi.ridValue;
							firstAxis = vi.letT;
							fx = vi.newBuildD;
						} else if (vi.index == anyPair.get(1)) {
							secondAxis = vi.letT;
							fy = vi.newBuildD;
						}
					}
					for (int jr: ridorder) {
						Pair<Object, Integer> fungsi = aggpernode.get(jr);
						System.out.println(anyPair + " " + functioname + " " + ridTuple + " " 
								+ fungsi + " " + ridValue.get(jr) + " " +
								fx.keySet().size() + " " + fy.keySet().size()); // + " " + updateBox.size());
						
						if (updateBox.isEmpty()) { 
							Multimap<Integer, DoubleBox<TwoD>> forone = 
								planeConstruction(fx, fy, collect.get(fungsi.getSecond()), firstAxis, secondAxis, 
								fungsi, ridValue.get(jr), baseClass.get(fungsi.getSecond()), fungsi.getSecond(), jr);
						
							if (forone.isEmpty()) {
								System.out.println("this aggregate returns empty");
								break outer;
							} else {
								updateBox.addAll(new ArrayList<>(forone.get(jr))); 
								System.out.println("Created bounding box " + updateBox.size());
								for (DoubleBox<TwoD> up : updateBox) {
									System.out.println("Created bounding box ["+up.minX.getFirst().x+" "+
											up.minX.getFirst().y+"] ["+
											up.minX.getSecond().x+" "+up.minX.getSecond().y+"] ["+
											up.maxX.getFirst().x+" "+up.maxX.getFirst().y+"] ["+
											up.maxX.getSecond().x+" "+up.maxX.getSecond().y+"]");
								}
							}
						} else {
							//if updateBox only has one box
							if (updateBox.size() > 1) {
								for (int u=0 ; u < updateBox.size()-1 ; u++) {
									DoubleBox<TwoD> subplane = updateBox.get(u);
									for (int w=u+1; w < updateBox.size(); w++) {
										DoubleBox<TwoD> dubplane = updateBox.get(w);
										if (dubplane.maxX.getFirst().equals(subplane.maxX.getFirst()) && 
												dubplane.maxX.getSecond().equals(subplane.maxX.getSecond()) ) {
											updateBox.remove(dubplane);
										}
									}
								}
							}
							List<DoubleBox<TwoD>> repeatupdate = new ArrayList<>();
							for (int r=0; r<updateBox.size(); r++) {
								DoubleBox<TwoD> subplane = updateBox.get(r);
								List<DoubleBox<TwoD>> crossmulti = 
										crossvalidation(subplane, collect.get(fungsi.getSecond()), 
												fx, fy, baseClass.get(fungsi.getSecond()), 
												fungsi, ridValue.get(jr), jr);
								repeatupdate.addAll(crossmulti);
							}
							if (repeatupdate.isEmpty()) {
								System.out.println("cross aggregate no valid");
								updateBox.clear();
								break outer;
							} else {
								updateBox.clear();
								updateBox = repeatupdate;
								System.out.println("Updated bounding box " + updateBox.size());
								for (DoubleBox<TwoD> up : updateBox) {
									System.out.println("Updated bounding box ["+up.minX.getFirst().x+" "+
											up.minX.getFirst().y+"] ["+
											up.minX.getSecond().x+" "+up.minX.getSecond().y+"] ["+
											up.maxX.getFirst().x+" "+up.maxX.getFirst().y+"] ["+
											up.maxX.getSecond().x+" "+up.maxX.getSecond().y+"]");
								}
							}
						}
					}
				}
				if (!updateBox.isEmpty()) {
					for (DoubleBox<TwoD> cross : updateBox) {
						System.out.println("stored ["+cross.minX.getFirst().x+" "+
						cross.minX.getFirst().y+"] ["+
						cross.minX.getSecond().x+" "+cross.minX.getSecond().y+"] ["+
						cross.maxX.getFirst().x+" "+cross.maxX.getFirst().y+"] ["+
						cross.maxX.getSecond().x+" "+cross.maxX.getSecond().y+"]");
					
						Pair<ArrayList<Integer>, DoubleBox<TwoD>> whereSatuMerah = new Pair<>(anyPair, cross);
						whereDuaBiru.put(whereSatuMerah, functioname);
					}
				} else {
					System.out.println("not any box returns");
				}
			}
		} 	
		return whereDuaBiru;
	}
	
	public static <T extends Comparable<T>> Multimap<Integer, DoubleBox<TwoD>> planeConstruction (
			ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy, 
			Map<Integer, T> collect, List<T> firstAxis, List<T> secondAxis,
			Pair<Object, Integer> fungsi, T ridValue, String baseClass, Integer bid, Integer jr) {
		Multimap<Integer, DoubleBox<TwoD>> totalmulti = ArrayListMultimap.create();
		if (fungsi.getFirst() == "max" || fungsi.getFirst() == "min") {
			//T[][] matrix = (T[][]) Array.newInstance(ridValue.getClass(), fx.size(), fy.size());
			Map<TwoD, T> cellxy =  new HashMap<>();             			// special class (x,y)
			List<TwoD> expandPoint = new ArrayList<>();
			Set<Integer> xaxiset = new HashSet<>();
			Set<Integer> yaxiset = new HashSet<>();
			for (int x=0; x<fx.keySet().size(); x++) {
				for (int y=0; y<fy.keySet().size(); y++) {
					TwoD xy = new TwoD(x,y);
					List<Integer> f = fx.get(x);
					List<Integer> g = fy.get(y);
					if (!f.contains(null) && !g.contains(null)) {
						List<Integer> c = intersectArrays(f,g);            					//@@@@
						//List<Integer> c = new ArrayList<>(f);
						//c.retainAll(g);
						if (!c.isEmpty()) {
							List<T> fc = new ArrayList<>();
							for (int cc : c) {
								fc.add(collect.get(cc));
							}
							xaxiset.add(x); yaxiset.add(y);
							if (fungsi.getFirst() == "max") {
								T maxf = Collections.max(fc);
								cellxy.put(xy, maxf);
								//matrix[x][y] = Collections.max(fc); 
								if (maxf.equals(ridValue)) {
									TwoD point = new TwoD(x,y);
									expandPoint.add(point);
								} 
							} else if (fungsi.getFirst() == "min") {
								T minf = Collections.min(fc);
								cellxy.put(xy, minf);
								//matrix[x][y] = Collections.min(fc); 
								if (minf.equals(ridValue)) {
									TwoD point = new TwoD(x,y);
									expandPoint.add(point);
								} 
							}
						} 
					} 
				}
			}
			//Multimap<Integer, DoubleBox<TwoD>> multimm = 
			//	trynewexpansion(cellxy, expandPoint, ridValue, fungsi.getFirst(), jr, bid, 
			//			0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset); 
			/*Multimap<Integer, DoubleBox<TwoD>> wantdie = 
				Pilihan.expansion(cellxy, expandPoint, ridValue, fungsi.getFirst(), jr, bid, 0, 
				firstAxis.size()-1, 0 , secondAxis.size()-1); */
			/**Multimap<Integer, DoubleBox<TwoD>> multimm = 
				Optional.shrinking(cellxy, expandPoint, ridValue, fungsi.getFirst(), jr, bid, 0, 
				firstAxis.size()-1, 0, secondAxis.size()-1, 
						xaxiset, yaxiset); **/
			Multimap<Integer, DoubleBox<TwoD>> multimm = 
					TwoDimen.expansionMinMax(cellxy, expandPoint, ridValue, fungsi.getFirst(), jr, bid, 
							0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset);
			totalmulti.putAll(multimm);
		} 
		else if (fungsi.getFirst() == "sum") {
			if (baseClass.contains("Integer")) {
				Integer intResult = new Integer(ridValue.toString());
				Set<Integer> xaxiset = new HashSet<>();
				Set<Integer> yaxiset = new HashSet<>();
				Map<TwoD, Integer> planeDigit = 
						TwoDimenDigit.producePlaneInteger(fx, fy, collect, xaxiset, yaxiset, fungsi.getFirst()); 
				Multimap<Integer, DoubleBox<TwoD>> multimm = 
						TwoDimenDigit.sumshrinking(planeDigit, intResult, fungsi.getFirst(), 
						jr, bid, 0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset);
				totalmulti.putAll(multimm); 
			} else if (baseClass.contains("BigDecimal")) {
				/**BigDecimal[][] planeDecimal = producePlaneDecimal(fx, fy, collect); */
				//Map<TwoD, BigDecimal> planeDecimal = producePlaneDecimal(fx, fy, collect);
				BigDecimal decimalResult = new BigDecimal(ridValue.toString());
				Set<Integer> xaxiset = new HashSet<>();
				Set<Integer> yaxiset = new HashSet<>();
				Map<TwoD, BigDecimal> planeDecimal = TwoDimenDecimal.producePlaneDecimal(fx, fy, collect, xaxiset, yaxiset); 
				Multimap<Integer, DoubleBox<TwoD>> multimm = 
						TwoDimenDecimal.sumshrinking(planeDecimal, decimalResult, fungsi.getFirst(), jr, bid, 
								0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset); 
				totalmulti.putAll(multimm);
				/**Multimap<Integer, DoubleBox<TwoD>> multimm = Pilihan.sumshrinkingdecimal(planeDecimal, decimalResult, 
				 	fungsi.getFirst(), jr, bid, 0, firstAxis.size()-1, 0, secondAxis.size()-1); **/
				/*******
				List<TwoD> pickpoint = new ArrayList<>();
				for (TwoD sel : planeDecimal.keySet()) {
					BigDecimal angka = planeDecimal.get(sel);
					if (angka.compareTo(decimalResult) < 0) {
						pickpoint.add(sel);
					}
				}
				Multimap<Integer, DoubleBox<TwoD>> multimm =
						sumexpansion(planeDecimal, pickpoint, decimalResult, fungsi.getFirst(), jr, bid, 
								0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset);
				*******/
			}
		} 
		else if (fungsi.getFirst() == "count") {
			int countvalue = new Integer(ridValue.toString());
			Set<Integer> xaxiset = new HashSet<>();
			Set<Integer> yaxiset = new HashSet<>();
			Map<TwoD, Integer> planeInteger = 
					TwoDimenDigit.producePlaneInteger(fx, fy, collect, xaxiset, yaxiset, fungsi.getFirst());
			Multimap<Integer, DoubleBox<TwoD>> multimm = 
				TwoDimenCount.countshrinking(planeInteger, countvalue, jr, bid, 
						0, firstAxis.size()-1, 0, secondAxis.size()-1, xaxiset, yaxiset);
			totalmulti.putAll(multimm);
		}
		return totalmulti;
	}
	
	private static <T extends Comparable<T>> List<DoubleBox<TwoD>> crossvalidation(DoubleBox<TwoD> plane, 
			Map<Integer, T> collect, ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy, String baseClass, 
			Pair<Object, Integer> fungsi, T ridvalue, int rid)  {
		List<DoubleBox<TwoD>> crossmulti = new ArrayList<>();
		
		Pair<TwoD, TwoD> maxplane = plane.maxX;
		int xa = maxplane.getFirst().x; int ya = maxplane.getFirst().y;
		int xb = maxplane.getSecond().x; int yb = maxplane.getSecond().y;
		if (fungsi.getFirst() == "count") {
			Map<TwoD, Integer> matrixint = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			List<Integer> ybx = new ArrayList<>();
			int total = 0;
			for (int x= xa; x <= xb; x++) {
				for (int y= ya; y <= yb; y++) {
					List<Integer> f = fx.get(x);
					List<Integer> g = fy.get(y);
					if (!f.contains(null) && !g.contains(null)) {
						List<Integer> c = intersectArrays(f,g);
						if (!c.isEmpty()) {
							//int temp = 0;
							//for (int cc : c) {
							//	temp += new Integer(collect.get(cc).toString());
							//}
							TwoD dua = new TwoD(x,y);
							matrixint.put(dua, c.size());
							xbx.add(x); 
							ybx.add(y);
							total += c.size();
						}
					}
				}
			}
			int countvalue = new Integer(ridvalue.toString());
			if (total == countvalue) {
				if (Collections.min(xbx) <= plane.minX.getSecond().x && plane.minX.getFirst().x <= Collections.max(xbx) &&
						Collections.min(ybx) <= plane.minX.getSecond().y && plane.minX.getFirst().y <= Collections.max(ybx)) {
					TwoD xch = new TwoD(Collections.min(xbx),Collections.min(ybx));
					TwoD ych = new TwoD(Collections.max(xbx),Collections.max(ybx));
					Pair<TwoD, TwoD> minbxch = new Pair<>(xch, ych);
					DoubleBox<TwoD> boundregion = new DoubleBox<>(minbxch, plane.maxX);
					crossmulti.add(boundregion);
					/**add into list*/
				} else {
					crossmulti.add(plane);
				}
			}
			else if (total > countvalue) {
				Set<Integer> xaxiset = new HashSet<>(xbx);
				Set<Integer> yaxiset = new HashSet<>(ybx);
				Pair<TwoD, TwoD> minplane = plane.minX;
				xaxiset.add(minplane.getFirst().x); xaxiset.add(minplane.getSecond().x);
				yaxiset.add(minplane.getFirst().y); yaxiset.add(minplane.getSecond().y);
				Multimap<Integer, DoubleBox<TwoD>> multimm = 
						TwoDimenCount.countshrinking(matrixint, countvalue, rid, fungsi.getSecond(), xa, xb, ya, yb, 
						xaxiset, yaxiset);
				for (int t : multimm.keySet()) {
					List<DoubleBox<TwoD>> newbox = (List<DoubleBox<TwoD>>) multimm.get(t);
					for (DoubleBox<TwoD> nb: newbox) {
						Pair<TwoD, TwoD> nubox = nb.minX;
						Pair<TwoD, TwoD> unbox = nb.maxX;
						if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
						if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
						if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
						if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
						if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
								unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y) {
							DoubleBox<TwoD> crossbox = new DoubleBox<>(nubox, unbox);
							/**System.out.println("validate RID " + t +" ["+crossbox.minX.getFirst().x+" "
							+crossbox.minX.getFirst().y+"] ["
							+crossbox.minX.getSecond().x+" "+crossbox.minX.getSecond().y+"]"+ 
									"["+crossbox.maxX.getFirst().x+" "+crossbox.maxX.getFirst().y+"] ["+
							crossbox.maxX.getSecond().x+" "+crossbox.maxX.getSecond().y+"]"); **/
							
							crossmulti.add(crossbox);
						}
					}
				}
			}
		}
		else if (fungsi.getFirst() == "sum" && baseClass.contains("BigDecimal")) {
			Map<TwoD, BigDecimal> matrixdecimal = new HashMap<>();
			List<Integer> xbx = new ArrayList<>();
			List<Integer> ybx = new ArrayList<>();
			BigDecimal total = new BigDecimal(0);
			//List<Integer> sortX = new ArrayList<>(fx.keySet());
			//List<Integer> sortY = new ArrayList<>(fy.keySet());
			//Collections.sort(sortX);
			//Collections.sort(sortY);
			//System.out.println("x " + xa + "-" + xb + " " + sortX.size() +
			//		" y " + ya + "-" + yb + " " + sortY.size() + " " + sortY.get(sortY.size()-1));
			for (int x= xa; x <= xb; x++) {
				for (int y= ya; y <= yb; y++) {
					//System.out.println("x " + x + " y " + y);
					List<Integer> f = fx.get(x);
					List<Integer> g = fy.get(y);
					if (!f.contains(null) && !g.contains(null)) {
						List<Integer> c = intersectArrays(f,g);
						//List<Integer> c = new ArrayList<>(f);
						//c.retainAll(g);
						if (!c.isEmpty()) {
							if (fungsi.getFirst() == "sum" && baseClass.contains("BigDecimal")) {
								BigDecimal temp = new BigDecimal(0);
								for (int cc : c) {
									temp = temp.add(new BigDecimal(collect.get(cc).toString()));
								}
								TwoD dua = new TwoD(x,y);
								matrixdecimal.put(dua, temp);
								xbx.add(x); 
								ybx.add(y);
								total = total.add(temp);
	 							/**sumbigxy.put(xy, temp); */
							}
						} 
					} 
				}
			}
			BigDecimal compareBig = new BigDecimal(ridvalue.toString());
			if (total.equals(compareBig)) {
				//equal, update min bound box
				if (Collections.min(xbx) <= plane.minX.getSecond().x && plane.minX.getFirst().x <= Collections.max(xbx) &&
						Collections.min(ybx) <= plane.minX.getSecond().y && plane.minX.getFirst().y <= Collections.max(ybx)) {
					TwoD xch = new TwoD(Collections.min(xbx),Collections.min(ybx));
					TwoD ych = new TwoD(Collections.max(xbx),Collections.max(ybx));
					Pair<TwoD, TwoD> minbxch = new Pair<>(xch, ych);
					DoubleBox<TwoD> boundregion = new DoubleBox<>(minbxch, plane.maxX);
					crossmulti.add(boundregion);
					/**add into list*/
				} else {
					crossmulti.add(plane);
				}
			} else if (total.compareTo(compareBig) > 0) {
				//refine by finding new bounding box
				Set<Integer> xaxiset = new HashSet<>(xbx);
				Set<Integer> yaxiset = new HashSet<>(ybx);
				Pair<TwoD, TwoD> minplane = plane.minX;
				xaxiset.add(minplane.getFirst().x); xaxiset.add(minplane.getSecond().x);
				yaxiset.add(minplane.getFirst().y); yaxiset.add(minplane.getSecond().y);
				/**
				TwoD baru = minplane.getFirst();
				TwoD lama = minplane.getSecond();
				List<TwoD> pickpoint = new ArrayList<>();
				for (TwoD expand : matrixdecimal.keySet()) {
					if (!baru.equals(lama)) {
						if (expand.equals(baru) && matrixdecimal.get(expand).compareTo(compareBig) <= 0) {
							pickpoint.add(expand);
						} else if (expand.equals(baru) && matrixdecimal.get(expand).compareTo(compareBig) <= 0) {
							pickpoint.add(lama);
						}
					} 
				}
				Multimap<Integer, DoubleBox<TwoD>> multimm =
					sumexpansion(matrixdecimal, minplane, compareBig, fungsi.getFirst(), rid, fungsi.getSecond(), xa, xb, ya, yb, 
							xaxiset, yaxiset); 
				*/
				Multimap<Integer, DoubleBox<TwoD>> multimm = 
					TwoDimenDecimal.sumshrinking(matrixdecimal, compareBig, fungsi.getFirst(), rid, fungsi.getSecond(), 
							xa, xb, ya, yb, xaxiset, yaxiset);
				
				for (int t : multimm.keySet()) {
					List<DoubleBox<TwoD>> newbox = (List<DoubleBox<TwoD>>) multimm.get(t);
					for (DoubleBox<TwoD> nb: newbox) {
						Pair<TwoD, TwoD> nubox = nb.minX;
						Pair<TwoD, TwoD> unbox = nb.maxX;
						if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
						if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
						if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
						if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
						if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
								unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y) {
							DoubleBox<TwoD> crossbox = new DoubleBox<>(nubox, unbox);
							/**System.out.println("validate RID " + t +" ["+crossbox.minX.getFirst().x
							+" "+crossbox.minX.getFirst().y+"] ["
							+crossbox.minX.getSecond().x+" "+crossbox.minX.getSecond().y+"]"+ 
									"["+crossbox.maxX.getFirst().x+" "+crossbox.maxX.getFirst().y+"] ["+
							crossbox.maxX.getSecond().x+" "+crossbox.maxX.getSecond().y+"]"); **/
							
							crossmulti.add(crossbox);
						}
					}
				}
			}
		} else if (fungsi.getFirst() == "max" || fungsi.getFirst() == "min") {
			//T[][] matrixvalid = (T[][]) Array.newInstance(ridvalue.getClass(), xlen, ylen);
			Map<TwoD, T> matrixvalid = new HashMap<>();
			List<TwoD> point = new ArrayList<>();
			List<T> gatherdata = new ArrayList<>();
			List<T> boundata = new ArrayList<>();
			Set<Integer> xaxiset = new HashSet<>();
			Set<Integer> yaxiset = new HashSet<>();
			Pair<TwoD, TwoD> minplane = plane.minX;
			xaxiset.add(minplane.getFirst().x); xaxiset.add(minplane.getSecond().x);
			yaxiset.add(minplane.getFirst().y); yaxiset.add(minplane.getSecond().y);
			
			for (int x= xa; x <= xb; x++) {
				for (int y= ya; y <= yb; y++) {
					
					List<Integer> f = fx.get(x);
					List<Integer> g = fy.get(y);
					if (!f.contains(null) && !g.contains(null)) {
						List<Integer> c = intersectArrays(f,g);
						if (!c.isEmpty()) {
							xaxiset.add(x); yaxiset.add(y);
							TwoD xy = new TwoD(x,y);
							List<T> fc = new ArrayList<>();
							for (int cc: c) {
								fc.add(collect.get(cc));
							}
							if (fungsi.getFirst() == "max") {
								/**cellxy.put(xy, Collections.max(fc)); */
								T somethg = Collections.max(fc);
								gatherdata.add(somethg);
								if (x >= minplane.getFirst().x && x <= minplane.getSecond().x &&
										y >= minplane.getFirst().y && y <= minplane.getSecond().y) {
									boundata.add(somethg);
								}
								matrixvalid.put(xy, somethg);
								if (somethg.equals(ridvalue)) {
									point.add(xy);
								}
							} else if (fungsi.getFirst() == "min") {
								/**cellxy.put(xy,  Collections.min(fc)); */
								T somethg = Collections.min(fc);
								gatherdata.add(somethg);
								if (x >= minplane.getFirst().x && x <= minplane.getSecond().x &&
										y >= minplane.getFirst().y && y <= minplane.getSecond().y) {
									boundata.add(somethg);
								}
								matrixvalid.put(xy, somethg);
								if (somethg.equals(ridvalue)) {
									point.add(xy);
								}
							} 
						}
					}
				}
			}
			if (fungsi.getFirst() == "max") {
				T somethg = Collections.max(gatherdata);
				if (somethg.equals(ridvalue)) {
					//equal, find new min bound box
					List<Integer> xs = new ArrayList<>();
					List<Integer> ys = new ArrayList<>();
					xs.add(plane.minX.getFirst().x);
					xs.add(plane.minX.getSecond().x);
					ys.add(plane.minX.getFirst().y);
					ys.add(plane.minX.getSecond().y);
					for (TwoD cell : point) {
						xs.add(cell.x);
						ys.add(cell.y);
							//plane.minX = new Pair<>(cell, cell);
							//Pair<Object, Integer> crossfx = new Pair<>(daftar.getFirst(), bid);
							//Pair<Pair<Object, Integer>, DoubleBox<TwoD>> crossplane = new Pair<>(crossfx, plane);
					}
					TwoD left = new TwoD(Collections.min(xs), Collections.min(ys));
					TwoD right = new TwoD(Collections.max(xs), Collections.max(ys));
					Pair<TwoD, TwoD> leftRight = new Pair<>(left, right);
					DoubleBox<TwoD> boundregion = new DoubleBox<>(leftRight, plane.maxX);
					crossmulti.add(boundregion);
					/**add into list*/
				} else if (gatherdata.contains(ridvalue) && Collections.max(boundata).equals(ridvalue) 
						&& somethg.compareTo(ridvalue) > 0) {
					
					Multimap<Integer, DoubleBox<TwoD>> multimm = 
						TwoDimen.expansionMinMax(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), 
								xa, xb, ya, yb, xaxiset, yaxiset);
					/**Multimap<Integer, DoubleBox<TwoD>> multimm = 
						Pilihan.expansion(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), xa, xb, ya, yb);
					Multimap<Integer, DoubleBox<TwoD>> multimm = 
						Optional.shrinking(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), 
						xa, xb, ya, yb, xaxiset, yaxiset); */
					
					for (int t : multimm.keySet()) {
						List<DoubleBox<TwoD>> newbox = (List<DoubleBox<TwoD>>) multimm.get(t);
						for (DoubleBox<TwoD> nb: newbox) {
							Pair<TwoD, TwoD> nubox = nb.minX;
							Pair<TwoD, TwoD> unbox = nb.maxX;
							if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
							if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
							if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
							if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
							if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
									unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y) {
								DoubleBox<TwoD> crossbox = new DoubleBox<>(nubox, unbox);
								crossmulti.add(crossbox);
								/**add into list*/
							}
						}
					}
				}
			} else if (fungsi.getFirst() == "min") {
				T somethg = Collections.min(gatherdata);
				if (somethg.equals(ridvalue)) {
					List<Integer> xs = new ArrayList<>();
					List<Integer> ys = new ArrayList<>();
					xs.add(plane.minX.getFirst().x);
					xs.add(plane.minX.getSecond().x);
					ys.add(plane.minX.getFirst().y);
					ys.add(plane.minX.getSecond().y);
					for (TwoD cell : point) {
						xs.add(cell.x);
						ys.add(cell.y);
					}
					TwoD left = new TwoD(Collections.min(xs), Collections.min(ys));
					TwoD right = new TwoD(Collections.max(xs), Collections.max(ys));
					Pair<TwoD, TwoD> leftRight = new Pair<>(left, right);
					DoubleBox<TwoD> boundregion = new DoubleBox<>(leftRight, plane.maxX);
					//plane.minX = new Pair<>(left, right);
					crossmulti.add(boundregion);
					/**add into list*/
				} else if (gatherdata.contains(ridvalue) && Collections.min(boundata).equals(ridvalue) 
						&& somethg.compareTo(ridvalue) < 0) {
					//refine by finding new bounding box
					/*Multimap<Integer, DoubleBox<TwoD>> multimm = 
							Optional.shrinking(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), 
									xa, xb, ya, yb, xaxiset, yaxiset); */
					/**Multimap<Integer, DoubleBox<TwoD>> multimm = 
						Pilihan.expansion(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), 
						xa, xb, ya, yb); **/
					Multimap<Integer, DoubleBox<TwoD>> multimm = 
							TwoDimen.expansionMinMax(matrixvalid, point, ridvalue, fungsi.getFirst(), rid, fungsi.getSecond(), 
							xa, xb, ya, yb, xaxiset, yaxiset);
				
					for (int t : multimm.keySet()) {
						List<DoubleBox<TwoD>> newbox = (List<DoubleBox<TwoD>>) multimm.get(t);
						for (DoubleBox<TwoD> nb: newbox) {
							Pair<TwoD, TwoD> nubox = nb.minX;
							Pair<TwoD, TwoD> unbox = nb.maxX;
							if (minplane.getFirst().x < nubox.getFirst().x) nubox.getFirst().x = minplane.getFirst().x;
							if (minplane.getFirst().y < nubox.getFirst().y) nubox.getFirst().y = minplane.getFirst().y;
							if (minplane.getSecond().x > nubox.getSecond().x) nubox.getSecond().x = minplane.getSecond().x;
							if (minplane.getSecond().y > nubox.getSecond().y) nubox.getSecond().y = minplane.getSecond().y;
							if (unbox.getFirst().x <= nubox.getFirst().x && unbox.getFirst().y <= nubox.getFirst().y &&
									unbox.getSecond().x >= nubox.getSecond().x && unbox.getSecond().y >= nubox.getSecond().y) {
								DoubleBox<TwoD> crossbox = new DoubleBox<>(nubox, unbox);
								crossmulti.add(crossbox);
								/**add into list*/
							}
						}
					}
				}
			}
		}
		return crossmulti;
	}
	
	public static <T> List<T> intersectArrays(List<T> f,
			List<T> g) {
		Map<T, Long> intersection = new HashMap<T, Long>();
		List<T> returnList = new LinkedList<T>();
		for (T elem : f) {
			Long count = intersection.get(elem);
			if (count != null) {
				intersection.put(elem, count+1);
			} else {
				intersection.put(elem, 1L);
			}
		}
		for (T elem : g) {
			Long count = intersection.get(elem);
			if (count != null && count > 0) {
				intersection.put(elem, count-1);
				returnList.add(elem);
			} else {
				intersection.put(elem, -1L);
			}
		}
		return returnList;
	}
	
	protected static <T> List<List<T>> cartesianProduct(List<List<T>> lists) {
	    List<List<T>> resultLists = new ArrayList<List<T>>();
	    if (lists.size() == 0) {
	        resultLists.add(new ArrayList<T>());
	        return resultLists;
	    } else {
	        List<T> firstList = lists.get(0);
	        List<List<T>> remainingLists = cartesianProduct(lists.subList(1, lists.size()));
	        for (T condition : firstList) {
	            for (List<T> remainingList : remainingLists) {
	                ArrayList<T> resultList = new ArrayList<T>();
	                resultList.add(condition);
	                resultList.addAll(remainingList);
	                resultLists.add(resultList);
	            }
	        }
	    }
	    return resultLists;
	}
}

/*
	private static <T extends Comparable<T>> Map<TwoD, BigDecimal> producePlaneDecimal(
			ListMultimap<Integer, Integer> fx, 
			ListMultimap<Integer, Integer> fy,
			Map<Integer, T> collect, Set<Integer> xaxiset, Set<Integer> yaxiset) {
		long produceTime = System.currentTimeMillis(); //-------------------------------------------------------
		Map<TwoD, BigDecimal> matrixmap = new HashMap<>();
		//List<Integer> sortX = new ArrayList<>(fx.keySet());
		//List<Integer> sortY = new ArrayList<>(fy.keySet());
		//Collections.sort(sortX);
		//Collections.sort(sortY);
		for (int x=0; x<fx.keySet().size(); x++) {
			for (int y=0; y<fy.keySet().size(); y++) {
				List<Integer> f = fx.get(x);
				List<Integer> g = fy.get(y);
				if (!f.contains(null) && !g.contains(null)) {
					//long produceTime = System.currentTimeMillis(); //-------------------------------------------------------
					List<Integer> c = new ArrayList<>(f);
					c.retainAll(g);
					//List<Integer> c = intersectArrays(f,g);
					//long produceEnd = System.currentTimeMillis();  //-------------------------------------------------------
					//double diff = (produceEnd - produceTime)/1e0;
					//System.out.println("----------------------------plane building-------------------------" + diff + " ms");
					
					if (!c.isEmpty()) {
						BigDecimal temp = new BigDecimal(0);
						for (int cc : c) {
							temp = temp.add(new BigDecimal(collect.get(cc).toString()));
						}
						TwoD dua = new TwoD(x,y);
						xaxiset.add(x); yaxiset.add(y);
						matrixmap.put(dua, temp);
					}
				}
			}
		}
		
		long produceEnd = System.currentTimeMillis();  //-------------------------------------------------------
		double diff = (produceEnd - produceTime)/1e0;
		System.out.println("----------------------------plane building-------------------------" + diff + " ms");
		return matrixmap;
		
	}
	
	private static <T extends Comparable<T>> Map<TwoD, Integer> producePlaneInteger(
			ListMultimap<Integer, Integer> fx, ListMultimap<Integer, Integer> fy, 
			Map<Integer, T> collect, Set<Integer> xaxiset, Set<Integer> yaxiset) {
		
		Map<TwoD, Integer> matrixmap = new HashMap<>();
		for (int x=0; x<fx.keySet().size(); x++) {
			for (int y=0; y<fy.keySet().size(); y++) {
				List<Integer> f = fx.get(x);
				List<Integer> g = fy.get(y);
				if (!f.contains(null) && !g.contains(null)) {
					List<Integer> c = intersectArrays(f,g);
					if (!c.isEmpty()) {
						//int temp = 0;
						//for (int cc : c) {
						//	temp += new Integer(collect.get(cc).toString());
						//}
						TwoD dua = new TwoD(x,y);
						xaxiset.add(x); yaxiset.add(y);
						matrixmap.put(dua, c.size());
					}
				}
			}
		}
		return matrixmap;
	}
	
*/