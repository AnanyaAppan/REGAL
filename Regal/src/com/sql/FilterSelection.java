package com.sql;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.BiMap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;

public class FilterSelection {
	public static <T extends Comparable<T>> void filterDiscover (
			Map<Map<String,Integer>, Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>>> whichtree,
			BiMap<Integer, String> columnName, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable,
			Map<List<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase, 
			Map <Integer, List<T>> baseColumn,
			Map<Integer, String> baseClass
			, Map<List<Integer>, List<Integer>> basePartialRow) 
	{
		
		BiMap<String, Integer> columnIndex = columnName.inverse();
		List<Map<String, Integer>> listOfTree = new ArrayList<>(whichtree.keySet());
		for (Map<String, Integer> sgTree : listOfTree) {
			System.out.println("FilterSelection_latticeRoot_" + sgTree);
			//@@@@Set<Integer> allBids = baseColumn.keySet();
			Map<CombNode, List<Map<Integer, Pair<Object, Integer>>>> whichnode = whichtree.get(sgTree);
			for (CombNode combnode : whichnode.keySet()) {
				ArrayList<String> cname = combnode.getAttr();
				ArrayList<Integer> combid = new ArrayList<>();
				ArrayList<Integer> comrid = new ArrayList<>();
				List<Map<String, List<Integer>>> combtuplebid = new ArrayList<>();
				
				for (String nm : cname) {
					combid.add(sgTree.get(nm));
					comrid.add(columnIndex.get(nm));
					//combtupleid.add(baseFreq.get(singTree.get(nm))); 
				}
				//@@
				Collections.sort(combid);
				for (int bid : combid) {
					combtuplebid.add(baseFreq.get(bid)); //refine order
				}
				//@@
				Map<List<String>, Multimap<Integer, T>> mapBID = latticeBase.get(combid);
				Map<List<String>, Multimap<Integer, T>> mapRID = latticeTable.get(comrid);
				Map<List<String>, Multimap<Integer, T>> reorderRID = new HashMap<>();
				for (List<String> ridTuple : mapRID.keySet()) {
					List<String> rerid = new ArrayList<>();
					for (int i=0; i < combnode.attr.size(); i++) {
						String attr = combnode.attr.get(i);
						int y = sgTree.get(attr);
						int z = combid.indexOf(y);
						rerid.add(ridTuple.get(z));
					}
					reorderRID.put(rerid, mapRID.get(ridTuple));
				}
				List<Integer> nongroupdimensi = basePartialRow.get(combid); //
				
				//List<Integer> nongroupdimensi = new ArrayList<Integer>(allBids); //basePartialRow
				//nongroupdimensi.removeAll(combid);
				//System.out.println("filterDiscover.DimensionInputRelation " + nongroupdimensi);
				//System.out.println("FilterSelection.reorderRID " + reorderRID.keySet());
				
				List<Map<Integer, Pair<Object, Integer>>> whichaggre = whichnode.get(combnode);
				System.out.println("FilterSelection.grouping_" + combnode + "_multi-dimensions_" + nongroupdimensi 
						+ "_aggregation-combinations_" + whichaggre.size());
				
				Map<Integer, Pair<Object, Integer>> firstaggre = whichaggre.get(0);
				List<Integer> ridorder = new ArrayList<>(firstaggre.keySet()); //columns for aggregations
				scaleGroups(combnode, combtuplebid, mapBID, reorderRID, baseColumn, baseFreq, baseClass, 
						whichaggre, ridorder, nongroupdimensi);
			}
		}
	}

	private static <T extends Comparable<T>> void scaleGroups(CombNode combnode,
			List<Map<String, List<Integer>>> combtuplebid,
			Map<List<String>, Multimap<Integer, T>> mapBID, 
			Map<List<String>, Multimap<Integer, T>> reorderRID,
			Map <Integer, List<T>> baseColumn, Map<Integer, Map<String, List<Integer>>> baseFreq,
			Map<Integer, String> baseClass,List<Map<Integer, Pair<Object, Integer>>> whichaggre, 
			List<Integer> ridorder, 
			List<Integer> nongroupdimensi) 
	{
		
		Map<List<String>, Integer> groupsizematter = new HashMap<>();
		Multimap<List<String>, DimenIndex<T>> ridTupleIndex = ArrayListMultimap.create();
		Multimap<List<String>, DimenIndex<T>> freqConstraint = ArrayListMultimap.create();
		
		ArrayList<List<String>> fromPhaseTwo = new ArrayList<>(); //groupings that no any selection
		if (combnode.isFrequency == true) {
			fromPhaseTwo = combnode.getInferFrequency();
		}
		
		double wotd = 0.0;// why - how :: the process of finding common indexes in matrices
		double hotd = 0.0;// which - where :: finding suitable dimensions (the length of dimension)
		double lotd = 0.0;// s - e :: the process of dimension_(sort, row_id)
		double totd = 0.0;// total time from start to end
		
		Set<Integer> renew_nongroup = new HashSet<>(); //@@@@
		
		for (List<String> retuple : reorderRID.keySet()) { //groupings in query output table
			long time_start = System.currentTimeMillis();
			if ((mapBID != null) && (fromPhaseTwo != null) && mapBID.keySet().contains(retuple) && !fromPhaseTwo.contains(retuple)) { //eliminate the whole range
				long why = System.currentTimeMillis();
				List<List<Integer>> listupleid = new ArrayList<>();
				for (String reitem : retuple) {
					List<Integer> tupleid = combtuplebid.get(retuple.indexOf(reitem)).get(reitem);
					listupleid.add(tupleid);
				}
				//@@
				Set<Integer> setcombid = PreprocessBase.commonListIndices(listupleid);
				//@@
				List<Integer> listcombid = new ArrayList<>(setcombid);
				//System.out.println("Size of common list " + listcombid.size());
				groupsizematter.put(retuple, listcombid.size());  //size of the base partition
				
				Map<Integer, T> aggreValue = new HashMap<>();
				for (int r : ridorder) {
					List<T> ridvalu= (List<T>) reorderRID.get(retuple).get(r);
					aggreValue.put(r, ridvalu.get(0));
				}
				System.out.println("FilterSelection.matrix_" + retuple + "_boundingValues_" + aggreValue); 
				long how = System.currentTimeMillis();
				wotd += (how - why)/1e0;
				
				for (int i : nongroupdimensi) 
				{ 
					//if (!baseClass.get(i).contains("BigDecimal"))  
					//{		
					long where = System.currentTimeMillis();
					ridTupleIndex.put(retuple, new DimenIndex<T>(i));
					Set<T> wholeColum = new HashSet<>(baseColumn.get(i));
					List<T> sortColum = new ArrayList<>(wholeColum);
					Collections.sort(sortColum); //suppkey e.g. 10000
					long which = System.currentTimeMillis();
					hotd += (which - where)/1e0;
					
					//System.out.println("index_" + i + "_sort_" + sortColum.size());
					if (sortColum.size() <= 10000) {
						
					long s = System.currentTimeMillis();	
					Map<Integer, T> haveCombid = new HashMap<>();
					ListMultimap<Integer, Integer> indexID2 = ArrayListMultimap.create();
					//Map<Integer, List<Integer>> indexID2 = new HashMap<>();
					
					for (int combid : listcombid) { 				//___each rowID in base partition 
						T valu = baseColumn.get(i).get(combid);
						haveCombid.put(combid, valu); 				//(rowID, dimension value)
						int tros = sortColum.indexOf(valu);
						indexID2.put(tros, combid); 				//(sort value, rowID)
						/**if (indexID2.keySet().contains(tros)) {
							indexID2.get(tros).add(combid);
						} else {
							List<Integer> star = new ArrayList<>();
							star.add(combid);
							indexID2.put(tros, star);
						}*/
					}
					
					/**Set<Integer> rightkey = indexID2.keySet();
					if (indexID2.size() != sortColum.size()) {
						for (int l=0; l<sortColum.size(); i++) {
							if (!indexID2.containsKey(l)) {
								indexID2.put(l, null);
							}
						}
					}*/
					
					long e = System.currentTimeMillis();
					lotd += (e-s)/1e0 ;
					//@@@@
					//****System.out.println("size_of_base_table_partition_" + haveCombid.size());
					//****System.out.println("size_of_dimension_scale_" + indexID2.size());
					/**Map<String, List<Integer>> uniqueItem = baseFreq.get(i);
					Map<Integer, List<Integer>> indexID = new HashMap<>();
					
					long longest = System.currentTimeMillis();	//kill time here
					for (int k = 0; k < sortColum.size(); k++) { //each suppkey
						List<Integer> indexIDvalu = new ArrayList<>();
						if (uniqueItem.get(sortColum.get(k).toString()) != null) {
							List<Integer> ids = uniqueItem.get(sortColum.get(k).toString());
							List<Integer> newcomid = new ArrayList<>(listcombid);
							//System.out.println("newcomid.size() " + newcomid.size() + " " 
							//+ " ids.size() " + ids.size() );
							Set<Integer> indexonly = 
									Sets.intersection(Sets.newHashSet(newcomid), Sets.newHashSet(ids));
							List<Integer> useSets = new ArrayList<>(indexonly);
							//newcomid.retainAll(ids);
							//System.out.println("afterRetainAll newcomid.size() " + useSets.size());
							if (!useSets.isEmpty()) {
								indexIDvalu.addAll(useSets);
							} else {
								indexIDvalu.add(null);
							}
						}
						indexID.put(k, indexIDvalu);
					}
					
					long longest_end = System.currentTimeMillis();
					lotd += (longest_end - longest)/1e0;
					System.out.println("size of uniquekeys " + indexID.size() + " time " + lotd + " ms");
					*/
					
					for (DimenIndex<T> cv : ridTupleIndex.get(retuple)) {
						if (cv.index == i) {
							cv.letT = sortColum;
							cv.ridValue = aggreValue;
							cv.indexMatch = haveCombid;
							cv.newBuildD = indexID2;  //@@@@
						}
					}
					
					renew_nongroup.add(i);
					}
					//} //
					//else if (baseClass.get(i).contains("BigDecimal")) {
					else { //for attributes that distinct values >= 10000 (because need to be discrete)
					Map<Integer, T> haveCombid = new HashMap<>();
					for (int combid : listcombid) { 
						haveCombid.put(combid, baseColumn.get(i).get(combid));
					}
					for (DimenIndex<T> cv : ridTupleIndex.get(retuple)) {
						if (cv.index == i) {
							cv.letT = sortColum;
							cv.ridValue = aggreValue;
							cv.indexMatch = haveCombid;
						}
					}
					}
				}
			} else if (fromPhaseTwo.contains(retuple)) {
				List<List<Integer>> listupleid = new ArrayList<>();
				for (String reitem : retuple) {
					List<Integer> tupleid = combtuplebid.get(retuple.indexOf(reitem)).get(reitem);
					listupleid.add(tupleid);
				}
				
				Set<Integer> setcombid = PreprocessBase.commonListIndices(listupleid);
				List<Integer> listcombid = new ArrayList<>(setcombid);
				
				Map<Integer, T> aggreValue = new HashMap<>();
				for (int r : ridorder) {
					List<T> ridvalu= (List<T>) reorderRID.get(retuple).get(r);
					aggreValue.put(r, ridvalu.get(0));
				}
				
				for (int i : nongroupdimensi) { 
					//if (!baseClass.get(i).contains("BigDecimal")) { //@@@@
					freqConstraint.put(retuple, new DimenIndex<T>(i));
					Set<T> wholeColum = new HashSet<>(baseColumn.get(i));
					List<T> sortColum = new ArrayList<>(wholeColum);
					Collections.sort(sortColum);
					
					//if (sortColum.size() <= 10000) { //@@@@
					Map<Integer, T> haveCombid = new HashMap<>();
					for (int combid : listcombid) { 
						haveCombid.put(combid, baseColumn.get(i).get(combid)); 
					}
					for (DimenIndex<T> cv : freqConstraint.get(retuple)) {
						if (cv.index == i) {
							cv.letT = sortColum;
							cv.indexMatch = haveCombid;
							cv.ridValue = aggreValue;
							//cv.newBuildD = indexID;
						}
					}
					renew_nongroup.add(i);
					//}
				}
			}
			long time_end = System.currentTimeMillis();
			totd += (time_end - time_start)/1e0;
			
			//else the tuple is existed in frequencyList in phaseTwo
			//calculate min fuzzy bounding box for them
		} 
		System.out.println("time1_" + wotd + "_time2_" + hotd + "_time3_" + lotd + " = " + totd + "_ms");
		List<Integer> renew = new ArrayList<>(renew_nongroup);
		//All mapBID.keySet() - All mapRID.keySet() = validationTuples
		if (!ridTupleIndex.isEmpty()){
			System.out.println("Start Experiment from One Dimensi");
			Map<List<String>, Integer> ascending = sortByValue(groupsizematter);
			
			long time_start = System.currentTimeMillis();
			Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> whereSatu = 
					startOneValid(renew, ridTupleIndex, baseClass, whichaggre, ridorder, ascending);
			long time_one = System.currentTimeMillis();
			double one = (time_one - time_start)/1e0;
			System.out.println("------------------------one dimension--------------------- " + one + "_ms");
			if(!whereSatu.isEmpty()) {
				
				ArrayList<Pair<Integer, DoubleBox<Integer>>> whereSatuSet = 
						new ArrayList<>(whereSatu.keySet());
				//-------------------------------------------------------------
				for (int i=0; i<whereSatu.size(); i++) {
					System.out.println(whereSatuSet.get(i).getFirst() + " " + 
							whereSatuSet.get(i).getSecond().minX + " " +
							whereSatuSet.get(i).getSecond().maxX);
					//---------------------------------------------------------
					List<Pair<Integer, Integer>> otherMin = new ArrayList<>();
					List<T> unit = new ArrayList<>();
					if (!freqConstraint.isEmpty()) {
						for (List<String> tuple : freqConstraint.keySet()) {
							for (DimenIndex<T> di : freqConstraint.get(tuple)) {
								if (di.index == whereSatuSet.get(i).getFirst()) {
									unit = di.letT;
									Pair<Integer,Integer> other = 
											new Pair<>(unit.indexOf(Collections.min(di.indexMatch.values())),
											unit.indexOf(Collections.max(di.indexMatch.values())));
									otherMin.add(other);
								}
							}
						}
					}
					else {
						for (List<String> tuple : ridTupleIndex.keySet()) {
							for (DimenIndex<T> di : ridTupleIndex.get(tuple)) {
								if (di.index == whereSatuSet.get(i).getFirst()) {
									unit = di.letT;
								}
							}
						}
					}
					otherMin.add(whereSatuSet.get(i).getSecond().minX);
					List<Integer> leftsmall = new ArrayList<>();
					List<Integer> rightsmall = new ArrayList<>();
					for (Pair<Integer, Integer> small : otherMin) {
						leftsmall.add(small.getFirst());
						rightsmall.add(small.getSecond());
					}
					//System.out.println("left " + leftsmall + " right " + rightsmall); //@@@@
					Pair<Integer, Integer> newmin = 
							new Pair<>(Collections.min(leftsmall),Collections.max(rightsmall));
					Pair<Integer, Integer> newmax = whereSatuSet.get(i).getSecond().maxX;
					if (newmin.getFirst() >= newmax.getFirst() && newmin.getSecond() <= newmax.getSecond()) 
					{
						Pair<T,T> smallT = 
								new Pair<>(unit.get(Collections.min(leftsmall)), 
										unit.get(Collections.max(rightsmall)));
						int xmin = newmax.getFirst();      //@@@@@@@@@@@-1
						int xmax = newmax.getSecond();	   //@@@@@@@@@@@+1
						T tmin = (xmin >= 0) ? unit.get(xmin) : null;  //@@@@@@@@@@@@>=
						T tmax = (xmax < unit.size()) ? unit.get(xmax) : null; 
						Pair<T,T> bigT = new Pair<>(tmin, tmax);
						System.out.println("OneDimenFilter BaseIndex=" + whereSatuSet.get(i).getFirst() + " " + 
						smallT + " " + bigT + " " + whereSatu.get(whereSatuSet.get(i)));
					}
				}
				
			} 
			else if(renew.size() >= 2){
				System.out.println("Failed one Dimension ! then start two Dimension");
				//TwoDimension 
				
				System.out.println("AfterOneDimension " + ridTupleIndex.keySet().size());
				long after_one = System.currentTimeMillis();
				Map<Pair<ArrayList<Integer>, DoubleBox<TwoD>>, List<Pair<Object, Integer>>> rect = 
					TwoDimension.regionCut(renew, ridTupleIndex, baseClass, whichaggre, ridorder);
				long time_two = System.currentTimeMillis();
				double two = (time_two - after_one)/1e0;
				System.out.println("------------------------two dimension--------------------- " + two + " ms");
				if (!rect.isEmpty()) {
					ArrayList<Pair<ArrayList<Integer>, DoubleBox<TwoD>>> rectkey = 
							new ArrayList<>(rect.keySet());
					for (Pair<ArrayList<Integer>, DoubleBox<TwoD>> lhkey : rectkey) {
						System.out.println("Aggregation " + rect.get(lhkey));
						ArrayList<Integer> unitindex = lhkey.getFirst();
						DoubleBox<TwoD> ranges = lhkey.getSecond();
						List<T> xlength = new ArrayList<>();
						List<T> ylength = new ArrayList<>();
						List<List<String>> querygroup = new ArrayList<>(ridTupleIndex.keySet());
						for (DimenIndex<T> di : ridTupleIndex.get(querygroup.get(0))) {
							if (di.index == unitindex.get(0)) {
								xlength = di.letT;	
							} 
							if (di.index == unitindex.get(1)) {
								ylength = di.letT;
							}
						}
					//System.out.println(unitindex + " " + xlength + " " + ylength);
						T xmin = xlength.get(ranges.maxX.getFirst().x);
						T xmax = xlength.get(ranges.maxX.getSecond().x);
						T ymin = ylength.get(ranges.maxX.getFirst().y);
						T ymax = ylength.get(ranges.maxX.getSecond().y);
						System.out.println("[" + xmin + " " + xmax + "] [" + ymin + " " + ymax + "]");
					}
				} 
				else 
				{
					System.out.println("Failed two Dimension ! then start three Dimension loh");
					long after_two = System.currentTimeMillis();
					Map<Pair<ArrayList<Integer>, DoubleBox<ThreeD>>, List<Pair<Object, Integer>>> cube = 
							ThreeDimension.cubeCut(renew, ridTupleIndex, baseClass, 
									whichaggre, ridorder);
					long time_thr = System.currentTimeMillis();
					double three = (time_thr - after_two)/1e0;
					System.out.println("------------------------three dimension--------------------- " + three + " ms");
					if (!cube.isEmpty()) {
						ArrayList<Pair<ArrayList<Integer>, DoubleBox<ThreeD>>> cubekey = 
								new ArrayList<>(cube.keySet());
						for (Pair<ArrayList<Integer>, DoubleBox<ThreeD>> cbk : cubekey) {
							System.out.println("Aggregation " + cube.get(cbk));
							ArrayList<Integer> unitindex = cbk.getFirst();
							DoubleBox<ThreeD> ranges = cbk.getSecond();
							List<T> xlength = new ArrayList<>();
							List<T> ylength = new ArrayList<>();
							List<T> zlength = new ArrayList<>();
							List<List<String>> querygroup = new ArrayList<>(ridTupleIndex.keySet());
							for (DimenIndex<T> di : ridTupleIndex.get(querygroup.get(0))) {
								if (di.index == unitindex.get(0)) {
									xlength = di.letT;	
								} 
								if (di.index == unitindex.get(1)) {
									ylength = di.letT;
								}
								if (di.index == unitindex.get(2)) {
									zlength = di.letT;
								}
							}
						
							T xmin = xlength.get(ranges.maxX.getFirst().x);
							T xmax = xlength.get(ranges.maxX.getSecond().x);
							T ymin = ylength.get(ranges.maxX.getFirst().y);
							T ymax = ylength.get(ranges.maxX.getSecond().y);
							T zmin = zlength.get(ranges.maxX.getFirst().z);
							T zmax = zlength.get(ranges.maxX.getSecond().z);
							System.out.println("[" + xmin + " " + xmax + "] "
									+ "[" + ymin + " " + ymax + "] "
											+ "[" + zmin + " " + zmax +"]");
						}
					}
				} 
				//*/
			} 
			long time_end = System.currentTimeMillis();
			double time_diff = (time_end - time_start)/1e0;
			System.out.println("time for one/two/three dimensi_" + time_diff + "_ms");
			
		} 
		else System.out.println("No any filter on aggregate attributes for groupby " + combnode.attr);
		
	}
	
	public static <K, V extends Comparable<? super V>> Map<K, V> 
    sortByValue( Map<K, V> map )
    {
		List<Map.Entry<K, V>> list = new LinkedList<>( map.entrySet() );
		Collections.sort( list, new Comparator<Map.Entry<K, V>>()
		{
			@Override
			public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
			{
				return (o1.getValue()).compareTo( o2.getValue() );
			}
		} );

		Map<K, V> result = new LinkedHashMap<>();
		for (Map.Entry<K, V> entry : list)
		{
			result.put( entry.getKey(), entry.getValue() );
		}
		return result;
    }
	
	private static <T extends Comparable<T>> Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> 
	startOneValid (
			List<Integer> nongroupdimensi, 
			Multimap<List<String>, DimenIndex<T>> ridTupleIndex,
			Map<Integer, String> baseClass, 
			List<Map<Integer, Pair<Object, Integer>>> whichaggre, 
			List<Integer> ridorder, Map<List<String>, Integer> ascending) 
	{
		
		Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> whereSatu = new HashMap<>();
		List<List<String>> reshuffle = new ArrayList<>(ascending.keySet());
		
		//long seed = System.nanoTime();
		//Collections.shuffle(reshuffle, new Random(seed));
		//long sed = System.nanoTime();
		//Collections.shuffle(reshuffle, new Random(sed));
		//Collections.reverse(reshuffle);	
		
		for (int ib : nongroupdimensi) {
			//if (!baseClass.get(ib).contains("String")) {	//@@@@
			//List<Pair<Object, Integer>> pruningRedundant = new ArrayList<>();
			List<T> wholeIndex = new ArrayList<>();
			for (Map<Integer, Pair<Object, Integer>> aggpernode : whichaggre) {
				List<Pair<Object, Integer>> functioname = new ArrayList<>(aggpernode.values());
				
				List<DoubleBox<Integer>> updateValid = new ArrayList<>();
				
				outer:
				for (List<String> ridTuple : reshuffle) {
					System.out.println("index_" + ib + "_combination_" + functioname + "_matrix_" + ridTuple);
					long singlegroup = System.currentTimeMillis();
					Map<Integer, Map<Integer, T>> comidValue = new HashMap<>();
					ListMultimap<Integer, Integer> rangeId = ArrayListMultimap.create();
					//Map<Integer, List<Integer>> rangeId = new HashMap<>();
					Map<Integer, T> ridValue = new HashMap<>();
					for (DimenIndex<T> cv : ridTupleIndex.get(ridTuple)) {
						comidValue.put(cv.index, cv.indexMatch);
						if (cv.index == ib) {
							rangeId = cv.newBuildD;
							wholeIndex = cv.letT;
							ridValue = cv.ridValue;
						}
					}
					Collections.sort(ridorder);
					Collections.reverse(ridorder);
					for (int jr : ridorder) {
						Pair<Object, Integer> fungsi = aggpernode.get(jr);
						if (updateValid.isEmpty()) {
							
							Multimap<Integer, DoubleBox<Integer>> forone = 
									perusalAggregate(rangeId, fungsi, ridValue.get(jr), 
											comidValue.get(fungsi.getSecond()),
											baseClass.get(fungsi.getSecond()), wholeIndex, fungsi.getSecond(), jr);
							if (forone.isEmpty()) {
								System.out.println("this aggregate returns empty"); 		
								
								break outer;
							} else {
								updateValid.addAll(new ArrayList<>(forone.get(jr))); 
								System.out.println("Created bounding box " + updateValid.size() + " " + fungsi);
								/**for (DoubleBox<Integer> up : updateValid) {
									System.out.println("Created bounding box ["+up.minX.getFirst()+" "
									+up.minX.getSecond()+"]"+ 
										"["+up.maxX.getFirst()+" "+up.maxX.getSecond()+"]");
								} */
							}
						} else {
							
							if (updateValid.size() > 1) {
								for (int u=0 ; u < updateValid.size()-1 ; u++) {
									DoubleBox<Integer> subplane = updateValid.get(u);
									for (int w=u+1; w < updateValid.size(); w++) {
										DoubleBox<Integer> dubplane = updateValid.get(w);
										if (dubplane.maxX.getFirst().equals(subplane.maxX.getFirst()) && 
												dubplane.maxX.getSecond().equals(subplane.maxX.getSecond()) ) {
											updateValid.remove(dubplane);
										}
									}
								}
							}
							List<DoubleBox<Integer>> repeatupdate = new ArrayList<>();
							for (int r=0; r<updateValid.size(); r++) {
								DoubleBox<Integer> subplane = updateValid.get(r);
								//cross validation
								List<DoubleBox<Integer>> crossmulti = 
										Extrawork.crossValidation(subplane, comidValue.get(fungsi.getSecond()), 
												baseClass.get(fungsi.getSecond()), 
												fungsi, ridValue.get(jr), jr, rangeId, wholeIndex);
								repeatupdate.addAll(crossmulti);
							}
							if (repeatupdate.isEmpty()) {
								System.out.println("cross aggregate no valid");
								updateValid.clear();
								break outer;
							} else {
								updateValid.clear();
								updateValid = repeatupdate; 
								//System.out.println("Updated bounding box " + updateValid.size() + " " + fungsi);
								/**for (DoubleBox<Integer> up : updateValid) {
									System.out.println("Updated bounding box " + 
									"["+up.minX.getFirst()+" "+up.minX.getSecond()+"]"+ 
											"["+up.maxX.getFirst()+" "+up.maxX.getSecond()+"]");
								} */ 
							}
							
						}
					}
					long singlegroup_end = System.currentTimeMillis();
					double diff = (singlegroup_end - singlegroup)/1e0;
					System.out.println("Time for grouping_" + ridTuple + " = " + diff + "_ms");
				}
				if (!updateValid.isEmpty()) {
					for (DoubleBox<Integer> cross : updateValid) {
						/**System.out.println("store into wheresatubiru " + 
						 		"["+cross.minX.getFirst()+" "+cross.minX.getSecond()+"]"+ 
								"["+cross.maxX.getFirst()+" "+cross.maxX.getSecond()+"]"); 
						*/
					
						Pair<Integer, DoubleBox<Integer>> whereSatuMerah = new Pair<>(ib, cross);
						whereSatu.put(whereSatuMerah, functioname);
					}
				} else {
					System.out.println("not any box returns");
				}
			}
		//} //@@@@
		}
		return whereSatu;
	}
	
	private static <T extends Comparable<T>> Multimap<Integer, DoubleBox<Integer>> perusalAggregate (
			ListMultimap<Integer, Integer> rangeId, Pair<Object, Integer> aggregation, T ridValue, 
			Map<Integer, T> comidElem, String unitclass, List<T> wholeIndex, int j, int rid) 
	{
		
		Multimap<Integer, DoubleBox<Integer>> multiFunBox = ArrayListMultimap.create();
		
		int startLen = 0;
		int unitLen = wholeIndex.size(); 
		List<Integer> startEnd = new ArrayList<>();
		for (int i = startLen; i < unitLen; i++) {
			startEnd.add(i);
		}
		if (unitclass.contains("Integer") ) //&& aggregation.getFirst() == "count")   
		{
			//List<Integer> unitInteger = new ArrayList<>();
			Map<Integer, Integer> unitMapInteger = new HashMap<>();
			Object function = aggregation.getFirst();
			List<Integer> count = new ArrayList<>();
			List<Integer> pinpoint = new ArrayList<>();
			//@@@@
			//unitInteger = IntegerScale.produceUnit(function, rangeId, comidElem);
			//@@@@
			int result = Integer.parseInt(ridValue.toString());
			unitMapInteger = IntegerScale.temProduceInteger(function, rangeId, comidElem, count, result, pinpoint);
			List<Integer> unitKey = new ArrayList<Integer>(unitMapInteger.keySet());
			Collections.sort(unitKey);
			int totalsum = 0;
			for (int i : unitKey) {
				totalsum += unitMapInteger.get(i);
			}
			if (function == "sum") {
				
				OneDimenDigit.sumpop(unitMapInteger, result, rid, unitKey, startLen, unitLen, 
						multiFunBox, function, j, totalsum);
				
			} else if (function == "avg") {
				
				int times = 0;
				for (int u = 0; u < count.size(); u++) {
					times += count.get(u);
				}
				OneDimenDigit.avgop(unitMapInteger, result, rid, unitKey, startLen, unitLen, 
						multiFunBox, function, j, totalsum, count, times);
					
			} else if (function == "max") {
				OneDimenDigit.betterexpand(unitMapInteger, result,  rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, pinpoint);
			} else if (function == "min") {
				OneDimenDigit.betterexpand(unitMapInteger, result,  rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, pinpoint);
			}	
			
		} else if (unitclass.contains("BigDecimal")) //&& aggregation.getFirst() == "sum") //@@@@
		{
			
			Map<Integer, BigDecimal> unitMapDecimal = new HashMap<>();
			Object function = aggregation.getFirst();
			BigDecimal result = new BigDecimal(ridValue.toString());
			List<Integer> count = new ArrayList<>();
			List<Integer> pinpoint = new ArrayList<>();
			
			//@@@@
			long q = System.currentTimeMillis();
			unitMapDecimal = 
					IntegerScale.temProduceDecimal(function, rangeId, comidElem, count, result, pinpoint);
			
			long m = System.currentTimeMillis();
			double g = (m-q)/1e0;
			System.out.println("fill_matrix_cost_" + g + "_ms");
			//@@@@
			
			List<Integer> unitKey = new ArrayList<Integer>(unitMapDecimal.keySet());
			Collections.sort(unitKey);
			
			BigDecimal totalsum = new BigDecimal(0);
			for (int i : unitKey) {
				totalsum = totalsum.add(unitMapDecimal.get(i));
			}

			if (function == "sum") {
				
				OneDimenDecimal.sumop(unitMapDecimal, result, rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, totalsum);
				//SumExpandShrink.sumop(unitMapDecimal, result, rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, totalsum);
				/*int initial = 0;
				while (initial < unitMapDecimal.size()) {
					SumExpandShrink.nativexpand(unitMapDecimal, result, rid, initial, 
							startLen, unitLen, unitKey, multiFunBox, function, j);
					initial++;
				} */
				//expansion for sum
				//SumExpandShrink.decimalSumFilter(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j); 
				
				//shrinking for sum
				//SumExpandShrink.shrinkSumFilter(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, totalsum);
				
			} else if (function == "avg") {
				
				int times = 0;
				for (int u = 0; u < count.size(); u++) {
					times += count.get(u);
				}
				OneDimenDecimal.avgop(unitMapDecimal, result, rid, unitKey, startLen, unitLen, 
						multiFunBox, function, j, totalsum, count, times);
				//SumExpandShrink.avgop(unitMapDecimal, result, rid, unitKey, startLen, unitLen, 
				//		multiFunBox, function, j, totalsum, count, times);
				//expansion for avg
				/**int initial = 0;
				while (initial < unitMapDecimal.size()) {
					SumExpandShrink.trialAverage(unitMapDecimal, result,  rid, initial, 
							startLen, unitLen, unitKey, multiFunBox, function, j, count);
					initial++;
				} */ 
				//SumExpandShrink.shrinkAvgFilter(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, count, totalsum);	
			} 
			else if (function == "max") {
				
				OneDimenDecimal.betterexpand(unitMapDecimal, result,  rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, pinpoint);
				//MonoExpandShrink.betterexpand(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, pinpoint); 
				//MonoExpandShrink.monopo(unitMapDecimal, result, rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j);
				//MonoExpandShrink.shrinkMaxMin(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, pinpoint);
				//MonoExpandShrink.exhausticShrink(0, unitKey.size(), unitMapDecimal, result, unitKey, 
				//		function, multiFunBox, rid);
				
				
			}
			else if (function == "min") {
				
				OneDimenDecimal.betterexpand(unitMapDecimal, result,  rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, pinpoint);
				//MonoExpandShrink.shrinkMaxMin(unitMapDecimal, result,  rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, pinpoint);
				//MonoExpandShrink.monopo(unitMapDecimal, result, rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, pinpoint);
				//MonoExpandShrink.exhausticShrink(0, unitKey.size(), unitMapDecimal, result, unitKey, 
				//		function, multiFunBox, rid);
				
			} 
			else if (function == "median") {
				//System.out.println("rangeId_" + rangeId);
				Multimap<Integer, BigDecimal> unitMultiDecimal = ArrayListMultimap.create();
				List<Integer> tnuoc = new ArrayList<>();
				unitMultiDecimal =
						IntegerScale.temProduceMulti(rangeId, comidElem, result, tnuoc);
				List<Integer> sortKey = new ArrayList<Integer>(unitMultiDecimal.keySet());
				Collections.sort(sortKey);
				int times = 0;
				for (int u = 0; u < tnuoc.size(); u++) {
					times += tnuoc.get(u);
				}
				
				BigDecimal[] allDecimals = new BigDecimal[times];
				int y = 0;
				for (int k : sortKey) {
					List<BigDecimal> values = (List<BigDecimal>) unitMultiDecimal.get(k);
					for (int z = 0; z < values.size(); z++) {
						allDecimals[z+y] = values.get(z);
					}
					y += values.size();
				}
				//System.out.println(startLen + "---" + unitLen);
				AdditionalUDF.searchMedian(unitMultiDecimal, result, sortKey, multiFunBox, allDecimals, tnuoc, 
						startLen, unitLen, rid);
				//
				//List<BigDecimal> tempList = new ArrayList<>(Arrays.asList(allDecimals));
				//Collections.sort(tempList);
				//AdditionalUDF.decrementMedian(unitMultiDecimal, result, sortKey, multiFunBox, tempList, tnuoc, 
				//		startLen, unitLen, rid);
			}
		} else if (unitclass.equals("count")) {
			
			Map<Integer, Integer> unitMapInteger = new HashMap<>();
			Object function = aggregation.getFirst();
			
			unitMapInteger = IntegerScale.temProduceCount(function, rangeId, comidElem);
			
			int totalcount = 0;
			for (int i : unitMapInteger.keySet())
				totalcount += unitMapInteger.get(i);
			
			int result = Integer.parseInt(ridValue.toString());
			List<Integer> unitKey = new ArrayList<Integer>(unitMapInteger.keySet());
			Collections.sort(unitKey);
			
			if (function == "count") {
				
				OneDimenDigit.countop(unitMapInteger, result, rid, unitKey, 
						startLen, unitLen, multiFunBox, function, j, totalcount);
				/*int initial = 0;
				while (initial < unitMapInteger.size()) {
					CountExpandShrink.countexpand(unitMapInteger, result, rid, initial, 
							startLen, unitLen, unitKey, multiFunBox, function, j);
					initial++;
				} */
				//CountExpandShrink.countfilter(unitMapInteger, result, rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j);  
				//CountExpandShrink.shrinkcountfilter(unitMapInteger, result, rid, unitKey, 
				//		startLen, unitLen, multiFunBox, function, j, totalcount); 
			}
		}
		return multiFunBox;
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

/**
private static <T extends Comparable<T>> Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> 
startOneMix (
		List<Integer> nongroupdimensi, 
		Multimap<List<String>, DimenIndex<T>> ridTupleIndex,
		Map<Integer, String> baseClass, 
		List<Map<Integer, Pair<Object, Integer>>> whichaggre, List<Integer> ridorder, 
		Map<List<String>, Integer> gan) 
{
	
	Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> whereSatuBiru = new HashMap<>();
	List<List<String>> reshuffle = new ArrayList<>(gan.keySet());
	//long seed = System.nanoTime();
	//Collections.shuffle(reshuffle, new Random(seed));
	
	//Collections.reverse(reshuffle);
	System.out.println("1d-Filter on dimensi " + nongroupdimensi + " for group " + reshuffle);
	for (int ib : nongroupdimensi) {
		if (!baseClass.get(ib).contains("BigDecimal")) {
		List<T> wholeIndex = new ArrayList<>();
		
		for (Map<Integer, Pair<Object, Integer>> aggpernode : whichaggre) {
			List<Pair<Object, Integer>> functioname = new ArrayList<>(aggpernode.values());
			List<List<DoubleBox<Integer>>> crossgroup = new ArrayList<>();
			List<DoubleBox<Integer>> crossvalid = new ArrayList<>();
			
			outer:
				
			for (List<String> ridTuple : reshuffle) {	
				System.out.println(ib + " " + functioname + " " + ridTuple);
				long single_start = System.currentTimeMillis();
				Map<Integer, Map<Integer, T>> comidValue = new HashMap<>();
				ListMultimap<Integer, Integer> rangeId = ArrayListMultimap.create();
				Map<Integer, T> ridValue = new HashMap<>();
				for (DimenIndex<T> cv : ridTupleIndex.get(ridTuple)) {
					comidValue.put(cv.index, cv.indexMatch);
					if (cv.index == ib) {
						rangeId = cv.newBuildD;
						wholeIndex = cv.letT;
						ridValue = cv.ridValue;
					}
				}
				if (crossvalid.isEmpty()) {
				Multimap<Integer, DoubleBox<Integer>> forall = ArrayListMultimap.create();
				for (int jr : ridorder) {
					Pair<Object, Integer> fungsi = aggpernode.get(jr);
					Multimap<Integer, DoubleBox<Integer>> forone = 
							perusalAggregate(rangeId, fungsi, ridValue.get(jr), 
									comidValue.get(fungsi.getSecond()), baseClass.get(fungsi.getSecond()), 
									wholeIndex, fungsi.getSecond(), jr);
					if (forone.isEmpty()) {
						break outer;
					} else {
						List<DoubleBox<Integer>> updateValid = new ArrayList<>(forone.get(jr)); 
						System.out.println("Created bounding box " + updateValid.size());
						for (DoubleBox<Integer> up : updateValid) {
							System.out.println("Created bounding box "
									+ "["+up.minX.getFirst()+" "+up.minX.getSecond()+"]"+ 
								"["+up.maxX.getFirst()+" "+up.maxX.getSecond()+"]");
						} 
						forall.putAll(forone);
					}
				}
				
				List<List<DoubleBox<Integer>>> overallScale = new ArrayList<>();
				for (int k : forall.keySet()) {
					List<DoubleBox<Integer>> pairFilter = (List<DoubleBox<Integer>>) forall.get(k);
					overallScale.add(pairFilter);
				}
				List<List<DoubleBox<Integer>>> everyScale = cartesianProduct(overallScale);
				
				List<DoubleBox<Integer>> ui = crossaggregate(everyScale);
				if (!ui.isEmpty()) {
					crossgroup.add(ui);
				} else {
					break outer;
				}
				
				if (crossgroup.size() == 7) {
					long cross_start = System.currentTimeMillis();
					//@@@@
					List<List<DoubleBox<Integer>>> cutgroup = crossgroup.subList(0, 2);
					List<List<DoubleBox<Integer>>> tempgroup = cartesianProduct(cutgroup);
					List<DoubleBox<Integer>> duajadisatu = crossaggregate(tempgroup);
					int hl = 2;
					if (hl < crossgroup.size() && !duajadisatu.isEmpty()) {
						lagilagi(duajadisatu, crossgroup, hl, 7, crossvalid);
						if (crossvalid.isEmpty()) {
							System.out.println(" One Dimension failed becos fail crossgroup aggregation");
						}
					} else {
						System.out.println(" One Dimension failed ");
					} 
					//@@@@
					List<List<DoubleBox<Integer>>> combinegroup = cartesianProduct(crossgroup);	
					List<DoubleBox<Integer>> actuallycrossgroup = crossaggregate(combinegroup);		
					if (!actuallycrossgroup.isEmpty()) {
						crossvalid.addAll(actuallycrossgroup);
					} else {
						System.out.println(" One Dimension failed becos fail crossgroup aggregation");
					} 
					
					long cross_end = System.currentTimeMillis();
					double crosstime = (cross_end-cross_start)/1e0;
					System.out.println("cross elapsed " + ridTuple + " " + crosstime);
				}
			} else {
				for (int jr : ridorder) {
					Pair<Object, Integer> fungsi = aggpernode.get(jr);
					if (crossvalid.size() > 1) {
						for (int u=0 ; u < crossvalid.size()-1 ; u++) {
							DoubleBox<Integer> subplane = crossvalid.get(u);
							for (int w=u+1; w < crossvalid.size(); w++) {
								DoubleBox<Integer> dubplane = crossvalid.get(w);
								if (dubplane.maxX.getFirst().equals(subplane.maxX.getFirst()) && 
										dubplane.maxX.getSecond().equals(subplane.maxX.getSecond()) ) {
									crossvalid.remove(dubplane);
								}
							}
						}
					}
					List<DoubleBox<Integer>> repeatupdate = new ArrayList<>();
					for (int r=0; r<crossvalid.size(); r++) {
						DoubleBox<Integer> subplane = crossvalid.get(r);
						//cross validation
						List<DoubleBox<Integer>> crossmulti = 
								Extrawork.crossValidation(subplane, comidValue.get(fungsi.getSecond()), 
										baseClass.get(fungsi.getSecond()), 
										fungsi, ridValue.get(jr), jr, rangeId, wholeIndex);
						repeatupdate.addAll(crossmulti);
					}
					if (repeatupdate.isEmpty()) {
						System.out.println("cross aggregate no valid");
						crossvalid.clear();
						break outer;
					} else {
						crossvalid.clear();
						crossvalid = repeatupdate;
						System.out.println("Updated bounding box " + crossvalid.size());
						//for (DoubleBox<Integer> up : crossvalid) {
						//	System.out.println("Updated bounding box " + 
						//	["+up.minX.getFirst()+" "+up.minX.getSecond()+"]"+ 
						//			"["+up.maxX.getFirst()+" "+up.maxX.getSecond()+"]");
						//} 
					}
				}
			}
			long single_end = System.currentTimeMillis();
			double time = (single_end-single_start)/1e0;
			System.out.println("time elapsed " + ridTuple + " " + time);
			} 
			if (!crossvalid.isEmpty()) {
				for (DoubleBox<Integer> cross : crossvalid) {
					System.out.println("store into wheresatubiru "
							+ "["+cross.minX.getFirst()+" "+cross.minX.getSecond()+"]"+ 
							"["+cross.maxX.getFirst()+" "+cross.maxX.getSecond()+"]");
					
					Pair<Integer, DoubleBox<Integer>> whereSatuMerah = new Pair<>(ib, cross);
					whereSatuBiru.put(whereSatuMerah, functioname);
				}
			} else {
				System.out.println("not any box returns");
			}
		}
	}
	}
	return whereSatuBiru;
}
*/
/**
private static void lagilagi(List<DoubleBox<Integer>> duajadisatu,
		List<List<DoubleBox<Integer>>> crossgroup, int hl, int target, List<DoubleBox<Integer>> crossvalid) 
{
	//List<DoubleBox<Integer>> pulang = new ArrayList<>();
	List<List<DoubleBox<Integer>>> bukadua = new ArrayList<>();
	bukadua.add(duajadisatu);
	bukadua.add(crossgroup.get(hl));
	List<List<DoubleBox<Integer>>> dotproduct = cartesianProduct(bukadua);
	List<DoubleBox<Integer>> gabung = crossaggregate(dotproduct);
	if (!gabung.isEmpty() && hl < target-1) {
		hl++;
		lagilagi(gabung, crossgroup, hl, target, crossvalid);
	} else if (hl == target-1){
		crossvalid.addAll(gabung);
	}
}
*/
/**
private static <T extends Comparable<T>> Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> 
startOneDimen (
		List<Integer> nongroupdimensi, 
		Multimap<List<String>, DimenIndex<T>> ridTupleIndex,
		Map<Integer, String> baseClass, 
		List<Map<Integer, Pair<Object, Integer>>> whichaggre, List<Integer> ridorder) 
{
	
	Map<Pair<Integer, DoubleBox<Integer>>, List<Pair<Object, Integer>>> whereSatuBiru = new HashMap<>();
	System.out.println("1d-Filter on dimensi " + nongroupdimensi + " for group " + ridTupleIndex.keySet());
	for (int ib : nongroupdimensi) {
		List<T> wholeIndex = new ArrayList<>();
		
		for (Map<Integer, Pair<Object, Integer>> aggpernode : whichaggre) {
			List<Pair<Object, Integer>> functioname = new ArrayList<>(aggpernode.values());
			List<List<DoubleBox<Integer>>> crossgroup = new ArrayList<>();
			outer:
			for (List<String> ridTuple : ridTupleIndex.keySet()) {
				System.out.println(ib + " " + functioname + " " + ridTuple);
				Map<Integer, Map<Integer, T>> comidValue = new HashMap<>();
				ListMultimap<Integer, Integer> rangeId = ArrayListMultimap.create();
				//Map<Integer, List<Integer>> rangeId = new HashMap<>();
				Map<Integer, T> ridValue = new HashMap<>();
				for (DimenIndex<T> cv : ridTupleIndex.get(ridTuple)) {
					comidValue.put(cv.index, cv.indexMatch);
					if (cv.index == ib) {
						rangeId = cv.newBuildD;
						wholeIndex = cv.letT;
						ridValue = cv.ridValue;
					}
				}

				Multimap<Integer, DoubleBox<Integer>> forall = ArrayListMultimap.create();
				for (int jr : ridorder) {
					Pair<Object, Integer> fungsi = aggpernode.get(jr);
					Multimap<Integer, DoubleBox<Integer>> forone = 
							perusalAggregate(rangeId, fungsi, ridValue.get(jr), 
									comidValue.get(fungsi.getSecond()), baseClass.get(fungsi.getSecond()), 
									wholeIndex, fungsi.getSecond(), jr);
					if (forone.isEmpty()) {
						break outer;
					} else {
						List<DoubleBox<Integer>> updateValid = new ArrayList<>(forone.get(jr)); 
						System.out.println("Created bounding box " + updateValid.size());
						//for (DoubleBox<Integer> up : updateValid) {
						//	System.out.println("Created bounding box " +
						//	"["+up.minX.getFirst()+" "+up.minX.getSecond()+"]"+ 
						//		"["+up.maxX.getFirst()+" "+up.maxX.getSecond()+"]"); }
						//
						forall.putAll(forone);
					}
				}
				
				List<List<DoubleBox<Integer>>> overallScale = new ArrayList<>();
				for (int k : forall.keySet()) {
					List<DoubleBox<Integer>> pairFilter = (List<DoubleBox<Integer>>) forall.get(k);
					overallScale.add(pairFilter);
				}
				List<List<DoubleBox<Integer>>> everyScale = cartesianProduct(overallScale);
				
				List<DoubleBox<Integer>> ui = crossaggregate(everyScale);
				if (!ui.isEmpty()) {
					crossgroup.add(ui);
				} else {
					break outer;
				}
			}
			
			if (crossgroup.size() != ridTupleIndex.keySet().size()) {
				System.out.println(" One Dimension failed lah crossgroup " + crossgroup.size() 
				+ " suppose " + ridTupleIndex.keySet().size());
			} else {
				List<List<DoubleBox<Integer>>> combinegroup = cartesianProduct(crossgroup);	
				
				List<DoubleBox<Integer>> actuallycrossgroup = crossaggregate(combinegroup);		
				if (!actuallycrossgroup.isEmpty()) {
					for (DoubleBox<Integer> where : actuallycrossgroup) {
						Pair<Integer, DoubleBox<Integer>> whereSatuMerah = new Pair<>(ib, where);
						whereSatuBiru.put(whereSatuMerah, functioname);
					}
				} else {
					System.out.println(" One Dimension failed becos fail crossgroup aggregation");
				}
			}
		}
	}
	return whereSatuBiru;
}
*/
/**
private static List<DoubleBox<Integer>> crossaggregate (List<List<DoubleBox<Integer>>> oneScale) {
	List<DoubleBox<Integer>> ui = new ArrayList<>();
	for (List<DoubleBox<Integer>> scale : oneScale) {
		List<Integer> mxleft = new ArrayList<>();
		List<Integer> mxright = new ArrayList<>();
		List<Integer> mnleft = new ArrayList<>();
		List<Integer> mnright = new ArrayList<>();
		for (DoubleBox<Integer> elacs : scale) {
			mxleft.add(elacs.maxX.getFirst());
			mxright.add(elacs.maxX.getSecond());
			mnleft.add(elacs.minX.getFirst());
			mnright.add(elacs.minX.getSecond());
		}
		if (Collections.max(mxleft) <= Collections.min(mxright) && 
				Collections.min(mnleft) <= Collections.max(mnright)) 
		{
			if (Collections.min(mnleft) >= Collections.max(mxleft) && 
					Collections.max(mnright) <= Collections.min(mxright)) 
			{
				Pair<Integer, Integer> mxintersect = 
						new Pair<Integer, Integer>(Collections.max(mxleft), Collections.min(mxright));
				Pair<Integer, Integer> mnunion = 
						new Pair<Integer, Integer>(Collections.min(mnleft), Collections.max(mnright));
				DoubleBox<Integer> mxmn = new DoubleBox<Integer>(mnunion, mxintersect);
				ui.add(mxmn);
			}
		}	
	}
	return ui;
}
*/