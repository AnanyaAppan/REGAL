package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import com.google.common.collect.Multimap;

public class TopDown {
	public static <T> void reverseBase(List<CombNode> leafnodes, ArrayList<String> resultRoot, List<String> tuple, Map<String,Integer> resultTobas,
			Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>>latticeBase, Set<CombNode> coconut) {
		
		Set<CombNode> upperNodes = new LinkedHashSet<>();
		for (CombNode node : leafnodes) {
			int saiz = node.attr.size()+1;
			if (node.getAttr().size() == 1) {
				node.isContainment = true;
				List<CombNode> naikatas = node.getParents();
				for (CombNode k : naikatas) {
					//System.out.println("oneOfParent " + k.getAttr());
					if (k.attr.size() == saiz) {
						upperNodes.add(k);
					}
				}
			} 
			else if (node.isPruned==false && node.isContainment==true && node.isVisited==false && node.isKeyness==true
				&& node.getAttr().size() > 1 ) {
				coconut.add(node);
				BeginTree.nodecount++;
				List<String> testTuple = getTestTuple(node.getAttr(), resultRoot, tuple);
				List<Integer> baseIn = new ArrayList<Integer>();
				for (String s : node.getAttr()) {
					baseIn.add(resultTobas.get(s));
				}
				Collections.sort(baseIn);
			
				List<String> retuple = new ArrayList<>();
				for (int i=0; i<node.attr.size(); i++) {
					String r = node.attr.get(i);
					int y = resultTobas.get(r);
					int z = baseIn.indexOf(y);
					retuple.add(testTuple.get(z));
				}
				Set<List<String>> baseTuples = latticeBase.get(baseIn).keySet();
				if (baseTuples.contains(retuple)) {
					node.isContainment = true; node.isVisited = true;
					List<CombNode> naikatas = node.getParents();
					for (CombNode k : naikatas) {
						if (k.attr.size() == saiz) {
							upperNodes.add(k);
						}	
					}
				} else {
					node.isContainment = false; node.isVisited = true;
					changeStatus(node);
				}
			} else if (node.isPruned == true || node.isContainment == false || node.isVisited == true || node.isKeyness == false
					) {
				List<CombNode> postRoot = node.getParents();
				for (CombNode p : postRoot) {
					if (p.attr.size() == saiz) {
						upperNodes.add(p);
					}
				}
			}
		}
		//System.out.println( "upyou " + upperNodes);
		List<CombNode> uplevel = new ArrayList<>(upperNodes);
		if (!uplevel.isEmpty())
			reverseBase(uplevel, resultRoot, tuple, resultTobas, latticeBase, coconut);
	}	

	private static void changeStatus(CombNode node) {
		List<CombNode> naikatas = node.getParents();
		for (CombNode k : naikatas) {
			k.isPruned = true; k.isVisited = true; k.isContainment = false;
			changeStatus(k);
		}	
	}

	public static <T> void scanBase(List<CombNode> bfs_akar, ArrayList<String> resultRoot, List<String> tuple, 
			Map<String,Integer> resultTobas,
			Map<Integer, List<String>> baseRowStore,
			// //Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>>latticeBase, 
			Set<CombNode> coconut) 
	{		
		Set<CombNode> next_bfs = new LinkedHashSet<>();
		for (CombNode akar : bfs_akar) {
			int saiz = akar.attr.size()-1;
			if (akar.getAttr().size() == 1)
				akar.isContainment = true;
			if (akar.isPruned==false && akar.isContainment==true && akar.isVisited==false && akar.isKeyness==true
					&& akar.getAttr().size() > 1) {
				coconut.add(akar);
				BeginTree.nodecount++;
				List<String> testTuple = getTestTuple(akar.getAttr(), resultRoot, tuple);
				List<Integer> baseIn = new ArrayList<Integer>();
				for (String s : akar.getAttr()) {
					baseIn.add(resultTobas.get(s));
				}
				Collections.sort(baseIn);
			
				List<String> retuple = new ArrayList<>();
				for (int i=0; i < akar.attr.size(); i++) {
					String r = akar.attr.get(i);
					int y = resultTobas.get(r);
					int z = baseIn.indexOf(y);
					retuple.add(testTuple.get(z));
				}
				//System.out.println(akar.attr + " " + baseIn + " == " + retuple);
				//System.out.println(latticeBase.get(baseIn));
				/*
				int l = 0;
				while (l < baseRowStore.size()) {
					List<String> retrieveStore = new ArrayList<>(baseRowStore.get(l));
					List<String> tuplet = new ArrayList<>();
					for (int j : baseIn) {
						tuplet.add(retrieveStore.get(j-1));
					}
					if (tuplet.size() != retuple.size()) {
						akar.isContainment = false;
						akar.isVisited = true;
					} else if (tuplet.contains(retuple)) {
						akar.isContainment = true; akar.isVisited = true;
					} else {
						akar.isContainment = false; akar.isVisited = true;
					}
					
				}
				*/
				long s = System.currentTimeMillis();
				Set<List<String>> baseTuples = new HashSet<>();
				for (int i : baseRowStore.keySet()) {
					List<String> tuplet = new ArrayList<>();
					for (int j : baseIn) {
						tuplet.add(baseRowStore.get(i).get(j-1));
					}
					baseTuples.add(tuplet);
				}
				long e = System.currentTimeMillis();
				double d = (e-s)/1e0;
				System.out.println("Time for CONTAINMENT " + d + " ms " + retuple + " " + akar.getAttr());
				if (baseTuples.isEmpty()) {
					akar.isContainment = false;
					akar.isVisited = true;
				} else {
					if (baseTuples.contains(retuple)) {
						akar.isContainment = true; akar.isVisited = true;
					} else {
						akar.isContainment = false; akar.isVisited = true;
					}	
				}
				/**
				if (latticeBase.get(baseIn)==null) {
					akar.isContainment = false; akar.isVisited = true;
				} else {
					Set<List<String>> baseTuples = latticeBase.get(baseIn).keySet();
					//System.out.println(baseTuples + " == " + retuple);
					if (baseTuples.contains(retuple)) {
						akar.isContainment = true; akar.isVisited = true;
					} else {
						akar.isContainment = false; akar.isVisited = true;
					}
				}
				*/
				List<CombNode> kekanak = akar.getChildren();
				for (CombNode k : kekanak) {
					if (k.attr.size() == saiz) {
						if (checkAbove(k, resultRoot, tuple, resultTobas, coconut))
							next_bfs.add(k);
					}
				}
			} else if (akar.isPruned == true || akar.isContainment == false || akar.isVisited == true) {
				List<CombNode> postRoot = akar.getChildren();
				for (CombNode p : postRoot) {
					if (p.attr.size() == saiz) {
						if (checkAbove(p, resultRoot, tuple, resultTobas, coconut))
							next_bfs.add(p);
					}
				}
			} 
		}
		List<CombNode> lowerNodes = new ArrayList<CombNode>(next_bfs);
		if (!lowerNodes.isEmpty())
			scanBase(lowerNodes, resultRoot, tuple, resultTobas, baseRowStore, coconut);
	}
	
	private static <T> boolean checkAbove(CombNode node, ArrayList<String> resultRoot, List<String> tuple, 
			Map<String,Integer> resultTobas,
			// //Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>>latticeBase,
			//Map<Integer, List<String>> baseRowStore,
			Set<CombNode> coconut) 
	{
		int count = 0;
		List<CombNode> ancestors = node.getParents();
		for (CombNode an : ancestors) {
			if (an.isContainment==true && an.isVisited==true && an.isPruned==false)
				count++;
		}
		if (count == 0) {
			//System.out.println("count " + count);
			//reverseBase(node, resultRoot, tuple, resultTobas, latticeBase, coconut);
			//scanBase(node, resultRoot, tuple, resultTobas, latticeBase, coconut);
			return true;
		} else {
			node.isContainment = true;
			node.isVisited = true;
			return false;
		}
	}
	
	public static List<String> getTestTuple(ArrayList<String> attr,
			ArrayList<String> resultRoot, List<String> tuple) {
		// when the node is not in the rootnode size
		List<String> testTuple = new ArrayList<>();
		if (attr.size() != resultRoot.size()) {
			for (int i=0; i<attr.size(); i++) {
				testTuple.add(tuple.get(resultRoot.indexOf(attr.get(i))));
			}
		} else testTuple = tuple;
		return testTuple;
	}
}
