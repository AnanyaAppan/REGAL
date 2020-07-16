package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class WitnessTable {

	public static void tablescan(List<CombNode> bfsNodes, ArrayList<String> resultRoot, 
			List<List<String>> totalWitness,
			Map<String, Integer> resultTobas, Map<Integer, List<String>> baseRowStore,
			Map<Integer, List<List<String>>> witnessList)
	{
		Set<CombNode> next_bfs = new LinkedHashSet<>();
		for (CombNode akar : bfsNodes) {
			int saiz = akar.attr.size()-1;
			if (akar.getAttr().size() == 1)
				akar.isContainment = true;
			if (akar.isPruned==false && akar.isContainment==true && akar.isVisited==false && akar.isKeyness==true
					&& akar.getAttr().size() > 1) {
				
				BeginTree.nodecount++;
				
				List<Integer> baseIn = new ArrayList<Integer>();
				for (String s : akar.getAttr()) {
					baseIn.add(resultTobas.get(s));
				}
				Collections.sort(baseIn);
				
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
				//System.out.println("Time for scanning one node " + d + " ms " + akar.getAttr());
				
				List<List<String>> subsetWitness = new ArrayList<>();
				if (baseTuples.isEmpty()) {
					akar.isContainment = false;
					akar.isVisited = true;
				} else {
					for (int i=0; i < totalWitness.size(); i++) {
						List<String> testTuple = 
								TopDown.getTestTuple(akar.getAttr(), resultRoot, totalWitness.get(i));
						if (!baseTuples.contains(testTuple)) {
							subsetWitness.add(totalWitness.get(i));
							//akar.isContainment = true; akar.isVisited = true;
						} 
						//else {
							//akar.isContainment = false; akar.isVisited = true;
						//}
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
				//System.out.println("SubsetWitness " + subsetWitness.size());
				if (!subsetWitness.isEmpty()) {
					//continue
					akar.isContainment = false; akar.isVisited = true;
					witnessList.put(akar.nodeID, subsetWitness);
					List<CombNode> des_akar = akar.getChildren();
					for (CombNode cnode : des_akar) {
						if (cnode.attr.size() == saiz) {
							if (lookUpTree(cnode))
								next_bfs.add(cnode);
						}
					}
				} else {
					akar.isContainment = true; akar.isVisited = true;
					List<CombNode> des_akar = akar.getChildren();
					labelContainmentTrue(des_akar);
				}
				System.out.println("Containment_" + d + " ms_" + akar.getAttr() + "_" + akar.isContainment); 
				//@@@@experiment_1
				
			} else if (akar.isPruned == true || akar.isContainment == false || akar.isVisited == true) {
				List<CombNode> postRoot = akar.getChildren();
				for (CombNode p : postRoot) {
					if (p.attr.size() == saiz) {
						if (lookUpTree(p))
							next_bfs.add(p);
					}
				}
			} 
		}
		List<CombNode> lowerNodes = new ArrayList<CombNode>(next_bfs);
		if (!lowerNodes.isEmpty()) {
			for (CombNode cnode : lowerNodes) {
				Set<List<String>> setWitness = new HashSet<>();
				List<CombNode> higherNodes = cnode.getParents();
				for (CombNode high : higherNodes) {
					if (witnessList.keySet().contains(high.nodeID)) {
						List<List<String>> retrieveWitness = witnessList.get(high.nodeID);
						setWitness.addAll(retrieveWitness);
					}
				}
				List<List<String>> updateWitness = new ArrayList<>(setWitness);
				if (!updateWitness.isEmpty())
					tablescan(lowerNodes, resultRoot, updateWitness, resultTobas, baseRowStore, witnessList);
			}
		}		
	}
	
	private static void labelContainmentTrue(List<CombNode> des_akar) {
		for (CombNode des : des_akar) {
			if (des.isContainment != true) {
				des.isContainment = true;
			}
			if (des.hasChildren()) {
				List<CombNode> get_des = des.getChildren();
				labelContainmentTrue(get_des);
			}
		}
		
	}
	
	private static void labelContainmentFalse(CombNode node) {
		List<CombNode> ancestors = node.getParents();
		if (!ancestors.isEmpty()) {
			for (CombNode any : ancestors) {
				any.isContainment = false;
				any.isVisited = true;
				labelContainmentFalse(any);
			}
		}
		
	}	
	
	private static <T> boolean lookUpTree(CombNode node) 
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

	public static void bottomscan(ArrayList<CombNode> leaves, ArrayList<String> resultRoot,
			List<List<String>> totalWitness, Map<String, Integer> resultTobas, 
			Map<Integer, List<String>> baseRowStore) 
	{
		Set<CombNode> upperNodes = new LinkedHashSet<>();
		for (CombNode node : leaves) {
			int saiz = node.attr.size()+1;
			if (node.getAttr().size() == 1) {
				node.isContainment = true;
				List<CombNode> naikatas = node.getParents();
				for (CombNode k : naikatas) {
					if (k.attr.size() == saiz) {
						upperNodes.add(k);
					}
				}
			} 
			else if (node.isPruned==false && node.isContainment==true && node.isVisited==false 
				//&& node.isKeyness==true
				&& node.getAttr().size() > 1 ) {
			
				BeginTree.nodecount++;
				
				List<Integer> baseIn = new ArrayList<Integer>();
				for (String s : node.getAttr()) {
					baseIn.add(resultTobas.get(s));
				}
				Collections.sort(baseIn);
				
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
				System.out.println("Time for CONTAINMENT " + d + " ms " + node.getAttr());
				
				if (baseTuples.isEmpty()) {
					node.isContainment = false;
					node.isVisited = true;
				} else {
					int k = 0;
					while (k < totalWitness.size() && node.isContainment == true) {
						List<String> testTuple = 
								TopDown.getTestTuple(node.getAttr(), resultRoot, totalWitness.get(k));
						if (!baseTuples.contains(testTuple)) {
							node.isContainment = false; //node.isVisited = true;
							labelContainmentFalse(node);
							//break;
						} else {
							
						}
						k++;
					}
					System.out.println("loop " + k + " " + node.getAttr() + " " + node.isContainment);
					node.isVisited = true;
				}
				
				/*if (node.isContainment == true) {
					//continue check above
					List<CombNode> postRoot = node.getParents();
					for (CombNode p : postRoot) {
						if (p.attr.size() == saiz) {
							upperNodes.add(p);
						}
					}
				} else {
					//failed above
					labelContainmentFalse(node);
				}*/
				
			} /*else if (node.isPruned == true || node.isContainment == false || node.isVisited == true || 
					node.isKeyness == false) 
			{
				List<CombNode> postRoot = node.getParents();
				for (CombNode p : postRoot) {
					if (p.attr.size() == saiz) {
						upperNodes.add(p);
					}
				}
			}*/
		}
		
		//Set<CombNode> uplevelNodes = new HashSet<>();
		for (CombNode f : leaves) {
			int saiz = f.attr.size() + 1;
			ArrayList<CombNode> interNodes = f.getParents();
			for (CombNode g : interNodes) {
				if (g.getAttr().size() == saiz && g.isContainment == true) {
					upperNodes.add(g);
				}
			}	
		}
		
		ArrayList<CombNode> uplevel = new ArrayList<>(upperNodes);
		if (!uplevel.isEmpty())
			bottomscan(uplevel, resultRoot, totalWitness, resultTobas, baseRowStore);		
	}

}
