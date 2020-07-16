package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.BiMap;
import com.google.common.collect.Multimap;

public class BuildLattice {
	public static <T> Map<Map<String, Integer>, List<CombNode>> buildLattice (
			Multimap<Integer, Integer> resTobase, BiMap<Integer, String> resultName,
			Map<Integer, List<T>> resultColumn, List<List<Integer>> rootProduct,
			Map<Integer, Map<String, List<Integer>>> baseFreq, int cardinality,
			Map<Integer, List<String>> baseRowStore, Map<List<Integer>, List<Integer>> basePartialRow,
			Map<Integer, List<String>> rowStore, Map<List<Integer>, List<Integer>> partialRow )
			// Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable,
			// Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeBase
	{
		// building lattice
		List<Integer> bimap = new ArrayList<Integer>(resTobase.keySet());
		Collections.sort(bimap);
		ArrayList<String> rootString = new ArrayList<>();
		for (int i : bimap) {
			rootString.add(resultName.get(i));
		}
		CombNode cn = new CombNode(rootString, 0);
		CombTree ct = new CombTree(cn, 0);
		CombNode akar = ct.root;
		System.out.println("lattice_root " + akar.attr);
		Map<Integer, CombNode> allNodes = new HashMap<Integer, CombNode>();
		allNodes.put(akar.nodeID, akar);
		printNodeRecursiveFromRoot(akar, allNodes);
		
		//bottomUpKeyness Algorithm
		BiMap<String, Integer> resultIndex = resultName.inverse();
		
		ArrayList<CombNode> leaves = ct.leaves;
		if (leaves.isEmpty()) {
			leaves.add(akar);
		}
		ArrayList<CombNode> topDownRoot = new ArrayList<>();
		topDownRoot.add(akar);
		
		long start_BU = System.currentTimeMillis();
		bottomUpKeyness(leaves, resultIndex, resultColumn, cardinality, rowStore); //latticeTable
		/////topDownKeyness(topDownRoot, resultIndex, resultColumn, cardinality, rowStore);
		long end_BU = System.currentTimeMillis();
		double diff_BU = (end_BU - start_BU)/1e0; 
		System.out.println( "BuildLattice.lattice_size " + allNodes.size() + " keyness_count " + BeginTree.count);
		
		for (int i : allNodes.keySet()) {
			if (allNodes.get(i).isKeyness == true)
				System.out.println("Keyness_" + allNodes.get(i).getAttr() );
		}
		
		//Discover groupby and return
		Map<Map<String, Integer>, List<CombNode>> groupbyAttr = new HashMap<>();
		//topDownContainment Algorithm
		//tupleLevel
		
		List<List<Integer>> rootCollect = new ArrayList<>();
		double total_TD = 0.0;
		for (List<Integer> root : rootProduct) {
			long start_TD = System.currentTimeMillis();
			//before the root starts going down, reset labels
			for (int i : allNodes.keySet()) {
				allNodes.get(i).isContainment = true;
				allNodes.get(i).isPruned = false;
			}
			rootCollect.add(root);
						
			if (rootCollect.size() > 1) {
				makingSecondTree(rootCollect.get(0), root, ct.leaves, allNodes, groupbyAttr);	
			} 
			 
			Map<String, Integer> resultTobas = new HashMap<>();
			Multimap<Integer, String> baseTores = ArrayListMultimap.create();
			for (int i = 0; i < root.size(); i++) {
				resultTobas.put(rootString.get(i), root.get(i));
				baseTores.put(root.get(i), rootString.get(i));
			}
			for (int i : baseTores.keySet()) {
				if (baseTores.get(i).size() > 1) {
					//compare result columns that share same base attribute
					//do pruning 
					List<String> prune = new ArrayList<>(baseTores.get(i));
					pruningSimilar(prune, resultIndex, resultColumn, allNodes); 
				}
			}
		
			ArrayList<String> resultRoot = akar.getAttr();
			Set<List<String>> resultTuples = new HashSet<>();
			for (int i : rowStore.keySet()) {
				List<String> tuplet = new ArrayList<>();
				for (int j : bimap) {
					tuplet.add(rowStore.get(i).get(j-1));
				}
				resultTuples.add(tuplet);
			}
			// //Set<List<String>> resultTuples = latticeTable.get(bimap).keySet();
			
			/*******
			Set<CombNode> coconut = new LinkedHashSet<>();
			for (List<String> tuple : resultTuples) {
				akar.isVisited = false;
				resetAkarVisit(akar);
				List<CombNode> bfs_nodes = new ArrayList<>();
				bfs_nodes.add(akar);
				TopDown.scanBase(bfs_nodes, resultRoot, tuple, resultTobas, baseRowStore, coconut); // //latticeBase
				//TopDown.reverseBase(leaves, resultRoot, tuple, resultTobas, latticeBase, coconut);
			}
			*/
		
			akar.isVisited = false;
			resetAkarVisit(akar);
			List<CombNode> bfsNodes = new ArrayList<>();
			bfsNodes.add(akar);
			List<List<String>> multituple = new ArrayList<>(resultTuples);
			List<Integer> baseIn = new ArrayList<Integer>();
			for (String s : akar.getAttr()) {
				baseIn.add(resultTobas.get(s));
			}
			Collections.sort(baseIn);
			
			List<List<String>> totalWitness = new ArrayList<>();
			for (List<String> telput : multituple) {
				List<String> retuple = new ArrayList<>();
				for (int i=0; i < akar.attr.size(); i++) {
					String r = akar.attr.get(i);
					int y = resultTobas.get(r);
					int z = baseIn.indexOf(y);
					retuple.add(telput.get(z));
				}
				totalWitness.add(retuple);
			}
			Map<Integer, List<List<String>>> witnessList = new HashMap<>();
			WitnessTable.tablescan(bfsNodes, resultRoot, totalWitness, resultTobas, baseRowStore, witnessList);
			/////WitnessTable.bottomscan(leaves, resultRoot, totalWitness, resultTobas, baseRowStore);
			
			//System.out.println(coconut);
			///for (int i : allNodes.keySet()) {
			///	CombNode cono = allNodes.get(i);
			///	if (cono.isContainment==false) {
			///		System.out.println("Containment failed " + cono);
			///	}
			///}
			long end_TD = System.currentTimeMillis();
			double diff_TD = (end_TD - start_TD)/1e0; 
			total_TD += diff_TD;
			//System.out.println("Lattice_nodes " + BeginTree.nodecount + " diff_TD " + diff_TD); //@@@@experiment_1
			/**
			long start_BU = System.currentTimeMillis();
			bottomUpKeyness(leaves, resultIndex, resultColumn, cardinality, rowStore);
			/////topDownKeyness(topDownRoot, resultIndex, resultColumn, cardinality, rowStore);
			long end_BU = System.currentTimeMillis();
			System.out.println( "allNodes_total " + allNodes.size() + " key count " + BeginTree.count);
			double diff_BU = (end_BU - start_BU)/1e0; 
			System.out.println("Total_TD " + total_TD + " Diff_BU " + diff_BU ); 
			
			for (int i : allNodes.keySet()) {
				if (allNodes.get(i).isKeyness == true)
					System.out.println("BuildLattice " + allNodes.get(i).getAttr() + " true keyness");
			}
			*/
			List<CombNode> possibleCandidate = new ArrayList<>();
			for (int i : allNodes.keySet()) {
				CombNode cono = allNodes.get(i);
				if (cono.isContainment==true && cono.isKeyness==true && cono.isPruned==false) {
					possibleCandidate.add(cono);
				}
			}
			if (!possibleCandidate.isEmpty()) {
				//insertion sort for possible lattice nodes
				for (int j = 1; j < possibleCandidate.size(); j++) {
					int length = possibleCandidate.get(j).getAttr().size();
					CombNode rm = possibleCandidate.get(j);
					int i = j - 1;
					while (i >= 0 && possibleCandidate.get(i).getAttr().size() > length) {
						possibleCandidate.set(i+1, possibleCandidate.get(i));
						i = i - 1;
						possibleCandidate.set(i+1, rm);
						length = possibleCandidate.get(i+1).getAttr().size();
					}
				}
				groupbyAttr.put(resultTobas, possibleCandidate);
			}
			
		}
		System.out.println("Time_TD " + total_TD + " Time_BU " + diff_BU ); 
		return groupbyAttr;
	}

	private static void makingSecondTree(List<Integer> oldroot,
			List<Integer> newroot, ArrayList<CombNode> leaves, Map<Integer, CombNode> allNodes, 
			Map<Map<String, Integer>, List<CombNode>> groupbyAttr) {
		// TODO Auto-generated method stub
		List<Pair<Integer, Integer>> differnode = new ArrayList<>();
		for (int i = 0; i < oldroot.size(); i++) {
			if (oldroot.get(i) != newroot.get(i)) {
				Pair<Integer, Integer> diff = new Pair<Integer, Integer>(i, newroot.get(i));
				differnode.add(diff);
			}
		}
		List<Integer> leafNodeID = new ArrayList<>();
		for (CombNode leaf : leaves) {
			leafNodeID.add(leaf.nodeID);
		}
		Collections.sort(leafNodeID);
		Map<Integer, CombNode> secondTreeNodes = new HashMap<>();
		List<Integer> pruneID = new ArrayList<>();
		for (Pair<Integer, Integer> dn : differnode) {
			int noid = leafNodeID.get(dn.getFirst());
			pruneID.add(noid);
			CombNode newLeaf = allNodes.get(noid);
			secondTreeNodes.put(newLeaf.nodeID, newLeaf);
			buildSecondTree(newLeaf, secondTreeNodes);
		}
		
		Set<Integer> smallerTreeKeys = secondTreeNodes.keySet();
		Set<Integer> biggerTreeKeys = allNodes.keySet();
		Set<Integer> unwanted = new HashSet<>();
		for (Integer b : biggerTreeKeys) {
			if (!smallerTreeKeys.contains(b))
				unwanted.add(b);
		}
		
		for (int j : unwanted) {
			allNodes.get(j).isPruned = true;
		}
		
		for (int i : allNodes.keySet()) {
			if (allNodes.get(i).isKeyness == true && allNodes.get(i).isPruned == false)
				System.out.println("BuildLattice.secondTreebeforePropagate " + allNodes.get(i).getAttr() );
		}
		
		if (!groupbyAttr.isEmpty()) {
			propagateFirstTree(groupbyAttr, allNodes, pruneID, oldroot);
		}
		//return secondTreeNodes;
		
	}

	private static void propagateFirstTree(
			Map<Map<String, Integer>, List<CombNode>> groupbyAttr,
			Map<Integer, CombNode> allNodes, List<Integer> pruneID,
			List<Integer> oldroot) {
		// TODO Auto-generated method stub
		CombNode origin = allNodes.get(0);
		Map<String, Integer> first = new HashMap<>();
		for (int i=0; i<origin.attr.size(); i++) {
			first.put(origin.attr.get(i), oldroot.get(i));
		}
		if (groupbyAttr.containsKey(first)) {
			List<CombNode> success = groupbyAttr.get(first);
			//System.out.println("BuildLattice.propagateFirstTree " + success);
			List<List<String>> successnodes = new ArrayList<>();
			int minlen = success.get(0).attr.size();
			for (CombNode node : success) {
				//System.out.println("node " + node.attr + " key " + node.isKeyness + " cont " + node.isContainment);
				successnodes.add(node.attr);
				if (node.attr.size() < minlen) {
					minlen = node.attr.size();
				}	
			}
			ArrayList<Integer> startbottom = new ArrayList<>(allNodes.keySet());
			Collections.reverse(startbottom);
			for (int j : startbottom) {
				if (allNodes.get(j).isPruned == false && allNodes.get(j).isKeyness == true && 
						allNodes.get(j).attr.size() >= 3) {
					//System.out.println("BuildLattice.propagate " + allNodes.get(j));
					List<String> temp = new ArrayList<>(allNodes.get(j).attr);
					List<String> left = new ArrayList<>();
					for (int k : pruneID) {
						left.addAll(allNodes.get(k).attr);
					}
					temp.removeAll(left);
					//System.out.println("BuildLattice.propagate_temp " + temp);
 					if (!successnodes.contains(temp) && temp.size() >= minlen) {
						//mark allNodes.get(j)
 						//System.out.println("BuildLattice.propagate_temp " + temp);
						allNodes.get(j).isPruned = true; allNodes.get(j).isContainment = false;
						prunedAbove(allNodes.get(j));
					}
				}
			}
		}
	}

	private static void prunedAbove(CombNode combNode) {
		// TODO Auto-generated method stub
		if (combNode.nodeID != 0) {
		List<CombNode> getParent = combNode.getParents();
		for (CombNode parent : getParent) {
			if (parent.isPruned == false) {
				parent.isPruned = true; parent.isContainment = false;
			}
		}
		}
	}

	private static void buildSecondTree(CombNode newLeaf,
			Map<Integer, CombNode> secondTreeNodes) {
		// TODO Auto-generated method stub
		if (newLeaf.hasParents()) {
			List<CombNode> m = newLeaf.getParents();
			for (CombNode n : m) {
				if (!secondTreeNodes.containsKey(n.nodeID))
					secondTreeNodes.put(n.nodeID, n);
				buildSecondTree(n, secondTreeNodes);
			}
		}
	}

	private static void resetAkarVisit(CombNode akar) {
		// TODO Auto-generated method stub
		if (akar.hasChildren()){
			List<CombNode> children = akar.getChildren();
			for (CombNode c : children) {
				if (c.isVisited == true)
					c.isVisited = false;
				resetAkarVisit(c);
			}
		}
	}
	
	/*__________________________________________________alternative______________________________________________
	private static List<List<Integer>> tupleContainment(
			ArrayList<String> resultRoot, List<String> tuple, Map<String, Integer> resultTobas,
			Map<Integer, Map<String, List<Integer>>> baseFreq) {
		
		List<List<Integer>> allIndices;
		allIndices= new ArrayList<>();
		for (int i=0; i<tuple.size(); i++) {
			//System.out.println(tuple.get(i));
			String s = resultRoot.get(i);
			int t = resultTobas.get(s);
			List<Integer> indices = baseFreq.get(t).get(tuple.get(i).toString());
			allIndices.add(indices);
		}
		return allIndices;
	}*/
	
	private static void printNodeRecursiveFromRoot(CombNode akar,
			Map<Integer, CombNode> allNodes) {
		// TODO Auto-generated method stub
		if (akar.hasChildren()){
			List<CombNode> children = akar.getChildren();
			for (CombNode c : children) {
				if (!allNodes.containsKey(c.nodeID))
					allNodes.put(c.nodeID, c);
				printNodeRecursiveFromRoot(c, allNodes);
			}
		} 
	}
	
	private static <T> void topDownKeyness(ArrayList<CombNode> topDown, BiMap<String, Integer> resultIndex, 
			Map<Integer, List<T>> resultColumn, int cardinality, 
			Map<Integer, List<String>> rowStore) 
	{	//all tree nodes are in keyness true now;
		Set<CombNode> lowlevelNodes = new HashSet<>();
		for (CombNode akar : topDown) {
			BeginTree.count++;
			if (akar.isKeyness == true && akar.isContainment == true) {
				List<Integer> nodeID = new ArrayList<>();
				for (String s : akar.getAttr()) {
					nodeID.add(resultIndex.get(s));
				}
				//Set<List<String>> tupleUnique = latticeTable.get(nodeID).keySet();
				long s = System.currentTimeMillis();
				Set<List<String>> tupleUnique = new HashSet<>();
				for (int i : rowStore.keySet()) {
					List<String> tuplet = new ArrayList<>();
					for (int j : nodeID) {
						tuplet.add(rowStore.get(i).get(j-1));
					}
					tupleUnique.add(tuplet);
				}
				long e = System.currentTimeMillis();
				double d = (e-s)/1e0;
				//System.out.println("Time to retrieve for KEYNESS " + d + " ms");
				System.out.println("Time for TDKEYNESS " + d + " ms " + akar.attr);
				if (cardinality > tupleUnique.size()) {
					akar.isKeyness = false;
					keynessFalse(akar);
				} 
			}
		}
		for (CombNode akar : topDown) {
			int saiz = akar.attr.size()-1;
			ArrayList<CombNode> interNodes = akar.getChildren();
			for (CombNode g : interNodes) {
				if (g.attr.size() == saiz && g.isKeyness == true && g.isContainment == true) {
					lowlevelNodes.add(g);
				}
			}
		}
		ArrayList<CombNode> lowerNodes = new ArrayList<>(lowlevelNodes);
		if (!lowerNodes.isEmpty())
			topDownKeyness(lowerNodes, resultIndex, resultColumn, cardinality, rowStore);
	}
	
	private static void keynessFalse(CombNode f) {
		// TODO Auto-generated method stub
		if (f.hasChildren()) {
			ArrayList<CombNode> interNodes = f.getChildren();
			for (CombNode in : interNodes) {
				if (in.isKeyness == true) {
					in.isKeyness = false;
					keynessFalse(in);
				}
			}	
		}
	}
	
	private static <T> void bottomUpKeyness(ArrayList<CombNode> leaves, BiMap<String, Integer> resultIndex, 
			Map<Integer, List<T>> resultColumn, int cardinality, 
			Map<Integer, List<String>> rowStore) 
			//Map<List<Integer>, List<Integer>> partialRow
			//Map<ArrayList<Integer>, Map<List<String>, Multimap<Integer, T>>> latticeTable
	{	//all tree nodes are in keyness false now;
		for (CombNode f : leaves) {		
			if (f.isKeyness != true) { // && f.isContainment == true) {
				//System.out.println("Go Keyness Checking " + f.attr);
				BeginTree.count++;
				List<Integer> nodeID = new ArrayList<>();
				for (String s : f.getAttr()) {
					nodeID.add(resultIndex.get(s));
				}
				// //Set<List<String>> tupleUnique = latticeTable.get(nodeID).keySet();
				long s = System.currentTimeMillis();
				Set<List<String>> tupleUnique = new HashSet<>();
				for (int i : rowStore.keySet()) {
					List<String> tuplet = new ArrayList<>();
					for (int j : nodeID) {
						tuplet.add(rowStore.get(i).get(j-1));
					}
					tupleUnique.add(tuplet);
				}
				long e = System.currentTimeMillis();
				double d = (e-s)/1e0;
				// System.out.println("Time for BUKEYNESS " + d + " ms " + f.attr); //@@@@experiment_1
				if (cardinality > tupleUnique.size()) {
					f.isKeyness = false;
				} else {
					f.isKeyness = true; 
					keynessTrueness(f);
				}
			}
		}
		Set<CombNode> uplevelNodes = new HashSet<>();
		for (CombNode f : leaves) {
			int saiz = f.attr.size() + 1;
			ArrayList<CombNode> interNodes = f.getParents();
			for (CombNode g : interNodes) {
				if (g.getAttr().size() == saiz && g.isKeyness == false) { // && f.isContainment == true) {
					uplevelNodes.add(g);
				}
			}	
		}
		ArrayList<CombNode> upperNodes = new ArrayList<>(uplevelNodes);
		if (!upperNodes.isEmpty())
			bottomUpKeyness(upperNodes, resultIndex, resultColumn, cardinality, rowStore);
	}

	private static void keynessTrueness(CombNode f) {
		// TODO Auto-generated method stub
		if (f.hasParents() && f.nodeID > 0) {
			ArrayList<CombNode> interNodes = f.getParents();
			for (CombNode in : interNodes) {
				if (in.isKeyness == false) {
					in.isKeyness = true;
					keynessTrueness(in);
				}
			}	
		}
	}
	
	private static <T> void pruningSimilar(List<String> prune, 
			BiMap<String, Integer> resultIndex, Map<Integer, List<T>> resultColumn, Map<Integer, CombNode> allNodes) {
		List<List<T>> similarColumns = new ArrayList<>();
		for (String s : prune) {
			List<T> leafItems = resultColumn.get(resultIndex.get(s));
			similarColumns.add(leafItems);
		}
		List<String> similarNode = new ArrayList<String>();
		//when the lists are more than two to be compared
		for (int i=0; i<similarColumns.size()-1; i++) {
			similarNode.add(prune.get(i));
			for (int j=i+1; j<similarColumns.size(); j++){
				if(!similarColumns.get(i).equals(similarColumns.get(j))) {
					
					similarNode.add(prune.get(j));
					//System.out.println("BuildLattice.pruning " + similarNode);
					//eliminateSimilar(similarNode,  akar);//______allNodes,
					eliminateSimilar(similarNode, allNodes);
				}
			}
		}
	}

	private static void eliminateSimilar(List<String> similarNode,
			 Map<Integer, CombNode> allNodes) {
		// TODO Auto-generated method stub_______________________Map<Integer, CombNode> allNodes,
		for (int k : allNodes.keySet()) {
			//set flag to false if the repeated base name actually not true for query columns
			if (allNodes.get(k).getAttr().containsAll(similarNode) && allNodes.get(k).isPruned == false)
				allNodes.get(k).isPruned = true;
		}
		/*
		akar.isPruned = true;
		List<CombNode> akarbelow = akar.getChildren();
		for (CombNode ab : akarbelow) {
			if (ab.getAttr().containsAll(similarNode) && ab.isPruned == false){
				ab.isPruned = true;
				eliminateSimilar(similarNode, ab); //_________________________ allNodes,
			}
		}*/	
	}
}
