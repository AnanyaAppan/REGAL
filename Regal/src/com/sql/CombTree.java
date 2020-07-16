package com.sql;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;

public class CombTree {
	int treeID; // start from zero
	CombNode root; // root of the tree
	ArrayList <CombNode> leaves; //leaves of the tree
	ArrayList <CombNode> treeArray; // store tree in array, not implemented
	
	//constructor
	CombTree(CombNode root, int treeID){ 
		this.treeID = treeID;
		this.root = root;
		init(root);
	}
	
	//tree initialization
	private void init(CombNode node){
		
		//get all combs
		ArrayList <ArrayList<String>> allComb = comb(new ArrayList<String>(), node.getAttr(), new ArrayList <ArrayList<String>> ());
		Collections.sort(allComb, new Comparator<ArrayList<String>>() {
		    public int compare(ArrayList<String> a, ArrayList<String> b) {
		        return b.size() - a.size();
		    }
		});
		//System.out.println("Before: " + allComb);
		allComb.remove(0);
		//System.out.println("After: " + allComb);
		/*
		Before: [[A, B, MC, MD], [A, B, MC], [A, B, MD], [A, MC, MD], [B, MC, MD], [A, B], [A, MC], [A, MD], [B, MC], [B, MD], [MC, MD], [A], [B], [MC], [MD]]
		After: [[A, B, MC], [A, B, MD], [A, MC, MD], [B, MC, MD], [A, B], [A, MC], [A, MD], [B, MC], [B, MD], [MC, MD], [A], [B], [MC], [MD]]
		*/
		
		//construct the tree
		ArrayList<CombNode> parentList = new ArrayList<CombNode>();
		ArrayList<CombNode> tmpParent = new ArrayList<CombNode>();
		leaves = new ArrayList<CombNode>();
		parentList.add(root);
		ArrayList<Integer> l = new ArrayList<Integer> ();
		for (CombNode seed : parentList)
			l.add(parentList.indexOf(seed));
		//System.out.println("parentListforRoot: " +l);
		int nodeID = 1;
		//System.out.println("NodeSize: " +node.getAttr().size());	//size=4
		for(int i = node.getAttr().size(); i> 0; i--){
			for (int j =0; j < allComb.size(); j++){
				if (allComb.get(j).size() == i-1){
					CombNode curr = add(new CombNode (allComb.get(j), nodeID), parentList);
					nodeID++;
					tmpParent.add(curr);
					if(i-1 == 1) leaves.add(curr);
				}else if (allComb.get(j).size() < i-1){
					parentList = new ArrayList<CombNode> (tmpParent);
					tmpParent.clear();
					break;
				}
			}
		}
	}
	
	//add a node to the tree
	private CombNode add(CombNode curr, ArrayList<CombNode> pl){
		for (CombNode cn : pl){
			if (cn.contains(curr)){ 
				curr.parents.add(cn); 
				cn.children.add(curr);
			}
		}
		return curr;
	}
	
	//get combs
	private ArrayList <ArrayList<String>> comb(ArrayList<String> prefix, ArrayList<String> s, ArrayList <ArrayList<String>> allComb) {
        if (s.size() > 0) {
        	allComb.add(addString(prefix, s.get(0)));
        	//System.out.println("prefix: " + prefix);
        	//System.out.println("s: " + s);
        	//System.out.println("allComb: " + allComb);
        	comb(addString(prefix, s.get(0)), rmFirst(s), allComb);
        	comb(prefix, rmFirst(s), allComb);
        }
        return allComb;
    }
	
	private ArrayList<String> addString(ArrayList<String> a, String b){
		ArrayList<String> tmp  = new ArrayList<String>(a);
		tmp.add(b);
		return tmp;
	}
	
	private ArrayList<String> rmFirst(ArrayList<String> a){
		ArrayList<String> tmp  = new ArrayList<String>(a);
		tmp.remove(0);
		return tmp;
	}
	
	//iterate tree from root
	public void getTreeFromRoot(){
		
		ArrayList <CombNode> src = new ArrayList <CombNode> ();
		src.add(root);
		printLevel(src);
		ArrayList <CombNode> tmp = getChildrenLevel(src);
		while (tmp.size() > 0){
			printLevel(tmp);
			tmp = getChildrenLevel(tmp);
		}
	}
	
	public ArrayList<CombNode> getChildrenLevel(ArrayList <CombNode> list){
		HashSet<CombNode> hs = new HashSet<CombNode> ();
		for(CombNode cn : list){
			if (cn.hasChildren()) hs.addAll(cn.getChildren());
		}
		return new ArrayList<CombNode>(hs);		

	}
	
	//iterate tree from leaves
	public void getTreeFromLeaves(){
		
		printLevel(leaves);
		//printColumnNo(leaves);
		ArrayList <CombNode> tmp = getParentLevel(leaves);
		while (tmp.size() > 0){
			printLevel(tmp);
			//printColumnNo(tmp);
			tmp = getParentLevel(tmp);
		}
	}
	
	public ArrayList<CombNode> getParentLevel(ArrayList <CombNode> list){
		HashSet<CombNode> hs = new HashSet<CombNode> ();
		for(CombNode cn : list){
			if (cn.hasParents()) hs.addAll(cn.getParents());
		}
		return new ArrayList<CombNode>(hs);
	}
	
	private void printLevel(ArrayList<CombNode> l){
		for(CombNode cn : l){
			System.out.println(cn.toString() + "\t"+ cn.parentsToString() + "\t"+ cn.childrenToString());
		}
	}
}
