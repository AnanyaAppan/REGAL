package com.sql;

import java.util.ArrayList;
import java.util.List;

public class CombNode {
	ArrayList<CombNode> parents; 
	ArrayList<CombNode> children;
	ArrayList<String> attr;
	ArrayList<List<String>> inferFrequency;
	int nodeID;
	boolean isValid; // the node is valid for further verification
	boolean isKeyness;
	boolean isContainment;
	boolean isPruned;
	boolean isVisited;
	boolean isFrequency;
	
	CombNode(ArrayList<String> attr, int nodeID) {
		this.attr = attr;
		this.parents = new ArrayList<CombNode> ();
		this.children = new ArrayList<CombNode> ();
		this.isValid = true;
		this.nodeID = nodeID;
		this.isKeyness = false; //false for bottom-up keyness
		this.isContainment = true;
		this.isFrequency = false;
		this.inferFrequency = new ArrayList<>();
		//this.isContainment = false; //first set false, once true cannot change
		this.isPruned = false;
		this.isVisited = false;
		// TODO Auto-generated constructor stub
	}
	public void setInferFrequency(ArrayList<List<String>> inferFrequency) {
		this.inferFrequency = inferFrequency;
	}
	
	ArrayList<List<String>> getInferFrequency(){
		return inferFrequency;
	}
	
	ArrayList<String> getAttr(){
		
		return attr;
	}
	
	ArrayList<CombNode> getChildren(){
		
		return children;
	}
	
	ArrayList<CombNode> getParents(){
		
		return parents;
	}
	
	boolean hasChildren(){

		return children != null;
	}
	
	boolean hasParents(){

		return parents != null;
	}
	
	public String toString (){
		String out = "(";
		for (String s : attr){
			out += s + " ";
		}
		out += ")";
		return out;
	}
	
	public String parentsToString (){
		if (parents.size() == 0) return "";
		
		String str = "Parents ";
		for (CombNode cn : parents){
			str += cn.toString() + ", ";
		}

		return str;
	}
	
	public String childrenToString (){
		if (children.size() == 0) return "";
		
		String str = "Children ";
		for (CombNode cn : children){
			str += cn.toString() + ", ";
		}

		return str;
	}
	
	
	boolean contains(CombNode cn){
		return attr.containsAll(cn.getAttr());
	}
}
