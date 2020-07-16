package com.sql;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;

public class DimenIndex<T> {
	int index;
	List<T> letT;
	Map<Integer, T> ridValue;
	ListMultimap<Integer, Integer> newBuildD;
	Map<Integer, T> indexMatch;

	DimenIndex(int index) {
		this.index = index;
		this.letT = new ArrayList<>();
		this.ridValue = new HashMap<>();
		this.newBuildD = ArrayListMultimap.create();
		this.indexMatch = new HashMap<>();
	}
	
	public void setLetT (List<T> letT) {
		this.letT = letT;
	}
	
	List<T> getLetT () {
		return letT;
	}
}
