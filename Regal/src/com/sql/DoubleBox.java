package com.sql;

class DoubleBox<T> {
	Pair<T, T> minX;
	Pair<T, T> maxX;
	
	DoubleBox(Pair<T, T> a, Pair<T, T> b) {
		this.minX = a;
		this.maxX = b;
	}
	
	DoubleBox() {
		this.minX = new Pair<>();
		this.maxX = new Pair<>();
	}
	
	public String toString() {
		return "(" + minX + ", " + maxX + ")";
	}
}
