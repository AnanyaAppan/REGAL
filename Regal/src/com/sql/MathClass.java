package com.sql;

class MathClass<T> {
	T min;
	T max;
	T totpos;
	T totneg;
	
	MathClass() {
		
	}
	
	MathClass(T min, T max, T totpos, T totneg) {
		this.min = min;
		this.max = max;
		this.totpos = totpos;
		this.totneg = totneg;
	}
	
	public T getMin() {
		return min;
	}
	public T getMax() {
		return max;
	}
	public T totpos() {
		return totpos;
	}
	public T totneg() {
		return totneg;
	}
	
	public void setMin(T min) {
		this.min = min;
	}
	public void setMax(T max) {
		this.max = max;
	}
	public void setTotpos(T totpos) {
		this.totpos = totpos;
	}
	public void setTotneg(T totneg) {
		this.totneg = totneg;
	}
	public String toString() {
		return "(" + min + ", " + max + ", " + totpos + ", " + totneg + ")";
	}
}
