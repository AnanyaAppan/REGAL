package com.sql;

class TwoD {
int x, y;
	
	TwoD(int a, int b) {
		x = a;
		y = b;
	}
	
	int getX() {
		return x;
	}
	
	int getY() {
		return y;
	}
	
	void showType() {
		System.out.println("array.x " + this.x + " array.y " + this.y);
	}
	
	@Override
	public boolean equals (Object obj) {
		if (obj instanceof TwoD) {
			if (x == ((TwoD) obj).x && y == ((TwoD)obj).y )
				return true;
		}
		return false;
	}
}
