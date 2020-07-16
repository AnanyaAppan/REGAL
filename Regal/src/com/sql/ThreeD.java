package com.sql;

class ThreeD {
int x, y, z;
	
	ThreeD(int a, int b, int c) {
		x = a;
		y = b;
		z = c;
	}
	
	int getX() {
		return x;
	}
	
	int getY() {
		return y;
	}
	
	int getZ() {
		return z;
	}
	
	void showType() {
		System.out.println("array.x " + this.x + " array.y " + this.y + " array.z " + this.z);
	}
	
	@Override
	public boolean equals (Object obj) {
		if (obj instanceof ThreeD) {
			if (x == ((ThreeD) obj).x && y == ((ThreeD)obj).y && z == ((ThreeD)obj).z)
				return true;
		}
		return false;
	}
}
