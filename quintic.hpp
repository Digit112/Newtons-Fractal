#ifndef NEWTON_QUINTIC
#define NEWTON_QUINTIC

// Quick & dirty complex number class
class complex {
public:
	float x;
	float y;
	
	complex() : x(0), y(0) {}
	
	// Initialize from a real
	complex(float x) : x(x), y(0) {}
	
	// Initialize from complex
	complex(float x, float y) : x(x), y(y) {}
	
	// Take the conjugate (invert the imaginary component)
	complex operator~() const {
		return complex(x, -y);
	}
	
	// Negate
	complex operator-() const {
		return complex(-x, -y);
	}
	
	// Multiply by a real number.
	complex operator*(const float& val) const {
		return complex(x*val, y*val);
	}
	
	// Multiply by another complex number.
	complex operator*(const complex& val) const {
		return complex(x*val.x - y*val.y, x*val.y + val.x*y);
	}
	
	// Add to a real number
	complex operator+(const float& val) const {
		return complex(x + val, y);
	}
	
	// Add to a complex number.
	complex operator+(const complex& val) const {
		return complex(x + val.x, y + val.y);
	}
	
	// Subtract a complex number
	complex operator-(const complex& val) const {
		return complex(x - val.x, y - val.y);
	}
	
	// Divide by a real number.
	complex operator/(const float& val) const {
		return complex(x / val, y / val);
	}
	
	// Divide by a complex number.
	complex operator/(const complex& val) const {
		complex inv_this = ~val;
		
		return (*this * inv_this) / (val * inv_this).x;
	}
	
	// Raise to integer power.
	complex pow(int val) const {
		if (val == 1) return complex(x, y); // Base case
		
		return *this * pow(val-1); // Recurse
	}
	
	// Find the square distance between to complex numbers.
	static float sqr_dis(const complex& a, const complex& b) {
		float dx = a.x - b.x;
		float dy = a.y - b.y;
		
		return dx*dx + dy*dy;
	}
	
	void debug() {
		if (y < 0) {
			printf("(%.2f - %.2fi)", x, -y);
		}
		else {
			printf("(%.2f + %.2fi)", x, y);
		}
	}
};

// Quick & dirty rgb class.
class rgb {
public:
	uint8_t r;
	uint8_t g;
	uint8_t b;
	
	rgb(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
};

// Represents quintic formula
class quintic {
public:
	// Coefficients (y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f)
	complex a;
	complex b;
	complex c;
	complex d;
	complex e;
	complex f;
	
	// Cefficients of the derivative (y = 5ax^4 + 4bx^3 + 3cx^2 + 2dx + e = dax^4 + dbx^3 + dcx^2 + ddx + de)
	complex da;
	complex db;
	complex dc;
	complex dd;
	complex de; // Equal to e, kind of redundant.
	
	// Construct from a list of 5 roots.
	quintic(const complex* roots);
	
	// Constructs quintic and takes its derivative
	quintic(complex a, complex b, complex c, complex d, complex e, complex f);
	
	// Evaluate function at the passed complex point x + yi
	complex eval(const complex& val);
	
	// Evaluate the derivative at the passed complex point x + yi
	complex der_eval(const complex& val);
	
	// Prints the quintic
	void debug();
};

#include "quintic.cpp"

#endif