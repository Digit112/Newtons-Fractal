#ifndef NEWTON_COMPLEX
#define NEWTON_COMPLEX

// Class for manipulating complex numbers.
class complex {
public:
	float x;
	float y;
	
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex() : x(0), y(0) {}
	
	// Initialize from a real
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex(float x) : x(x), y(0) {}
	
	// Initialize from complex
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex(float x, float y) : x(x), y(y) {}
	
	// Take the conjugate (invert the imaginary component)
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator~() const {
		return complex(x, -y);
	}
	
	// Negate
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator-() const {
		return complex(-x, -y);
	}
	
	// Multiply by a real number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator*(const float& val) const {
		return complex(x*val, y*val);
	}
	
	// Multiply by another complex number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator*(const complex& val) const {
		return complex(x*val.x - y*val.y, x*val.y + val.x*y);
	}
	
	// Add to a real number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator+(const float& val) const {
		return complex(x + val, y);
	}
	
	// Add to a complex number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator+(const complex& val) const {
		return complex(x + val.x, y + val.y);
	}
	
	// Subtract a complex number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator-(const complex& val) const {
		return complex(x - val.x, y - val.y);
	}
	
	// Divide by a real number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator/(const float& val) const {
		return complex(x / val, y / val);
	}
	
	// Divide by a complex number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex operator/(const complex& val) const {
		complex inv_this = ~val;
		
		return (*this * inv_this) / (val * inv_this).x;
	}
	
	// Raise to integer power.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex pow(int val) const {
		if (val == 1) return complex(x, y); // Base case
		
		return *this * pow(val-1); // Recurse
	}
	
	// Find the square distance between to complex numbers.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	static float sqr_dis(const complex& a, const complex& b) {
		float dx = a.x - b.x;
		float dy = a.y - b.y;
		
		return dx*dx + dy*dy;
	}
	
	// Lerp between two complex numbers
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	static complex lerp(const complex& a, const complex& b, float t) {
		return complex(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
	}
	
	// Print the complex number.
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	void debug() {
		if (y < 0) {
			printf("(%.2f - %.2fi)", x, -y);
		}
		else {
			printf("(%.2f + %.2fi)", x, y);
		}
	}
};

#endif