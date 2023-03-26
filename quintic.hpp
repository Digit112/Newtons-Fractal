#ifndef NEWTON_QUINTIC
#define NEWTON_QUINTIC

#include "complex.hpp"

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
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	quintic(const complex* roots);
	
	// Constructs quintic and takes its derivative
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	quintic(complex a, complex b, complex c, complex d, complex e, complex f);
	
	// Evaluate function at the passed complex point x + yi
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex eval(const complex& val);
	
	// Evaluate the derivative at the passed complex point x + yi
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	complex der_eval(const complex& val);
	
	// Print the quintic
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	void debug();
};

#include "quintic.cpp"

#endif