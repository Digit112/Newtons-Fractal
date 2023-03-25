quintic::quintic(double a, double b, double c, double d, double e, double f) :
	a(a), b(b), c(c), d(d), e(e), f(f),
	da(5*a), db(4*b), dc(3*c), dd(2*d), de(e) {}

complex quintic::eval(const complex& val) {
	// Extremely inefficient.
	// val is being multiplied by itself far too many times. Improve this if performance becomes a problem.
	return val.pow(5)*a + val.pow(4)*b + val.pow(3)*c + val.pow(2)*d + val*e + f;
}

complex quintic::der_eval(const complex& val) {
	// Also extremely inefficient
	return val.pow(4)*da + val.pow(3)*db + val.pow(2)*dc + val*dd + de;
}