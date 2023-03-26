// Geenerate a quintic with the given roots. Coefficient a will always be 1.
quintic::quintic(const complex* roots) : a(1), b(0), c(0), d(0), e(0), f(0) {
	// Calculate b
	for (int i = 0; i < 5; i++) {
		b = b - roots[i];
	}
	
	// Calculate c
	for (int i = 0; i < 4; i++) {
		for (int j = i+1; j < 5; j++) {
			c = c + roots[i] * roots[j];
		}
	}
	
	// Calculate d
	for (int i = 0; i < 3; i++) {
		for (int j = i+1; j < 4; j++) {
			for (int k = j+1; k < 5; k++) {
				d = d - roots[i] * roots[j] * roots[k];
			}
		}
	}
	
	// Calculate e
	for (int i = 0; i < 2; i++) {
		for (int j = i+1; j < 3; j++) {
			for (int k = j+1; k < 4; k++) {
				for (int l = k+1; l < 5; l++) {
					e = e + roots[i] * roots[j] * roots[k] * roots[l];
				}
			}
		}
	}
	
	// Calculate f
	f = -roots[0] * roots[1] * roots[2] * roots[3] * roots[4];
	
	// Take the derivative.
	da = a*5;
	db = b*4;
	dc = c*3;
	dd = d*2;
	de = e;
}

quintic::quintic(complex a, complex b, complex c, complex d, complex e, complex f) :
	a(a), b(b), c(c), d(d), e(e), f(f),
	da(a*5), db(b*4), dc(c*3), dd(d*2), de(e) {}

void quintic::scale(complex val) {
	a = a * val;
	b = b * val;
	c = c * val;
	d = d * val;
	e = e * val;
	f = f * val;
	
	da = da * val;
	db = db * val;
	dc = dc * val;
	dd = dd * val;
	de = e;
}

complex quintic::eval(const complex& val) const {
	complex temp = val;
	complex ret = temp*e + f;
	
	temp = temp * val;
	ret = ret + temp*d;
	
	temp = temp * val;
	ret = ret + temp*c;
	
	temp = temp * val;
	ret = ret + temp*b;
	
	temp = temp * val;
	return ret + temp*a;
}

complex quintic::der_eval(const complex& val) const {
	complex temp = val;
	complex ret = temp*dd + de;
	
	temp = temp * val;
	ret = ret + temp*dc;
	
	temp = temp * val;
	ret = ret + temp*db;
	
	temp = temp * val;
	return ret + temp*da;
}

void quintic::debug() {
	printf("f(x) = ");
	
	a.debug();
	printf("x^5 + ");
	
	b.debug();
	printf("x^4 + ");
	
	c.debug();
	printf("x^3 + ");
	
	d.debug();
	printf("x^2 + ");
	
	e.debug();
	printf("x + ");
	
	f.debug();
	printf("\n");
}