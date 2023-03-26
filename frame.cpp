#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "quintic.hpp"

int main() {
	printf("Hello Quintics!\n");
	
	int width = 1920*2;
	int height = 1080*2;
	
	const int iters = 16;
	
	int frames = 192;
	
	uint8_t* img = new uint8_t[width*height*3];
	
	char fn[64];
	char hdr[64];
	
	for (int f = 0; f < frames; f++) {
		float t = (float) f / frames * 2 * 3.14159;
		
		complex roots[5] = {complex(sin(t) - 0.325, 0), complex(0.662, 0.562), complex(0.662, -0.562), complex(0, 1), complex(0, -1)};
		rgb color[5] =     {rgb(30, 161, 206), rgb(56, 83, 141), rgb(73, 12, 87), rgb(87, 186, 97), rgb(24, 133, 142)};
		
		quintic func(roots);
		
		printf("Frame %d: ", f);
		func.debug();
		
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				complex p = complex((float) x / width * 16 / 3 - 2.6, (float) y / height * 9 / 3 - 1.5);
				
				// Iterate the point with Newton's formula
				for (int i = 0; i < iters; i++) {
					complex val = func.eval(p);
					complex der_val = func.der_eval(p);
					
					complex diff = val / der_val;
					
					//printf("p = (%.2f, %.2f), f(p) = (%.2f, %.2f), f'(p) = (%.2f, %.2f)\n", p.x, p.y, val.x, val.y, der_val.x, der_val.y);
					p = p - diff;
				}
				
				// Find which of the roots is the closest to where this point ends up.
				float min_dis = 100;
				int near_pnt = 0;
				for (int i = 0; i < 5; i++) {
					float dis = complex::sqr_dis(p, roots[i]);
					
					if (dis < min_dis) {
						min_dis = dis;
						near_pnt = i;
					}
				}
				
				int ind = (y*width + x) * 3;
				
				img[ind  ] = color[near_pnt].r;
				img[ind+1] = color[near_pnt].g;
				img[ind+2] = color[near_pnt].b;
			}
		}
		
		int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
		
		sprintf(fn, "./out/%03d.ppm", f);
		
		FILE* fout = fopen(fn, "wb");
		fwrite(hdr, 1, hdr_len, fout);
		fwrite(img, 1, width*height*3, fout);
		fclose(fout);
	}
	
	delete[] img;
	
	return 0;
}