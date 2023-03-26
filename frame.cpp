#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "camera.hpp"
#include "quintic.hpp"

int main() {
	printf("Hello Quintics!\n");
	
	// Render parameters
	const size_t width = 1920;
	const size_t height = 1080;
	
	const size_t iters = 16;
	
	const size_t frames = 192;
	
	camera cam(width, height, 0, 0, (float) 16 / 3, (float) 9 / 3); // 16:9
	
	// Colors for each root.
	rgb colors[5] = {rgb(30, 161, 206), rgb(56, 83, 141), rgb(73, 12, 87), rgb(87, 186, 97), rgb(24, 133, 142)};
	complex roots[5] = {complex(-1.325, 0), complex(0.662, 0.562), complex(0.662, -0.562), complex(0, 1), complex(0, -1)};
	
	// Allocate space for the output files' names and headers.
	char fn[64];
	
	for (int f = 0; f < frames; f++) {
		printf("Frame %d...\n", f);
		
		float t = (float) f / frames * 2 * 3.14159;
		
		// Render
		cam.render(roots, colors, iters, sin(t));
		
		// Save to file.
		sprintf(fn, "./out/%03d.ppm", f);
		cam.save(fn);
	}
	
	return 0;
}