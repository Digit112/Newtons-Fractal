#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "camera.hpp"
#include "quintic.hpp"

int main() {
	printf("Hello Quintics!\n");
	
	// Render parameters
	const size_t width = 2048;
	const size_t height = 1152;
	
	const size_t iters = 15;
	
	const size_t frames = 192;
	
	camera cam(width, height, 0, 0, (float) 16 / 3, (float) 9 / 3, iters); // 16:9
	
	// Colors for each root.
	rgb colors[5] = {rgb(30, 161, 206), rgb(56, 83, 141), rgb(73, 12, 87), rgb(87, 186, 97), rgb(24, 133, 142)};
	complex roots[5] = {complex(-1.325, 0), complex(0.662, 0.562), complex(0.662, -0.562), complex(0, 1), complex(0, -1)};
	
	// Allocate space for the output files' names and headers.
	char fn[64];
	
	cam.cache_paths(roots);
	
	for (int f = 0; f < frames; f++) {
		printf("Frame %d...\n", f);
		
		float t = (float) f / frames;
		t = t * t * t; // Makes the rate of change look smoother in the video.
		
		// Render
		cam.render_paths(roots, colors, t);
		
		// Save to file.
		sprintf(fn, "./out/%03d.ppm", f);
		cam.save(fn);
	}
	
	return 0;
}