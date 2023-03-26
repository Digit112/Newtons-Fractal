#ifndef NEWTON_CAMERA
#define NEWTON_CAMERA

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "quintic.hpp"

class camera {
public:
	// Position of camera.
	float cx;
	float cy;
	
	// Width and height of the region of the complex plane that we'll render.
	float cw;
	float ch;
	
	// Output image size.
	size_t width;
	size_t height;
	
	// Output image.
	uint8_t* img;
	
	camera(size_t width, size_t height, float cx, float cy, float cw, float ch);
	
	void render(const complex* roots, const rgb* color, size_t iters);
	
	// Render the scene
	static void render_CPU_GPU(const camera& cam, const complex* roots, const rgb* color, size_t iters);
	
	// Save the current img to file.
	void save(const char* fn);
	
	~camera();
};

// Allocate camera
camera::camera(size_t width, size_t height, float cx, float cy, float cw, float ch) : width(width), height(height), cx(cx), cy(cy), cw(cw), ch(ch) {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaMallocManaged(&img, width*height*3);
		if (err != cudaSuccess) {
			printf("Error: Could not allocate requested image: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
	#else
		img = new uint8_t[width*height*3];
		
		if (img == NULL) {
			printf("Error: Could not allocate requested image.\n");
			exit(1);
		}
	#endif
}

// Proxy for the render function so that it can be called as a member function.
// This function is also called the same way regardles of whether GPU_ENABLED is defined.
void camera::render(const complex* roots, const rgb* color, size_t iters) {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error occurred prior to rendering: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		dim3 block(16, 16);
		dim3 grid(width / 16, height / 16);
		
		// Copy the camera to the GPU
		camera* cam_gpu;
		err = cudaMallocManaged(&cam_gpu, sizeof(camera));
		if (err != cudaSuccess) {
			printf("Error: Allocated Scene Descriptor: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(cam_gpu, this, sizeof(camera), cudaMemcpyHostToDevice);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Moved Camera to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Render the scene
		render<<<grid, block>>>(*this, roots, color, iters);
		
	#else
		camera::render_CPU_GPU(*this, roots, color, iters);
	#endif
}

// Renders the scene. This is a static function so that it can also be a GPU kernel.
#ifdef GPU_ENABLED
__global__
#endif
void camera::render_CPU_GPU(const camera& cam, const complex* roots, const rgb* color, size_t iters) {
	quintic func(roots);
	
	// Calculate x and y if this is a kernel. Otherwise, use a for loop.
	#ifdef GPU_ENABLED
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	#else
	for (int x = 0; x < cam.width; x++) {
		for (int y = 0; y < cam.height; y++) {
	#endif
			complex p = complex((float) x / cam.width * cam.cw - cam.cw/2 + cam.cx, (float) y / cam.height * cam.ch - cam.ch/2 + cam.cy);
			
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
			
			int ind = (y*cam.width + x) * 3;
			
			cam.img[ind  ] = color[near_pnt].r;
			cam.img[ind+1] = color[near_pnt].g;
			cam.img[ind+2] = color[near_pnt].b;
	
	#ifndef GPU_ENABLED
		}
	}
	#endif
}

void camera::save(const char* fn) {
	FILE* fout = fopen(fn, "wb");
	if (fout == NULL) {
		printf("Error: Could not open file '%s'\n", fn);
		exit(1);
	}
	
	char hdr[64];
	int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
	
	fwrite(hdr, 1, hdr_len, fout);
	fwrite(img, 1, width*height*3, fout);
	fclose(fout);
}

camera::~camera() {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaFree(img)
		if (err != cudaSuccess) {
			printf("Error Freeing image:%s\n", cudaGetErrorName(err));
			exit(1);
		}
	#else
		delete[] img;
	#endif
}

#endif