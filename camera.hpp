#ifndef NEWTON_CAMERA
#define NEWTON_CAMERA

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "quintic.hpp"

class camera;

// Render the scene. Iterates the points iters times aand colors the image according to which root the final iteration landed nearest.
#ifdef GPU_ENABLED
__global__
#endif
void render_CPU_GPU(const camera& cam, const complex* roots, const quintic func, const rgb* color);

// Iterate points corresponding to the pixels according to Newton's equation and record the values they visit in the camera's paths structure.
#ifdef GPU_ENABLED
__global__
#endif
void cache_paths_CPU_GPU(const camera& cam, const quintic func);

// Render 
#ifdef GPU_ENABLED
__global__
#endif
void render_paths_CPU_GPU(const camera& cam, const complex* roots, const rgb* color, float t);

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
	
	// The number of iterations to perform.
	size_t iters;
	
	// Output image.
	uint8_t* img;
	
	// Paths travelled by points.
	complex* paths;
	
	camera(size_t width, size_t height, float cx, float cy, float cw, float ch, size_t iters);
	
	// Render the scene to the img buffer.
	void render(const complex* roots, const rgb* color);
	
	// Calculate and cache the paths of the points.
	void cache_paths(const complex* roots);
	
	// Interpolate the paths of the points in order 
	void render_paths(const complex* roots, const rgb* color, float t);
	
	// Save the current img to file.
	void save(const char* fn);
	
	~camera();
};

// Allocate camera
camera::camera(size_t width, size_t height, float cx, float cy, float cw, float ch, size_t iters) : width(width), height(height), cx(cx), cy(cy), cw(cw), ch(ch), iters(iters) {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaMallocManaged(&img, width*height*3);
		if (err != cudaSuccess) {
			printf("Error: Could not allocate requested image: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaMallocManaged(&paths, width*height*(iters + 1)*sizeof(complex));
		if (err != cudaSuccess) {
			printf("Error: Could not allocate requested path structure: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
	#else
		img = new uint8_t[width*height*3];
		
		if (img == NULL) {
			printf("Error: Could not allocate requested image.\n");
			exit(1);
		}
		
		paths = new complex[width*height*(iters + 1)];
		
		if (paths == NULL) {
			printf("Error: Could not allocate requested path structure.\n");
			exit(1);
		}
	#endif
}

// Proxy for the render function so that it can be called as a member function.
// This function is also called the same way regardles of whether GPU_ENABLED is defined.
void camera::render(const complex* roots, const rgb* colors) {
	// Generate the quintic.
	quintic func(roots);
	
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
			printf("Error: Moved Camera to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Copy the roots to the GPU
		complex* roots_gpu;
		err = cudaMallocManaged(&roots_gpu, sizeof(complex)*5);
		if (err != cudaSuccess) {
			printf("Error: Allocated Roots: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(roots_gpu, roots, sizeof(complex)*5, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error: Moved Roots to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Copy colors to the GPU
		rgb* colors_gpu;
		err = cudaMallocManaged(&colors_gpu, sizeof(rgb)*5);
		if (err != cudaSuccess) {
			printf("Error: Allocated Colors: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(colors_gpu, colors, sizeof(rgb)*5, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error: Moved Colors to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Render the scene
		render_CPU_GPU<<<grid, block>>>(*cam_gpu, roots_gpu, func, colors_gpu);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error: Launched Kernel: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			printf("Error: Synchronized Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
	#else
		render_CPU_GPU(*this, roots, func, colors);
	#endif
}

void camera::cache_paths(const complex* roots) {
	// Generate the quintic.
	quintic func(roots);
	
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error occurred prior to caching: %s\n", cudaGetErrorName(err));
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
			printf("Error: Moved Camera to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cache_paths_CPU_GPU<<<grid, block>>>(*cam_gpu, func);
		
		cudaDeviceSynchronize();
	#else
		cache_paths_CPU_GPU(*this, func);
	#endif
}

void camera::render_paths(const complex* roots, const rgb* colors, float t) {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error occurred prior to path rendering: %s\n", cudaGetErrorName(err));
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
			printf("Error: Moved Camera to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Copy the roots to the GPU
		complex* roots_gpu;
		err = cudaMallocManaged(&roots_gpu, sizeof(complex)*5);
		if (err != cudaSuccess) {
			printf("Error: Allocated Roots: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(roots_gpu, roots, sizeof(complex)*5, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error: Moved Roots to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Copy colors to the GPU
		rgb* colors_gpu;
		err = cudaMallocManaged(&colors_gpu, sizeof(rgb)*5);
		if (err != cudaSuccess) {
			printf("Error: Allocated Colors: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(colors_gpu, colors, sizeof(rgb)*5, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error: Moved Colors to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		render_paths_CPU_GPU<<<grid, block>>>(*cam_gpu, roots_gpu, colors_gpu, t);
		
		cudaDeviceSynchronize();
	#else
		render_paths_CPU_GPU(*this, roots, colors, t);
	#endif
}

void camera::save(const char* fn) {
	FILE* fout = fopen(fn, "wb");
	if (fout == NULL) {
		printf("Error: Could not open file '%s'\n", fn);
		exit(1);
	}
	
	char hdr[64];
	int hdr_len = sprintf(hdr, "P6 %zu %zu 255 ", width, height);
	
	fwrite(hdr, 1, hdr_len, fout);
	fwrite(img, 1, width*height*3, fout);
	fclose(fout);
}

camera::~camera() {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaFree(img);
		if (err != cudaSuccess) {
			printf("Error Freeing image: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaFree(paths);
		if (err != cudaSuccess) {
			printf("Error Freeing paths structure: %s\n", cudaGetErrorName(err));
			exit(1);
		}
	#else
		delete[] img;
		delete[] paths;
	#endif
}

// Renders the scene. This is a static function so that it can also be a GPU kernel.
#ifdef GPU_ENABLED
__global__
#endif
void render_CPU_GPU(const camera& cam, const complex* roots, const quintic func, const rgb* color) {
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
			for (int i = 0; i < cam.iters; i++) {
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

#ifdef GPU_ENABLED
__global__
#endif
void cache_paths_CPU_GPU(const camera& cam, const quintic func) {
	// Calculate x and y if this is a kernel. Otherwise, use a for loop.
	#ifdef GPU_ENABLED
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	#else
	for (int x = 0; x < cam.width; x++) {
		for (int y = 0; y < cam.height; y++) {
	#endif
			complex p = complex((float) x / cam.width * cam.cw - cam.cw/2 + cam.cx, (float) y / cam.height * cam.ch - cam.ch/2 + cam.cy);
			
			int path_ind = (y*cam.width + x)*(cam.iters+1);
			
			// Iterate the point with Newton's formula
			cam.paths[path_ind] = p;
			for (int i = 0; i < cam.iters; i++) {
				complex val = func.eval(p);
				complex der_val = func.der_eval(p);
				
				complex diff = val / der_val;
				
				//printf("p = (%.2f, %.2f), f(p) = (%.2f, %.2f), f'(p) = (%.2f, %.2f)\n", p.x, p.y, val.x, val.y, der_val.x, der_val.y);
				p = p - diff;
				
				// Record the values visited by this point.
				cam.paths[path_ind + i + 1] = p;
			}
	#ifndef GPU_ENABLED
		}
	}
	#endif
}

// Render 
#ifdef GPU_ENABLED
__global__
#endif
void render_paths_CPU_GPU(const camera& cam, const complex* roots, const rgb* color, float t) {		
	// Allocate space for the bezier calculation
	complex* bezier = new complex[cam.iters];
	
	// Calculate x and y if this is a kernel. Otherwise, use a for loop.
	#ifdef GPU_ENABLED
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	#else
	for (int x = 0; x < cam.width; x++) {
		for (int y = 0; y < cam.height; y++) {
	#endif
			int path_ind = (y*cam.width + x)*(cam.iters+1);
			
			// Perform the first pass of the bezier calculation.
			for (int i = 0; i < cam.iters; i++) {
				bezier[i] = complex::lerp(cam.paths[path_ind + i], cam.paths[path_ind + i + 1], t);
			}
			
			// Perform additional passes of the bezier calculation.
			for (int i = cam.iters - 1; i > 0; i--) {
				for (int j = 0; j < i; j++) {
					bezier[j] = complex::lerp(bezier[j], bezier[j + 1], t);
				}
			}
			
			// Now, God willing, bezier[0] will be the final, interpolated position of our point.
			
			// Color the point based on the nearest root.
			float min_dis = 100;
			int near_pnt = 0;
			for (int i = 0; i < 5; i++) {
				float dis = complex::sqr_dis(bezier[0], roots[i]);
				
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
	
	delete[] bezier;
}

#endif