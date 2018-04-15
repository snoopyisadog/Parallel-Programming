#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            row[x * 3] = ((p & 0xf) << 4);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
	/* argument parsing */
	assert(argc == 9);
    int num_threads = strtol(argv[1], 0, 10); // long int strtol (const char* str, char** endptr, int base);
    double left = strtod(argv[2], 0);         // double strtod (const char* str, char** endptr);
    double right = strtod(argv[3], 0);    // left & right are for the x-axis(real number)
    double lower = strtod(argv[4], 0);    // lower & upper are for the y-axis(imaginary number)
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);   // number of points in x-axis
    int height = strtol(argv[7], 0, 10);  // number of points in y-axis
    const char* filename = argv[8];
	
	double s1 = omp_get_wtime();
	
    /* allocate memory for image */
	int totNum = width * height;
	int* image = (int*)malloc(totNum * sizeof(int));
    assert(image);

	double x_chunk= ((double)(right-left) / width);
	double y_chunk= ((double)(upper-lower) / height);
	
	omp_set_num_threads(num_threads);
#pragma omp parallel
	{
		int rank_thread, size_thread;
		size_thread = omp_get_num_threads();
		rank_thread = omp_get_thread_num();
		double t1 = omp_get_wtime();
		int pt = rank_thread;
		while(pt<totNum){
			double x0 = (pt%width)*x_chunk + left;
			double y0 = (pt/width)*y_chunk + lower;
			int repeats = 0;
			double x = 0, y = 0, length_squared = 0; /* z0 = 0, z1 = z0^2 + C = C. */
			while(repeats < 100000 && length_squared < 4){
				double temp = x*x - y*y + x0;
				y = 2 * x * y + y0; /* this is the imag part */
				x = temp;           /* this is the real part */
				length_squared = x*x + y*y;
				++repeats;
			}
			//printf("pt=%d @%d\n",pt,rank_thread);
			image[pt] = repeats;
			pt += size_thread;
		}
		double t2 = omp_get_wtime();
		printf("time=%lf @thread %d\n", t2-t1, rank_thread);
	}

	write_png(filename, width, height, image);
    free(image);
	double s2 = omp_get_wtime();
	printf("omp total time:%lf sec\n", s2-s1);
}
