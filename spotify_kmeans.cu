/*-------------------------------------------------------------------------
 * spotify_kmeans.cu - CUDA Host Code
 *-------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kmeans.h"

// 1. Define Point struct locally since it is not in the header
typedef struct {
    double x;
    double y;
} Point;

// 2. Define CUDA_CHECK macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", \
                    cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

static int count_lines(FILE *f) {
    int count = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) count++;
    rewind(f);
    return count - 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Uso: %s <csv> <k> <saida> <threads_ignored_in_cuda>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    const char *out_filename = argv[3];

    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Usando GPU: %s\n", prop.name);

    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Erro ao abrir arquivo CSV");
        return 1;
    }

    int num_objs = count_lines(f);
    if (num_objs <= 0) {
        fprintf(stderr, "Nenhum dado encontrado.\n");
        fclose(f);
        return 1;
    }

    // Host allocation
    Point *h_pts = (Point*)calloc(num_objs, sizeof(Point));
    Point *h_init_centers = (Point*)calloc(k, sizeof(Point));
    int *h_clusters = (int*)calloc(num_objs, sizeof(int));

    // Parse CSV
    char line[256];
    fgets(line, sizeof(line), f); // skip header
    for (int i = 0; i < num_objs && fgets(line, sizeof(line), f); i++) {
        if (sscanf(line, "%lf,%lf", &h_pts[i].x, &h_pts[i].y) != 2) {
            h_pts[i].x = 0; h_pts[i].y = 0;
        }
    }
    fclose(f);
    printf("Lidas %d músicas.\n", num_objs);

    // Init Centroids (Random pick)
    srand(42); 
    for (int i = 0; i < k; i++) {
        int r = rand() % num_objs;
        h_init_centers[i] = h_pts[r];
    }

    // GPU Allocation
    Point *d_objs;
    Point *d_centers;
    int *d_clusters;

    size_t bytes_objs = num_objs * sizeof(Point);
    size_t bytes_centers = k * sizeof(Point);
    size_t bytes_clusters = num_objs * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_objs, bytes_objs));
    CUDA_CHECK(cudaMalloc(&d_centers, bytes_centers));
    CUDA_CHECK(cudaMalloc(&d_clusters, bytes_clusters));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_objs, h_pts, bytes_objs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centers, h_init_centers, bytes_centers, cudaMemcpyHostToDevice));

    // Setup Config
    kmeans_config config;
    config.k = k;
    config.num_objs = num_objs;
    config.max_iterations = 200;
    
    // Set Device Pointers
    // Cast to (Pointer*) which matches void** from the header
    config.objs = (Pointer*)d_objs;
    config.centers = (Pointer*)d_centers;
    config.clusters = d_clusters;
    
    // NOTE: removed config.h_clusters/h_centers assignments
    // because those members do not exist in kmeans.h

    printf("\nIniciando K-Means (CUDA) com k=%d...\n", k);
    time_t start = time(NULL);
    
    kmeans_result result = kmeans(&config);
    
    // Ensure all GPU work is done
    cudaDeviceSynchronize();
    time_t end = time(NULL);

    printf("\nK-Means concluído (%d iterações, tempo: %lds)\n",
           config.total_iterations, end - start);

    // Retrieve results manually since config didn't hold the host pointer
    CUDA_CHECK(cudaMemcpy(h_clusters, d_clusters, bytes_clusters, cudaMemcpyDeviceToHost));

    // Save results
    FILE *out = fopen(out_filename, "w");
    if (!out) {
        perror("Erro ao salvar saída");
    } else {
        fprintf(out, "danceability\tenergy\tcluster\n");
        for (int i = 0; i < num_objs; i++) {
            fprintf(out, "%.6f\t%.6f\t%d\n", h_pts[i].x, h_pts[i].y, h_clusters[i]);
        }
        fclose(out);
        printf("Resultados salvos em %s\n", out_filename);
    }

    // Cleanup
    free(h_pts);
    free(h_init_centers);
    free(h_clusters);
    
    cudaFree(d_objs);
    cudaFree(d_centers);
    cudaFree(d_clusters);

    return 0;
}