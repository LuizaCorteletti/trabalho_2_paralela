/*-------------------------------------------------------------------------
 * kmeans.cu - CUDA Version
 *-------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "kmeans.h"

// Define Point locally
typedef struct {
    double x;
    double y;
} Point;

// Define CUDA_CHECK locally
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", \
                    cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// Hardware limitation check
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
#error "This code requires a GPU with Compute Capability 6.0 or higher for double precision atomics."
#endif

__global__ void kernel_update_clusters(const Point* __restrict__ objs, 
                                       const Point* __restrict__ centers, 
                                       int* __restrict__ clusters, 
                                       int num_objs, 
                                       int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_objs) return;

    Point p = objs[idx];
    
    double min_dist = INFINITY;
    int best_cluster = 0;

    for (int i = 0; i < k; i++) {
        double dx = p.x - centers[i].x;
        double dy = p.y - centers[i].y;
        double dist = dx * dx + dy * dy;

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
        }
    }

    clusters[idx] = best_cluster;
}

__global__ void kernel_update_means(const Point* __restrict__ objs, 
                                    const int* __restrict__ clusters, 
                                    Point* __restrict__ new_centers, 
                                    int* __restrict__ counts, 
                                    int num_objs, 
                                    int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_objs) return;

    int cluster_id = clusters[idx];
    Point p = objs[idx];

    atomicAdd(&(new_centers[cluster_id].x), p.x);
    atomicAdd(&(new_centers[cluster_id].y), p.y);
    atomicAdd(&(counts[cluster_id]), 1);
}

__global__ void kernel_finalize_means(Point* centers, const int* counts, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= k) return;

    int count = counts[idx];
    if (count > 0) {
        centers[idx].x /= count;
        centers[idx].y /= count;
    }
}

extern "C" kmeans_result kmeans(kmeans_config *config) {
    int iterations = 0;
    
    Point *d_new_centers;
    int *d_counts;
    int *h_clusters_prev; 
    int *h_clusters_curr; // Added local buffer for current state

    size_t clusters_sz = sizeof(int) * config->num_objs;
    size_t centers_sz = sizeof(Point) * config->k;

    CUDA_CHECK(cudaMalloc(&d_new_centers, centers_sz));
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * config->k));
    
    h_clusters_prev = (int*)malloc(clusters_sz);
    h_clusters_curr = (int*)malloc(clusters_sz); // Allocate local buffer

    // Block/Grid configuration
    int blockSize = 256;
    int gridSize = (config->num_objs + blockSize - 1) / blockSize;

    while (1) {
        // 1. Store previous clusters to check convergence
        CUDA_CHECK(cudaMemcpy(h_clusters_prev, config->clusters, clusters_sz, cudaMemcpyDeviceToHost));

        // 2. Kernel: Assign Clusters
        // Cast the generic Pointer* (void**) to Point*
        kernel_update_clusters<<<gridSize, blockSize>>>(
            (Point*)config->objs, 
            (Point*)config->centers, 
            config->clusters, 
            config->num_objs, 
            config->k
        );
        CUDA_CHECK(cudaGetLastError());

        // 3. Update Means
        CUDA_CHECK(cudaMemset(d_new_centers, 0, centers_sz));
        CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * config->k));

        kernel_update_means<<<gridSize, blockSize>>>(
            (Point*)config->objs, 
            config->clusters, 
            d_new_centers, 
            d_counts, 
            config->num_objs, 
            config->k
        );
        
        int meanBlockSize = (config->k < 256) ? config->k : 256;
        int meanGridSize = (config->k + meanBlockSize - 1) / meanBlockSize;
        
        kernel_finalize_means<<<meanGridSize, meanBlockSize>>>(d_new_centers, d_counts, config->k);
        CUDA_CHECK(cudaGetLastError());

        // Update main centers
        CUDA_CHECK(cudaMemcpy(config->centers, d_new_centers, centers_sz, cudaMemcpyDeviceToDevice));

        // 4. Convergence Check
        // Copy new clusters to local host buffer instead of config->h_clusters
        CUDA_CHECK(cudaMemcpy(h_clusters_curr, config->clusters, clusters_sz, cudaMemcpyDeviceToHost));
        
        iterations++;
        
        if (memcmp(h_clusters_prev, h_clusters_curr, clusters_sz) == 0) {
            config->total_iterations = iterations;
            free(h_clusters_prev);
            free(h_clusters_curr);
            cudaFree(d_new_centers);
            cudaFree(d_counts);
            return KMEANS_OK;
        }

        if (iterations >= config->max_iterations) {
            config->total_iterations = iterations;
            free(h_clusters_prev);
            free(h_clusters_curr);
            cudaFree(d_new_centers);
            cudaFree(d_counts);
            return KMEANS_EXCEEDED_MAX_ITERATIONS;
        }
    }

    return KMEANS_ERROR;
}