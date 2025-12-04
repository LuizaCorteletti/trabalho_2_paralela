// spotify_kmeans_cuda.cu
// Compile: nvcc -O3 spotify_kmeans_cuda.cu -o spotify_kmeans_cuda
// Run (defaults): ./spotify_kmeans_cuda input.csv <k> out.tsv
// Run (custom):   ./spotify_kmeans_cuda input.csv <k> out.tsv <threads> <blocks>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

typedef struct {
    double x;
    double y;
} Point;

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(1);                                                  \
    }                                                             \
} while (0)

// Count lines (data lines excluding header)
static int count_lines(FILE *f) {
    int count = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) count++;
    rewind(f);
    return (count > 0) ? count - 1 : 0;
}

// Kernel #1: assign each point to nearest centroid
__global__ void assign_clusters_kernel(const Point *pts, const Point *centers, int *clusters, int num_objs, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < num_objs; i += stride) {
        Point p = pts[i];
        double bestd = INFINITY;
        int bestc = 0;
        for (int c = 0; c < k; ++c) {
            double dx = p.x - centers[c].x;
            double dy = p.y - centers[c].y;
            double d = dx*dx + dy*dy;
            if (d < bestd) { bestd = d; bestc = c; }
        }
        clusters[i] = bestc;
    }
}

// Kernel #2: compute per-block partial sum & count for a single cluster.
// Shared memory layout: [blockDim.x doubles sumx][blockDim.x doubles sumy][blockDim.x ints count]
// Since extern shared is double[], we reinterpret the tail as ints.
__global__ void reduce_cluster_kernel(const Point *pts, const int *clusters,
                                      Point *block_sums, int *block_counts,
                                      int num_objs, int k, int cluster_id) {
    extern __shared__ double sarr[]; // size must be: 2*blockDim.x*sizeof(double) + blockDim.x*sizeof(int)
    double *s_sumx = sarr;                       // blockDim.x doubles
    double *s_sumy = sarr + blockDim.x;          // blockDim.x doubles
    int *s_count = (int*)(sarr + 2*blockDim.x);  // blockDim.x ints (reinterpreted)

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bidx = bid * blockDim.x + tid;

    // Local accumulation
    double loc_sumx = 0.0;
    double loc_sumy = 0.0;
    int loc_count = 0;

    int stride = gridDim.x * blockDim.x;
    for (int i = bidx; i < num_objs; i += stride) {
        if (clusters[i] == cluster_id) {
            loc_sumx += pts[i].x;
            loc_sumy += pts[i].y;
            loc_count += 1;
        }
    }

    // Write to shared
    s_sumx[tid] = loc_sumx;
    s_sumy[tid] = loc_sumy;
    s_count[tid] = loc_count;
    __syncthreads();

    // Reduction in shared memory (power-of-two-friendly)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sumx[tid] += s_sumx[tid + s];
            s_sumy[tid] += s_sumy[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_idx = bid * k + cluster_id;
        block_sums[out_idx].x = s_sumx[0];
        block_sums[out_idx].y = s_sumy[0];
        block_counts[out_idx] = s_count[0];
    }
}

int main(int argc, char **argv) {
    // Accept either 4 args (defaults threads/blocks) or 6 args
    if (argc != 4 && argc != 6) {
        fprintf(stderr, "Uso: %s <csv> <k> <saida> [threadsPorBloco blocksPorGrid]\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    const char *out_filename = argv[3];

    // Defaults (overriden if argc == 6)
    int threads = 256;
    int blocks = 0; // compute after reading num_objs

    if (argc == 6) {
        threads = atoi(argv[4]);
        blocks  = atoi(argv[5]);
        if (threads <= 0 || blocks <= 0) {
            fprintf(stderr, "threads e blocks devem ser > 0\n");
            return 1;
        }
    }

    FILE *f = fopen(filename, "r");
    if (!f) { perror("Erro ao abrir arquivo CSV"); return 1; }

    int num_objs = count_lines(f);
    if (num_objs <= 0) { fprintf(stderr, "Nenhum dado lido.\n"); fclose(f); return 1; }

    Point *pts_h = (Point*)calloc((size_t)num_objs, sizeof(Point));
    Point *centers_h = (Point*)calloc((size_t)k, sizeof(Point));
    int *clusters_h = (int*)calloc((size_t)num_objs, sizeof(int));
    if (!pts_h || !centers_h || !clusters_h) { fprintf(stderr, "Memória insuficiente (host)\n"); return 1; }

    char line[4096];
    if (fgets(line, sizeof(line), f)) {} // skip header
    int read = 0;
    while (read < num_objs && fgets(line, sizeof(line), f)) {
        double a,b;
        if (sscanf(line, "%lf,%lf", &a, &b) == 2) {
            pts_h[read].x = a;
            pts_h[read].y = b;
        } else {
            pts_h[read].x = pts_h[read].y = 0.0;
        }
        read++;
    }
    fclose(f);

    printf("Lidas %d músicas.\n", num_objs);

    // If blocks not provided, compute a reasonable default
    if (blocks == 0) {
        blocks = (num_objs + threads - 1) / threads;
        if (blocks < 1) blocks = 1;
    }

    // Query device limits and validate parameters
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxThreads = prop.maxThreadsPerBlock;
    size_t sharedMb = prop.sharedMemPerBlock; // bytes

    if (threads > maxThreads) {
        fprintf(stderr, "Threads por bloco (%d) excede maxThreadsPerBlock (%d)\n", threads, maxThreads);
        return 1;
    }

    // Compute shared memory required for reduction kernel
    size_t shmem_per_block = (size_t)(2 * threads * sizeof(double)) + (size_t)(threads * sizeof(int));
    if (shmem_per_block > sharedMb) {
        fprintf(stderr, "Shared memory por bloco requerida (%zu bytes) excede limite da GPU (%zu bytes).\n",
                shmem_per_block, (size_t)sharedMb);
        fprintf(stderr, "Reduza 'threads' ou use outra GPU.\n");
        return 1;
    }

    // Simple check for allocation size of block partial arrays
    size_t blocks_k = (size_t)blocks * (size_t)k;
    if (blocks_k == 0) { fprintf(stderr, "blocks * k overflow / inválido\n"); return 1; }

    // Initialize centroids (deterministic seed)
    srand(42);
    for (int i = 0; i < k; ++i) {
        int r = rand() % num_objs;
        centers_h[i] = pts_h[r];
    }

    // Device allocations
    Point *pts_d = NULL;
    Point *centers_d = NULL;
    int *clusters_d = NULL;

    CUDA_CHECK(cudaMalloc((void**)&pts_d, sizeof(Point) * (size_t)num_objs));
    CUDA_CHECK(cudaMalloc((void**)&centers_d, sizeof(Point) * (size_t)k));
    CUDA_CHECK(cudaMalloc((void**)&clusters_d, sizeof(int) * (size_t)num_objs));

    CUDA_CHECK(cudaMemcpy(pts_d, pts_h, sizeof(Point) * (size_t)num_objs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(centers_d, centers_h, sizeof(Point) * (size_t)k, cudaMemcpyHostToDevice));

    // Allocate storage for per-block partial sums for all clusters: blocks * k entries
    Point *block_sums_d = NULL;
    int *block_counts_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&block_sums_d, sizeof(Point) * blocks_k));
    CUDA_CHECK(cudaMalloc((void**)&block_counts_d, sizeof(int) * blocks_k));

    // Host-side buffers for partials
    Point *block_sums_h = (Point*)malloc(sizeof(Point) * (size_t)blocks_k);
    int *block_counts_h = (int*)malloc(sizeof(int) * (size_t)blocks_k);
    if (!block_sums_h || !block_counts_h) { fprintf(stderr, "Memória insuficiente (host partials)\n"); return 1; }

    Point *sums_h = (Point*)malloc(sizeof(Point) * (size_t)k); // final sums per cluster
    int *counts_h = (int*)malloc(sizeof(int) * (size_t)k);     // final counts per cluster
    if (!sums_h || !counts_h) { fprintf(stderr, "Memória insuficiente (host sums)\n"); return 1; }

    const int max_iter = 200;
    const double eps = 1e-6;

    // Prepare CUDA timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        // 1) Assign clusters on GPU
        assign_clusters_kernel<<<(unsigned int)blocks, (unsigned int)threads>>>(pts_d, centers_d, clusters_d, num_objs, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) For each cluster, run reduction kernel to compute per-block partials
        // Shared memory per block already validated
        for (int c = 0; c < k; ++c) {
            reduce_cluster_kernel<<<(unsigned int)blocks, (unsigned int)threads, shmem_per_block>>>(
                pts_d, clusters_d, block_sums_d, block_counts_d, num_objs, k, c
            );
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3) Copy block partials to host
        CUDA_CHECK(cudaMemcpy(block_sums_h, block_sums_d, sizeof(Point) * blocks_k, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(block_counts_h, block_counts_d, sizeof(int) * blocks_k, cudaMemcpyDeviceToHost));

        // 4) Aggregate on host to compute new centroids
        for (int c = 0; c < k; ++c) {
            sums_h[c].x = 0.0;
            sums_h[c].y = 0.0;
            counts_h[c] = 0;
        }

        for (int b = 0; b < blocks; ++b) {
            for (int c = 0; c < k; ++c) {
                size_t idx = (size_t)b * (size_t)k + (size_t)c;
                sums_h[c].x += block_sums_h[idx].x;
                sums_h[c].y += block_sums_h[idx].y;
                counts_h[c] += block_counts_h[idx];
            }
        }

        double max_shift = 0.0;
        for (int c = 0; c < k; ++c) {
            double oldx = centers_h[c].x;
            double oldy = centers_h[c].y;
            if (counts_h[c] > 0) {
                centers_h[c].x = sums_h[c].x / counts_h[c];
                centers_h[c].y = sums_h[c].y / counts_h[c];
            } else {
                // Reinitialize empty cluster to a random point (simple fallback)
                int r = rand() % num_objs;
                centers_h[c] = pts_h[r];
            }
            double dx = centers_h[c].x - oldx;
            double dy = centers_h[c].y - oldy;
            double shift = sqrt(dx*dx + dy*dy);
            if (shift > max_shift) max_shift = shift;
        }

        // 5) Copy updated centers to device
        CUDA_CHECK(cudaMemcpy(centers_d, centers_h, sizeof(Point) * (size_t)k, cudaMemcpyHostToDevice));

        if (max_shift < eps) {
            printf("Converged after %d iterations\n", iter + 1);
            ++iter;
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // Copy final clusters back to host
    CUDA_CHECK(cudaMemcpy(clusters_h, clusters_d, sizeof(int) * (size_t)num_objs, cudaMemcpyDeviceToHost));

    printf("\nK-Means concluído (%d iterações, tempo: %.3f ms)\n", iter, elapsed_ms);

    // Write results to output
    FILE *out = fopen(out_filename, "w");
    if (out) {
        fprintf(out, "danceability\tenergy\tcluster\n");
        for (int i = 0; i < num_objs; ++i) {
            fprintf(out, "%.6f\t%.6f\t%d\n", pts_h[i].x, pts_h[i].y, clusters_h[i]);
        }
        fclose(out);
        printf("Resultados salvos em %s\n", out_filename);
    } else {
        perror("Erro ao abrir arquivo de saída");
    }

    // Cleanup
    free(pts_h);
    free(centers_h);
    free(clusters_h);
    free(block_sums_h);
    free(block_counts_h);
    free(sums_h);
    free(counts_h);

    CUDA_CHECK(cudaFree(pts_d));
    CUDA_CHECK(cudaFree(centers_d));
    CUDA_CHECK(cudaFree(clusters_d));
    CUDA_CHECK(cudaFree(block_sums_d));
    CUDA_CHECK(cudaFree(block_counts_d));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
