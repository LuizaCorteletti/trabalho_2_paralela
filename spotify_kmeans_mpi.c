/*-------------------------------------------------------------------------
 * spotify_kmeans_mpi.c - Versão HÍBRIDA com MPI + OpenMP
 * 
 * Este é o programaa Híbrido
 * 
 * Parcode: 
 * 
 *   1 processo e 4 threads: 10 segs
 *   2 processos e 2 threads: 23 segs
 *   4 processos e 0 thread:  57 segs
 * 
 *-------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kmeans.h"

typedef struct point {
    double x;
    double y;
} point;

static double pt_distance(const Pointer a, const Pointer b) {
    point *pa = (point *)a;
    point *pb = (point *)b;
    double dx = pa->x - pb->x;
    double dy = pa->y - pb->y;
    return dx * dx + dy * dy;
}

/* 
 * PARALELIZAÇÃO OPENMP: Cálculo de centróide paralelo
 * - #pragma omp parallel for: distribui iterações entre threads
 * - reduction(+:sumx,sumy,num_cluster): acumula valores de forma thread-safe
 * - schedule(guided): distribui carga dinamicamente
 */
static void pt_centroid(const Pointer *objs, const int *clusters,
                        size_t num_objs, int cluster, Pointer centroid) {
    int i;
    int num_cluster = 0;
    double sumx = 0.0, sumy = 0.0;
    point **pts = (point **)objs;
    point *center = (point *)centroid;

    // PARALELIZAÇÃO: Loop paralelo com reduction 
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sumx,sumy,num_cluster) schedule(guided)
    #endif
    for (i = 0; i < num_objs; i++) {
        if (clusters[i] != cluster || objs[i] == NULL)
            continue;
        sumx += pts[i]->x;
        sumy += pts[i]->y;
        num_cluster++;
    }

    if (num_cluster > 0) {
        center->x = sumx / num_cluster;
        center->y = sumy / num_cluster;
    }
}

/* 
 * PARALELIZAÇÃO OPENMP: Atualização de clusters paralela
 * - #pragma omp parallel for
 * - Cada thread processa um subconjunto de objetos
 * - schedule(guided): distribui carga dinamicamente
 */
static void update_r(kmeans_config *config) {
    int i;
    
    // PARALELIZAÇÃO: Loop paralelo sobre todos os objetos
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (i = 0; i < config->num_objs; i++) {
        double distance, curr_distance;
        int cluster, curr_cluster;
        Pointer obj;

        obj = config->objs[i];
        if (!obj) {
            config->clusters[i] = KMEANS_NULL_CLUSTER;
            continue;
        }

        curr_distance = (config->distance_method)(obj, config->centers[0]);
        curr_cluster = 0;

        for (cluster = 1; cluster < config->k; cluster++) {
            distance = (config->distance_method)(obj, config->centers[cluster]);
            if (distance < curr_distance) {
                curr_distance = distance;
                curr_cluster = cluster;
            }
        }

        config->clusters[i] = curr_cluster;
    }
}

static void update_means(kmeans_config *config) {
    int i;
    for (i = 0; i < config->k; i++) {
        (config->centroid_method)(config->objs, config->clusters,
                                config->num_objs, i, config->centers[i]);
    }
}

static int count_lines(FILE *f) {
    int count = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) count++;
    rewind(f);
    return count - 1; 
}

/*
 * DISTRIBUICAO MPI: Agregação de centróides entre processos
 * - Cada processo calcula centróides locais
 * - MPI_Allreduce combina somas de todos os processos
 * - Centróides finais são calculados a partir das somas globais
 */
static void mpi_aggregate_centroids(kmeans_config *config, int rank, int size) {
    int k = config->k;
    int i, j;
    
    double *local_sums_x = calloc(k, sizeof(double));
    double *local_sums_y = calloc(k, sizeof(double));
    int *local_counts = calloc(k, sizeof(int));
    
    double *global_sums_x = calloc(k, sizeof(double));
    double *global_sums_y = calloc(k, sizeof(double));
    int *global_counts = calloc(k, sizeof(int));
    
    for (i = 0; i < config->num_objs; i++) {
        if (config->objs[i] == NULL) continue;
        int cluster = config->clusters[i];
        if (cluster >= 0 && cluster < k) {
            point *pt = (point *)config->objs[i];
            local_sums_x[cluster] += pt->x;
            local_sums_y[cluster] += pt->y;
            local_counts[cluster]++;
        }
    }
    
    // DISTRIBUICAO MPI: Combina somas de todos os processos 
    MPI_Allreduce(local_sums_x, global_sums_x, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_sums_y, global_sums_y, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_counts, global_counts, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
     for (i = 0; i < k; i++) {
        if (global_counts[i] > 0) {
            point *center = (point *)config->centers[i];
            center->x = global_sums_x[i] / global_counts[i];
            center->y = global_sums_y[i] / global_counts[i];
        }
    }
    
    free(local_sums_x);
    free(local_sums_y);
    free(local_counts);
    free(global_sums_x);
    free(global_sums_y);
    free(global_counts);
}


kmeans_result kmeans_mpi(kmeans_config *config, int rank, int size) {
    int iterations = 0;
    int *clusters_last;
    size_t clusters_sz = sizeof(int) * config->num_objs;
    int converged = 0;

    if (!config->max_iterations)
        config->max_iterations = KMEANS_MAX_ITERATIONS;

    clusters_last = calloc(clusters_sz, 1);
    memset(config->clusters, 0, clusters_sz);

    while (!converged && iterations < config->max_iterations) {
        memcpy(clusters_last, config->clusters, clusters_sz);

        update_r(config);
        
        // DISTRIBUICAO MPI: Agrega centróides entre processos
        mpi_aggregate_centroids(config, rank, size);

        int local_converged = (memcmp(clusters_last, config->clusters, clusters_sz) == 0);
        
        // DISTRIBUICAO MPI: Verifica convergência global
        MPI_Allreduce(&local_converged, &converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        iterations++;
    }

    free(clusters_last);
    config->total_iterations = iterations;
    
    if (converged)
        return KMEANS_OK;
    else
        return KMEANS_EXCEEDED_MAX_ITERATIONS;
}

/* FUNÇÃO PRINCIPAL COM MPI */
int main(int argc, char **argv) {
    int rank, size;
    
    // INICIALIZAÇÃO MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            fprintf(stderr, "Uso: %s <csv> <k> <saida>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(atoi(argv[4]));

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    const char *out_filename = argv[3];
    
    int num_objs = 0;
    point *pts = NULL;
    point *init = NULL;
    kmeans_config config;

    if (rank == 0) {
        #ifdef _OPENMP
        printf("MPI+OpenMP habilitado. Processos: %d, Threads/processo: %d\n", 
               size, omp_get_max_threads());
        #else
        printf("MPI habilitado (sem OpenMP). Processos: %d\n", size);
        #endif

        FILE *f = fopen(filename, "r");
        if (!f) {
            perror("Erro ao abrir arquivo CSV");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        num_objs = count_lines(f);
        if (num_objs <= 0) {
            fprintf(stderr, "Nenhum dado encontrado.\n");
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        pts = calloc(num_objs, sizeof(point));
        char line[256];
        fgets(line, sizeof(line), f);

        for (int i = 0; i < num_objs && fgets(line, sizeof(line), f); i++) {
            if (sscanf(line, "%lf,%lf", &pts[i].x, &pts[i].y) != 2) {
                pts[i].x = pts[i].y = 0.0;
            }
        }
        fclose(f);
        printf("Lidas %d músicas.\n", num_objs);
    }

    // DISTRIBUICAO MPI: Broadcast do tamanho dos dados
    MPI_Bcast(&num_objs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        pts = calloc(num_objs, sizeof(point));
    }

    // DISTRIBUICAO MPI: Distribui dados para todos os processos
    MPI_Bcast(pts, num_objs * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    init = calloc(k, sizeof(point));
    config.k = k;
    config.num_objs = num_objs;
    config.max_iterations = 200;
    config.distance_method = pt_distance;
    config.centroid_method = pt_centroid;
    config.objs = calloc(num_objs, sizeof(Pointer));
    config.centers = calloc(k, sizeof(Pointer));
    config.clusters = calloc(num_objs, sizeof(int));

    for (int i = 0; i < num_objs; i++) {
        config.objs[i] = &pts[i];
    }

    if (rank == 0) {
        srand(42); 
        for (int i = 0; i < k; i++) {
            int r = rand() % num_objs;
            init[i] = pts[r];
        }
    }

    // DISTRIBUICAO MPI: Distribui centróides iniciais
    MPI_Bcast(init, k * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < k; i++) {
        config.centers[i] = &init[i];
    }

    if (rank == 0) {
        printf("\nIniciando K-Means com k=%d (MPI+OpenMP)...\n", k);
    }

    double start_time = MPI_Wtime();
    kmeans_result result = kmeans_mpi(&config, rank, size);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nK-Means concluído (%d iterações, tempo: %.4f segundos)\n",
               config.total_iterations, end_time - start_time);

        FILE *out = fopen(out_filename, "w");
        if (!out) {
            perror("Erro ao salvar saída");
        } else {
            fprintf(out, "danceability\tenergy\tcluster\n");
            for (int i = 0; i < num_objs; i++) {
                point *p = (point *)config.objs[i];
                fprintf(out, "%.6f\t%.6f\t%d\n", p->x, p->y, config.clusters[i]);
            }
            fclose(out);
            printf("Resultados salvos em %s\n", out_filename);
        }
    }

    free(pts);
    free(init);
    free(config.objs);
    free(config.clusters);
    free(config.centers);

    MPI_Finalize();
    return 0;
}
