/*-------------------------------------------------------------------------
 * kmeans_openmp_gpu.c - Versão com OpenMP GPU (Offload)
 *-------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kmeans.h"

/*
 * FUNÇÃO: update_r
 * PARALELIZAÇÃO: OpenMP GPU Offload
 */
static void update_r(kmeans_config *config) {
    int i;
    
    int *clusters = config->clusters;
    int num_objs = config->num_objs;
    int k = config->k;
    Pointer *objs = config->objs;
    Pointer *centers = config->centers;

    double distance, curr_distance;
    int cluster, curr_cluster;
    Pointer obj;

    #ifdef _OPENMP
    /* FIX: 
     * 1. Replaced 'is_device_ptr' with 'map'. Since data is managed by OpenMP, 
     * we map it (OpenMP handles address translation).
     * 2. Added explicit block { } to separate target from teams.
     */
    #pragma omp target map(to: objs[0:num_objs], centers[0:k]) \
                       map(tofrom: clusters[0:num_objs])
    {
        #pragma omp teams distribute parallel for simd \
                private(distance, curr_distance, cluster, curr_cluster, obj)
        for (i = 0; i < num_objs; i++) {
            
            obj = objs[i];
            /* Note: obj here is a pointer. If it points to Host memory 
             * and Unified Shared Memory is not active, this dereference 
             * might fail on GPU. Assuming flat data or USM. */
            
            if (!obj) {
                clusters[i] = KMEANS_NULL_CLUSTER;
                continue;
            }

            curr_distance = (config->distance_method)(obj, centers[0]);
            curr_cluster = 0;

            for (cluster = 1; cluster < k; cluster++) {
                distance = (config->distance_method)(obj, centers[cluster]);
                if (distance < curr_distance) {
                    curr_distance = distance;
                    curr_cluster = cluster;
                }
            }

            clusters[i] = curr_cluster;
        }
    }
    #else
    /* Sequential fallback if OpenMP is disabled */
    for (i = 0; i < num_objs; i++) {
        obj = objs[i];
        if (!obj) {
            clusters[i] = KMEANS_NULL_CLUSTER;
            continue;
        }
        curr_distance = (config->distance_method)(obj, centers[0]);
        curr_cluster = 0;
        for (cluster = 1; cluster < k; cluster++) {
            distance = (config->distance_method)(obj, centers[cluster]);
            if (distance < curr_distance) {
                curr_distance = distance;
                curr_cluster = cluster;
            }
        }
        clusters[i] = curr_cluster;
    }
    #endif
}

static void update_means(kmeans_config *config) {
    int i;
    for (i = 0; i < config->k; i++) {
        (config->centroid_method)(config->objs, config->clusters,
                                config->num_objs, i, config->centers[i]);
    }
}

kmeans_result kmeans(kmeans_config *config) {
    int iterations = 0;
    int *clusters_last;
    size_t clusters_sz = sizeof(int) * config->num_objs;

    assert(config);
    assert(config->objs);
    assert(config->num_objs);
    assert(config->distance_method);
    assert(config->centroid_method);
    assert(config->centers);
    assert(config->k);
    assert(config->clusters);
    assert(config->k <= config->num_objs);

    memset(config->clusters, 0, clusters_sz);

    if (!config->max_iterations)
        config->max_iterations = KMEANS_MAX_ITERATIONS;

    clusters_last = kmeans_malloc(clusters_sz);

    while (1) {
        memcpy(clusters_last, config->clusters, clusters_sz);

        update_r(config);
        
        update_means(config);

        int *current_clusters = config->clusters;
        int n_objs = config->num_objs;
        
        /* Suppress unused variable warning */
        (void)current_clusters;

        #ifdef _OPENMP
        #pragma omp target update from(current_clusters[0:n_objs])
        #endif
        
        if (memcmp(clusters_last, config->clusters, clusters_sz) == 0) {
            kmeans_free(clusters_last);
            config->total_iterations = iterations;
            return KMEANS_OK;
        }

        if (iterations++ > config->max_iterations) {
            kmeans_free(clusters_last);
            config->total_iterations = iterations;
            return KMEANS_EXCEEDED_MAX_ITERATIONS;
        }
    }

    kmeans_free(clusters_last);
    config->total_iterations = iterations;
    return KMEANS_ERROR;
}