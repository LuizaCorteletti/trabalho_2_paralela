/*-------------------------------------------------------------------------
 * kmeans.c - Versão com OpenMP
 * 
 * Código Kmeans principal
 * 
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
 * PARALELIZAÇÃO: OpenMP parallel for
 * 
 * Esta função atribui cada objeto ao cluster mais próximo.
 * 
 * MUDANÇA REALIZADA PARA PARALELIZAÇÃO:
 * - Adicionado #pragma omp parallel for schedule(guided)
 * - O loop externo sobre todos os objetos é paralelizado
 * - Cada thread calcula distâncias para um subconjunto de objetos
 * - schedule(guided) distribui carga dinamicamente para balanceamento
 */
static void update_r(kmeans_config *config) {
    int i;
    
    /* PARALELIZAÇÃO: Loop paralelo sobre os objetos
     * - schedule(guided): chunks de tamanho decrescente para balanceamento
     * - Cada thread trabalha em um subconjunto disjunto de objetos
     * - Não há race condition pois cada thread escreve em config->clusters[i] diferente
     */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided, 128)
    #endif
    for (i = 0; i < config->num_objs; i++) {
        double distance, curr_distance;
        int cluster, curr_cluster;
        Pointer obj;

        assert(config->objs != NULL);
        assert(config->num_objs > 0);
        assert(config->centers);
        assert(config->clusters);

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
