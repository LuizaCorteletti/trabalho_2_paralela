/*-------------------------------------------------------------------------
 * spotify_kmeans.c - K-Means para dados do Spotify com OpenMP
 * 
 * Este é o programa paralelizado com OpenMP
 * 
 * Intel Core i7-11370H:
 * 
 *  1 thread: 16 segs
 *  2 threads:  8 segs
 *  4 threads:  5 segs
 *  8 threads:  9 segs
 *
 * 
 * Parcode:
 *  1 thread: 35 segs
 *  2 threads: 17 segs
 *  4 threads: 12 segs
 *  8 threads: 15 segs
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


static void pt_centroid(const Pointer *objs, const int *clusters,
                        size_t num_objs, int cluster, Pointer centroid) {
    int i;
    int num_cluster = 0;
    double sumx = 0.0, sumy = 0.0;
    point **pts = (point **)objs;
    point *center = (point *)centroid;

    // PARALELIZAÇÃO OPENMP:
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

static int count_lines(FILE *f) {
    int count = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) count++;
    rewind(f);
    return count - 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Uso: %s <csv> <k> <saida>\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(atoi(argv[4]));

    const char *filename = argv[1];
    int k = atoi(argv[2]);
    const char *out_filename = argv[3];

    #ifdef _OPENMP
    printf("OpenMP habilitado. Threads disponíveis: %d\n", omp_get_max_threads());
    #else
    printf("Versão SEQUENCIAL (sem OpenMP)\n");
    #endif

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

    point *pts = calloc(num_objs, sizeof(point));
    point *init = calloc(k, sizeof(point));
    kmeans_config config;
    config.k = k;
    config.num_objs = num_objs;
    config.max_iterations = 200;
    config.distance_method = pt_distance;
    config.centroid_method = pt_centroid;
    config.objs = calloc(num_objs, sizeof(Pointer));
    config.centers = calloc(k, sizeof(Pointer));
    config.clusters = calloc(num_objs, sizeof(int));

    char line[256];
    fgets(line, sizeof(line), f); 
    for (int i = 0; i < num_objs && fgets(line, sizeof(line), f); i++) {
        if (sscanf(line, "%lf,%lf", &pts[i].x, &pts[i].y) == 2) {
            config.objs[i] = &pts[i];
        } else {
            config.objs[i] = NULL;
        }
    }
    fclose(f);
    printf("Lidas %d músicas.\n", num_objs);

    srand(42); 
    for (int i = 0; i < k; i++) {
        int r = rand() % num_objs;
        init[i] = pts[r];
        config.centers[i] = &init[i];
    }

    printf("\nIniciando K-Means com k=%d...\n", k);
    time_t start = time(NULL);
    kmeans_result result = kmeans(&config);
    time_t end = time(NULL);

    printf("\nK-Means concluído (%d iterações, tempo: %lds)\n",
           config.total_iterations, end - start);

    FILE *out = fopen(out_filename, "w");
    if (!out) {
        perror("Erro ao salvar saída");
    } else {
        fprintf(out, "danceability\tenergy\tcluster\n");
        for (int i = 0; i < num_objs; i++) {
            point *p = (point *)config.objs[i];
            if (p) {
                fprintf(out, "%.6f\t%.6f\t%d\n", p->x, p->y, config.clusters[i]);
            }
        }
        fclose(out);
        printf("Resultados salvos em %s\n", out_filename);
    }

    free(pts);
    free(init);
    free(config.objs);
    free(config.clusters);
    free(config.centers);

    return 0;
}
