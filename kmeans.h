/*-------------------------------------------------------------------------
 * kmeans.h - Header file for K-means clustering
 *-------------------------------------------------------------------------*/

#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>

/* Define types */
typedef void *Pointer;

/* Cluster ID for objects that don't belong to any cluster */
#define KMEANS_NULL_CLUSTER -1

/* Default max iterations */
#define KMEANS_MAX_ITERATIONS 100

/* Return values */
typedef enum kmeans_result {
    KMEANS_OK,
    KMEANS_ERROR,
    KMEANS_EXCEEDED_MAX_ITERATIONS
} kmeans_result;

/* Distance function: calculates distance between two objects */
typedef double (*kmeans_distance_method)(const Pointer a, const Pointer b);

/* Centroid function: calculates centroid for a cluster */
typedef void (*kmeans_centroid_method)(const Pointer *objs, 
                                        const int *clusters,
                                        size_t num_objs, 
                                        int cluster, 
                                        Pointer centroid);

/* Configuration structure for K-means */
typedef struct kmeans_config {
    int k;                              /* Number of clusters */
    size_t num_objs;                    /* Number of objects */
    Pointer *objs;                      /* Array of objects */
    Pointer *centers;                   /* Array of cluster centers */
    int *clusters;                      /* Cluster assignment for each object */
    int max_iterations;                 /* Maximum iterations */
    int total_iterations;               /* Actual iterations performed */
    kmeans_distance_method distance_method;
    kmeans_centroid_method centroid_method;
} kmeans_config;

/* Memory allocation wrappers */
#define kmeans_malloc(size) malloc(size)
#define kmeans_free(ptr) free(ptr)

/* Main K-means function */
kmeans_result kmeans(kmeans_config *config);

#endif /* KMEANS_H */
