AUTORES:
--------
- Diego Marchioni Alves dos Santos
- Gabriel Martins Rajão
- Luiza Dias Corteletti
- Sophia Carrazza Ventorim de Sousa

DESCRIÇÃO DA APLICAÇÃO:
------------------------
Este projeto implementa o algoritmo K-means clustering paralelizado para
agrupar músicas do Spotify baseado em características musicais.

O K-means é um algoritmo de aprendizado não-supervisionado que agrupa dados
em k clusters, onde cada cluster é representado por seu centróide (centro).

Neste projeto, usamos duas características do Spotify:
- Danceability: quão adequada é a música para dançar (0.0 a 1.0)
- Energy: intensidade e atividade da música (0.0 a 1.0)

CÓDIGO ORIGINAL:
----------------
https://github.com/pramsey/kmeans.git

EXECUÇÃO DO CÓDIGO:
---------------------

Para compilar:
```
make 
```

Para executar a versão openMP (Em que k deve ser o número de clusters, para nossos testes utilizamos k = 500): 

```
./spotify_kmeans_openmp datasetFilt.csv k resultados_spotify.txt NUM_THREADS 
```

Para executar a versão híbrida (Em que k deve ser o número de clusters, para nossos testes utilizamos k = 500): 

```
mpirun -np NUM_PROCESSOS ./spotify_kmeans_mpi datasetFilt.csv k resultados_spotify.txt NUM_THREADS 
```
