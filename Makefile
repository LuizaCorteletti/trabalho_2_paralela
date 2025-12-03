################################################################################
# Makefile para o projeto Spotify K-Means
# Suporta compilação de versões: OpenMP e MPI+OpenMP
################################################################################

# Compiladores
CC = gcc
MPICC = mpicc

# Flags comuns
CFLAGS_BASE = -O3 -g -Wall
LDFLAGS = -lm

# Arquivos fonte
KMEANS_OMP = kmeans_openmp.c
SPOTIFY_OMP = spotify_kmeans_openmp.c
SPOTIFY_MPI = spotify_kmeans_mpi.c
KMEANS_GPU_SRC = kmeans_openmp_gpu.c
SPOTIFY_GPU_SRC = spotify_kmeans_openmp_gpu.c

# Executáveis
EXE_OMP = spotify_kmeans_omp
EXE_MPI = spotify_kmeans_mpi
EXE_OMP_GPU = spotify_kmeans_omp_gpu

################################################################################
# ALVOS
################################################################################

# Alvo padrão: compila ambas as versões
all: openmp mpi
	@echo ""
	@echo "=========================================="
	@echo "Compilação concluída!"
	@echo "=========================================="
	@echo "Executáveis gerados:"
	@echo "  - $(EXE_OMP)  (OpenMP)"
	@echo "  - $(EXE_MPI)  (MPI+OpenMP)"
	@echo ""

# Versão OPENMP
openmp: $(EXE_OMP)

$(EXE_OMP): $(KMEANS_OMP) $(SPOTIFY_OMP) kmeans.h
	@echo "Compilando versão OpenMP..."
	$(CC) $(CFLAGS_BASE) -fopenmp $(KMEANS_OMP) $(SPOTIFY_OMP) -o $(EXE_OMP) $(LDFLAGS)
	@echo "✓ $(EXE_OMP) compilado com sucesso"

# Versão MPI + OpenMP (híbrida)
mpi: $(EXE_MPI)

$(EXE_MPI): $(SPOTIFY_MPI) kmeans.h
	@echo "Compilando versão MPI+OpenMP..."
	$(MPICC) $(CFLAGS_BASE) -fopenmp $(SPOTIFY_MPI) -o $(EXE_MPI) $(LDFLAGS)
	@echo "✓ $(EXE_MPI) compilado com sucesso"

gpu: $(EXE_OMP_GPU)
$(EXE_OMP_GPU): $(KMEANS_GPU_SRC) $(SPOTIFY_GPU_SRC) kmeans.h
	@echo "Tentando compilar versão GPU com: $(CC)"
	$(CC) $(CFLAGS_BASE) -fopenmp $(KMEANS_GPU_SRC) $(SPOTIFY_GPU_SRC) -o $(EXE_OMP_GPU) $(LDFLAGS)
	@echo "✓ Sucesso! Execute com: ./$(EXE_OMP_GPU) $(DATASET) 10 saida.txt 1"

# Limpeza
clean:
	@echo "Limpando arquivos compilados..."
	@rm -f *.o $(EXE_OMP) $(EXE_MPI)
	@echo "✓ Limpeza concluída"

# Ajuda
help:
	@echo "=========================================="
	@echo "Makefile - Projeto Spotify K-Means"
	@echo "=========================================="
	@echo ""
	@echo "Alvos disponíveis:"
	@echo "  make all        - Compila todas as versões"
	@echo "  make openmp     - Compila apenas versão OpenMP"
	@echo "  make mpi        - Compila apenas versão MPI+OpenMP"
	@echo "  make clean      - Remove arquivos compilados"
	@echo "  make help       - Mostra esta ajuda"
	@echo ""
	@echo "Exemplos de execução:"
	@echo ""
	@echo "OpenMP com diferentes números de threads:"
	@echo "  OMP_NUM_THREADS=1 ./$(EXE_OMP) data.csv 10 resultado.txt"
	@echo "  OMP_NUM_THREADS=2 ./$(EXE_OMP) data.csv 10 resultado.txt"
	@echo "  OMP_NUM_THREADS=4 ./$(EXE_OMP) data.csv 10 resultado.txt"
	@echo "  OMP_NUM_THREADS=8 ./$(EXE_OMP) data.csv 10 resultado.txt"
	@echo ""
	@echo "MPI+OpenMP (híbrido):"
	@echo "  OMP_NUM_THREADS=4 mpirun -np 1 ./$(EXE_MPI) data.csv 10 resultado.txt"
	@echo "  OMP_NUM_THREADS=2 mpirun -np 2 ./$(EXE_MPI) data.csv 10 resultado.txt"
	@echo "  OMP_NUM_THREADS=1 mpirun -np 4 ./$(EXE_MPI) data.csv 10 resultado.txt"
	@echo ""

.PHONY: all openmp mpi clean help
