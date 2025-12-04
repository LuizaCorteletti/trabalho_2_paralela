################################################################################
# Makefile para o projeto Spotify K-Means
# Suporta compilação de versões: OpenMP e Cuda
################################################################################

# Compiladores
CC = gcc
NVCC = nvcc

# Flags comuns
CFLAGS_BASE = -O3 -g -Wall
LDFLAGS = -lm

# Arquivos fonte
KMEANS_OMP_GPU_SRC = kmeans_openmp_gpu.c
SPOTIFY_OMP_GPU_SRC = spotify_kmeans_openmp_gpu.c
CUDA_SRC = spotify_kmeans_cuda.cu

# Executáveis
EXE_OMP_GPU = spotify_kmeans_omp_gpu
EXE_CUDA = spotify_kmeans_cuda

################################################################################
# ALVOS
################################################################################

# Alvo padrão: compila ambas as versões
all: omp_gpu cuda
	@echo ""
	@echo "=========================================="
	@echo "Compilação concluída!"
	@echo "=========================================="
	@echo "Executáveis gerados:"
	@echo "  - $(EXE_OMP_GPU)  (OpenMP)"
	@echo "  - $(EXE_CUDA)  (CUDA)"
	@echo ""


omp_gpu: $(EXE_OMP_GPU)
$(EXE_OMP_GPU): $(KMEANS_OMP_GPU_SRC) $(SPOTIFY_OMP_GPU_SRC) kmeans.h
	@echo "Tentando compilar versão GPU com: $(CC)"
	$(CC) $(CFLAGS_BASE) -fopenmp $(KMEANS_OMP_GPU_SRC) $(SPOTIFY_OMP_GPU_SRC) -o $(EXE_OMP_GPU) $(LDFLAGS)

cuda: $(EXE_CUDA)
$(EXE_CUDA): $(CUDA_SRC)
	@echo "Compilando versão CUDA com: $(NVCC)"
	$(NVCC) -O3 $(CUDA_SRC) -o $(EXE_CUDA)
	
clean:
	@echo "Limpando arquivos compilados..."
	@rm -f *.o $(EXE_OMP_GPU) $(EXE_CUDA)
	@echo "✓ Limpeza concluída"

help:
	@echo "=========================================="
	@echo "Makefile - Projeto Spotify K-Means"
	@echo "=========================================="
	@echo ""
	@echo "Alvos disponíveis:"
	@echo "  make all         - Compila todas as versões (OpenMP + CUDA)"
	@echo "  make omp_gpu     - Compila apenas versão OpenMP"
	@echo "  make cuda        - Compila apenas versão CUDA"
	@echo "  make clean       - Remove arquivos compilados"
	@echo "  make help        - Mostra esta ajuda"
	@echo ""
	@echo "Exemplos de execução:"
	@echo ""
	@echo "OpenMP:"
	@echo "  ./$(EXE_OMP_GPU) $(DATASET) 10 saida.txt 1"
	@echo ""
	@echo "CUDA:"
	@echo "  Padrão: ./$(EXE_CUDA) input.csv <k> out.tsv"
	@echo "  Custom: ./$(EXE_CUDA) input.csv <k> out.tsv <threads> <blocks>"
	@echo ""

.PHONY: all omp_gpu cuda clean help
