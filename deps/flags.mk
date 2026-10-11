# Shared compiler settings. Apple clang needs Homebrew's libomp for OpenMP.
BREW     ?= $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
CXX      ?= clang++
UNAME_M  := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  ARCHFLAGS ?= -mcpu=native
else
  ARCHFLAGS ?= -march=native
endif
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra $(ARCHFLAGS)
ifeq ($(shell uname -s),Darwin)
  OMP_CFLAGS  ?= -Xpreprocessor -fopenmp -I$(BREW)/opt/libomp/include
  OMP_LDFLAGS ?= -L$(BREW)/opt/libomp/lib -lomp
else
  OMP_CFLAGS  ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif
BLAS_CFLAGS  ?= -I$(BREW)/opt/openblas/include
BLAS_LDFLAGS ?= -L$(BREW)/opt/openblas/lib -lopenblas
