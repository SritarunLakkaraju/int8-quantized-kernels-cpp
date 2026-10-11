include deps/flags.mk

all: bench_qgemm quant_mnist

bench_qgemm: bench_qgemm.cpp qgemm.h deps/gemm.h deps/util.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $< -o $@ $(OMP_LDFLAGS)

quant_mnist: quant_mnist.cpp qgemm.h deps/engine.h deps/conv.h deps/gemm.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $< -o $@ $(OMP_LDFLAGS)

# The test images ship gzip-compressed; unpack them on first use.
model/test_images.bin: model/test_images.bin.gz
	gunzip -k -c $< > $@

run: bench_qgemm quant_mnist model/test_images.bin
	mkdir -p results
	./bench_qgemm results/qgemm.csv | tee results/qgemm.txt
	./quant_mnist model results/quant.json | tee results/quant.txt

clean:
	rm -f bench_qgemm quant_mnist

.PHONY: all run clean
