test: main
	./main
main: sha256go.o main.go
	go build -compiler gccgo -gccgoflags='-O3' main.go
sha256go.o: common.cu sha256.cu sha256go.cu common.cuh sha256.cuh sha256go.cuh
	nvcc -c -arch compute_50 --gpu-code sm_50 common.cu sha256.cu sha256go.cu
