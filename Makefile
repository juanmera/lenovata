GCCGOFLAGS=-static-libgcc -O3 -m64 ${PWD}/intelsha/intelsha.o

lenovata: clean intelsha/intelsha.o cuda/mySha256.o
	go build  -compiler gccgo -gccgoflags="${GCCGOFLAGS}"
test: intelsha/intelsha.o
	go test -v -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
bench: intelsha/intelsha.o
	go test -v -bench . -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
intelsha/intelsha.o: intelsha/intelsha.asm
	yasm -f x64 -f elf64 -X gnu -g null -D LINUX -o intelsha/intelsha.o intelsha/intelsha.asm
cuda/mySha256.o: cuda/mySha256.cu cuda/mySha256.cuh cuda/mySha256.h
	# nvcc -c cuda/mySha256.cu -o cuda/mySha256.o
	nvcc -c -arch compute_50 --gpu-code sm_50 -O3 cuda/mySha256.cu -o cuda/mySha256.o
clean:
	rm -f intelsha/intelsha.o
	rm -f cuda/mySha256.o
	rm -f lenovata
