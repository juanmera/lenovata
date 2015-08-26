GCCGOFLAGS=-static-libgcc -O3

lenovata: clean intelsha/intelsha.o cuda/lenovo.o
	go build  -compiler gccgo -gccgoflags="${GCCGOFLAGS}"
test: intelsha/intelsha.o
	go test -v -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
bench: intelsha/intelsha.o
	go test -v -bench . -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
intelsha/intelsha.o: intelsha/intelsha.asm
	yasm -f x64 -f elf64 -X gnu -g null -D LINUX -o intelsha/intelsha.o intelsha/intelsha.asm
cuda/lenovo.o: cuda/sha256.cu cuda/sha256.h cuda/lenovo.cu cuda/lenovo.h
	# nvcc -c -arch compute_50 --gpu-code sm_50  cuda/lenovo.cu -o cuda/lenovo.o
	nvcc -c -arch compute_50 --gpu-code sm_50 -O3 cuda/lenovo.cu -o cuda/lenovo.o
clean:
	# rm -f intelsha/intelsha.o
	rm -f cuda/sha256.o cuda/lenovo.o
	rm -f lenovata
