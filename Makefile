GCCGOFLAGS=-static-libgcc -O3 -m64 ${PWD}/intelsha/intelsha.o

lenovata: clean intelsha/intelsha.o
	go build  -compiler gccgo -gccgoflags="${GCCGOFLAGS}"
test: intelsha/intelsha.o
	go test -v -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
bench: intelsha/intelsha.o
	go test -v -bench . -compiler gccgo -gccgoflags="${GCCGOFLAGS}" ./...
intelsha/intelsha.o: intelsha/intelsha.asm
	yasm -f x64 -f elf64 -X gnu -g null -D LINUX -o intelsha/intelsha.o intelsha/intelsha.asm
clean:
	rm -f intelsha/intelsha.o
	rm -f lenovata
