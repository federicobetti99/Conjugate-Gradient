CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std c++11
LDFLAGS+=-lm $(CXXFLAGS)

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

cg.o: cg.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f cgsolver *.o *~
