CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std c++11 -I${OPENBLAS_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o

all: cgsolver

%.o:%.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f cgsolver *.o *~
