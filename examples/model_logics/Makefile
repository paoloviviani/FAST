include ../../make/config.mk
include ../../make/Makefile.global

EXECS      = mnist2D resnet2D

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
mnist2D.o: $(FAST_ROOT)/src/2D_torus_grid.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnet.o: $(FAST_ROOT)/src/2D_torus_grid.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -DDATA_TAG=data -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
mnist2D: mnist2D.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnet2D: resnet.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.o

cleanall: clean
	rm -rf $(EXECS)
	rm -rf *o
