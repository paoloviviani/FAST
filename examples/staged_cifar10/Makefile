include ../../make/config.mk
include ../../make/Makefile.global

EXECS      = resnetGrid resnetRing

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
resnetGrid.o: $(FAST_ROOT)/src/2D_torus_grid.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnetRing.o: $(FAST_ROOT)/src/1D_ring.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
resnetGrid: resnetGrid.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnetRing: resnetRing.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.o

cleanall:
	rm -rf $(EXECS)
	rm -rf *o
