include ../../make/config.mk
include ../../make/Makefile.global

EXECS      = mnist2Dgrid resnet2Dgrid mnist2nodes resnet2nodes

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
mnist2Dgrid.o: $(FAST_ROOT)/src/2D_torus_grid.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnet2Dgrid.o: $(FAST_ROOT)/src/2D_torus_grid.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
mnist2nodes.o: $(FAST_ROOT)/src/2_workers.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnet2nodes.o: $(FAST_ROOT)/src/2_workers.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
mnist2Dgrid: mnist2Dgrid.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnet2Dgrid: resnet2Dgrid.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
mnist2nodes: mnist2nodes.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnet2nodes: resnet2nodes.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.o

cleanall:
	rm -rf $(EXECS)
	rm -rf *o
