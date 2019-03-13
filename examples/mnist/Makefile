include ../../make/config.mk
include ../../make/Makefile.global

EXECS      = mnist2Dgrid mnistRing

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
mnist2Dgrid.o: $(FAST_ROOT)/src/2D_torus_grid.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
mnistRing.o: $(FAST_ROOT)/src/1D_ring.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
mnist2Dgrid: mnist2Dgrid.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
mnistRing: mnistRing.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
clean:
	rm *.o

cleanall:
	rm -rf $(EXECS)
	rm -rf *o
