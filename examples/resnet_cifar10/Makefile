include ../../make/config.mk
include ../../make/Makefile.global

EXECS      = resnetGrid resnetRing resnetGrid_full resnetRing_full

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
resnetGrid.o: $(FAST_ROOT)/src/2D_torus_grid.cpp resnet_epoch.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_epoch.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnetRing.o: $(FAST_ROOT)/src/1D_ring.cpp resnet_epoch.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_epoch.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnetGrid_full.o: $(FAST_ROOT)/src/2D_torus_grid.cpp resnet_epoch.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_full.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnetRing_full.o: $(FAST_ROOT)/src/1D_ring.cpp resnet_epoch.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include resnet_full.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
resnetGrid: resnetGrid.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnetRing: resnetRing.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnetGrid_full: resnetGrid_full.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnetRing_full: resnetRing_full.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)


clean:
	rm *.o

cleanall:
	rm -rf $(EXECS)
	rm -rf *o
