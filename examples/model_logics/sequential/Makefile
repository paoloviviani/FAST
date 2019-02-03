include ../../../make/config.mk
include ../../../make/Makefile.global

EXECS      = mnistSequential resnetSequential

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
mnistSequential.o: $(FAST_ROOT)/src/sequential.cpp mnist_logic.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<
resnetSequential.o: $(FAST_ROOT)/src/sequential.cpp resnet_cifar10.hpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -DDATA_TAG=data -include resnet_cifar10.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

# Execs
mnistSequential: mnistSequential.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
resnetSequential: resnetSequential.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.o

cleanall:
	rm -rf $(EXECS)
	rm -rf *o
