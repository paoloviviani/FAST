include config.mk

OPTIMIZE_FLAGS       = -O3 -DLOGLEVEL_INFO -g0
ifeq ($(DEBUG),debug)
OPTIMIZE_FLAGS       = -O0 -DLOGLEVEL_DEBUG -g3
endif
ifeq ($(DEBUG),info)
OPTIMIZE_FLAGS       = -O3 -DLOGLEVEL_INFO -g0
endif
ifeq ($(DEBUG),error)
OPTIMIZE_FLAGS       = -O3 -DLOGLEVEL_ERROR -g0
endif

CXX                  = g++
LINK_OPT             = 
VERSION              = 

CCFLAGS              = -Wall -Wfatal-errors -Wno-unused-variable #-Wextra -Wstrict-aliasing -pedantic
CFLAGS               = -std=c++11
LDFLAGS              = 
LIBS                 = -lpthread

INCLUDES             = -I$(FAST_ROOT)/fast -I$(FAST_ROOT)/tests/catch2 -I$(FAST_ROOT)/3rdparty/gam/gam/include
TARGET_OBJ_CPU       = mlp_cpu.o
TARGET_OBJ_GPU       = mlp_gpu.o
EXECS                = mnist_cpu
ifeq ($(USE_CUDA),1)
EXECS                += mnist_gpu
endif

#Libfabric dependency
ifeq ($(LIBFABRIC_LIB_DIR),)
LIBS                 += -lfabric
else
LIBS                 += -L$(LIBFABRIC_LIB_DIR) -lfabric
endif
ifneq ($(LIBFABRIC_INCLUDE_DIR),)
INCLUDES             += -I$(LIBFABRIC_INCLUDE_DIR)
endif

#MxNet dependency
INCLUDES             += -I$(MXNET_INCLUDE_DIR)
LIBS                 += -L$(MXNET_LIB_DIR) -lmxnet
CCFLAGS              += -DUSE_GAM

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp $(FAST_ROOT)/fast/fast/*.hpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
mnist_cpu: $(TARGET_OBJ_CPU)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
mnist_gpu: $(TARGET_OBJ_GPU)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

cleanbtr:
	rm -rf *.btr
clean:
	rm -rf *.o *~ $(TARGET_OBJ_CPU) $(TARGET_OBJ_GPU)

cleanall: clean
	rm -rf $(EXECS)
	rm -rf  $(TARGET_OBJ_CPU) $(TARGET_OBJ_GPU) *.d *.btr

