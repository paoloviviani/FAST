include ../config.mk

OPTIMIZE_FLAGS       = -O3 -DLOGLEVEL_INFO -g0
ifeq ($(DEBUG),debug)
OPTIMIZE_FLAGS       = -O0 -DLOGLEVEL_DEBUG -g3 -DGFF_LOG -DGAM_DBG
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

INCLUDES             = -I$(FAST_ROOT)/fast -I$(FAST_ROOT)/tests/catch2 -I$(FAST_ROOT)/3rdparty/fastflow/ -I$(FAST_ROOT)/3rdparty/fastflow/ff/d/gff -I$(FAST_ROOT)/3rdparty/gam/include
EXECS                = unit_test gam_unit_test gff_farm gff_all_reduce gff_all_reduce_multi gff_all_reduce_vector mxnet_worker_test
TARGET_OBJ_UNIT      = basic_unit_tests/basic_unit_test.o
TEST_DRIVER_OBJ      = catch2/test_driver.o

#Libfabric dependency
ifeq ($(LIBFABRIC_LIB_DIR),)
LIBS                 += -lfabric
else
LIBS                 += -L$(LIBFABRIC_LIB_DIR) -lfabric
endif

#MxNet dependency
MXNET_INCLUDE_DIR		=	$(FAST_ROOT)/3rdparty/mxnet/include
INCLUDES             += -I$(MXNET_INCLUDE_DIR)
LIBS                 += -L$(MXNET_LIB_DIR) -lmxnet
CCFLAGS              += -DMXNET_TENSOR

ifeq ($(USE_GAM),1)
CCFLAGS              += -DUSE_GAM
TARGET_OBJ_GAM       = gam_tests/gam_unit_test.o 
TARGET_OBJ_GFF_FARM  = gff_tests/basic_farm.o
TARGET_OBJ_GFF_ALL   = gff_tests/all_reduce.o
TARGET_OBJ_GFF_MULTI = gff_tests/all_reduce_multi.o
TARGET_OBJ_GFF_VECT = gff_tests/all_reduce_vector.o
TARGET_OBJ_MXNET_TEST = mxnet_test/mxnet_worker_test.o
endif

TARGET_OBJ = $(TARGET_OBJ_UNIT) $(TARGET_OBJ_GAM)  $(TARGET_OBJ_GFF_FARM) $(TARGET_OBJ_GFF_ALL) $(TARGET_OBJ_GFF_MULTI) $(TARGET_OBJ_MXNET_TEST) $(TEST_DRIVER_OBJ)

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp $(FAST_ROOT)/fast/fast/*.hpp $(FAST_ROOT)/fast/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
unit_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gam_unit_test: $(TARGET_OBJ_GAM) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_farm: $(TARGET_OBJ_GFF_FARM) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_all_reduce: $(TARGET_OBJ_GFF_ALL) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_all_reduce_multi: $(TARGET_OBJ_GFF_MULTI) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_all_reduce_vector: $(TARGET_OBJ_GFF_VECT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
mxnet_worker_test: $(TARGET_OBJ_MXNET_TEST) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)

cleanbtr:
	rm -rf *.btr
clean:
	rm -rf *.o *~ $(TARGET_OBJ_UNIT)

cleanall: clean
	rm -rf bin/$(EXECS)
	rm -rf $(TARGET_OBJ) *.d *.btr

