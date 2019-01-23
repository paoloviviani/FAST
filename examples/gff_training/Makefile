include config.mk

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

#Libfabric dependency
ifeq ($(LIBFABRIC_ROOT),)
LIBS                 += -lfabric
else
LIBS                 += -L$(LIBFABRIC_ROOT)/lib -lfabric
endif
ifneq ($(LIBFABRIC_ROOT),)
INCLUDES             += -I$(LIBFABRIC_ROOT)/include
endif

INCLUDES             = -I$(FAST_ROOT)/fast -I$(FAST_ROOT)/tests/catch2 -I$(FAST_ROOT)/3rdparty/fastflow/ -I$(FAST_ROOT)/3rdparty/fastflow/ff/d/gff -I$(FAST_ROOT)/3rdparty/gam/include -I$(FAST_ROOT)/3rdparty/armadillo
EXECS                = basic_gff_train
TARGET_OBJ           = basic_gff_train.o

#MxNet dependency
MXNET_INCLUDE_DIR		=	$(FAST_ROOT)/3rdparty/mxnet/include
INCLUDES             += -I$(MXNET_ROOT)/include
LIBS                 += -L$(MXNET_ROOT)/lib -lmxnet
CCFLAGS              += -DUSE_GAM

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp $(FAST_ROOT)/fast/fast/*.hpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
basic_gff_train: $(TARGET_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

cleanbtr:
	rm -rf *.btr
clean:
	rm -rf *.o *~ $(TARGET_OBJ)

cleanall: clean
	rm -rf bin/$(EXECS)
	rm -rf $(TARGET_OBJ) *.d *.btr

