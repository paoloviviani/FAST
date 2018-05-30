include config.mk

ifeq ($(DEBUG),1)
OPTIMIZE_FLAGS       = -O0 -DDO_DEBUG -g
else
OPTIMIZE_FLAGS       = -O3 -g0
endif

CXX                  = g++
LINK_OPT             = 
VERSION              = 
CCFLAGS              = -Wall #-DTRACE_FASTFLOW
CFLAGS               = -std=c++11
LDFLAGS              = 
LIBS                 = -lpthread

FAST_ROOT                 = ../include

INCLUDES             = -I$(FAST_ROOT) -I./catch2
EXECS                = unit_test
TARGET_OBJ_UNIT      = basic_unit_test.o
TEST_DRIVER_OBJ      = test_driver.o

ifeq ($(USE_MXNET),1)
TARGET_OBJ_UNIT      += mxnet_tests/mxnet_unit_test.o
INCLUDES             += -I$(MXNET_INCLUDE_DIR)
LIBS                 += -L$(MXNET_LIB_DIR) -lmxnet
endif

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#executables
unit_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
	
ifeq ($(USE_MXNET),1)
mxnet_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
endif

clean:
	rm -rf *.o *~ $(TARGET_OBJ_UNIT)

cleanall: clean
	rm -rf $(EXECS)
	rm -rf $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ) *.d

