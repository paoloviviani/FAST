include config.mk

ifeq ($(DEBUG),1)
OPTIMIZE_FLAGS       = -O0 -DDO_DEBUG -g
else
OPTIMIZE_FLAGS       = -O3 -g0
endif

CXX                  = g++
LINK_OPT             = 
VERSION              = 
CCFLAGS              = -Wall
CFLAGS               = -std=c++11
LDFLAGS              = 
LIBS                 = -lpthread

#Use #include "gam/include/xx.hpp" to avoid potential confusion with MxNet includes
INCLUDES             = -I$(FAST_ROOT)/fast -I$(FAST_ROOT)/tests/catch2 -I$(FAST_ROOT)/3rdparty/gam
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

#basic unit test executable
unit_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *~ $(TARGET_OBJ_UNIT)

cleanall: clean
	rm -rf $(EXECS)
	rm -rf $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ) *.d

