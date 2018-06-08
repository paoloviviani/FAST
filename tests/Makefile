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

CCFLAGS              = -Wall #-Wextra -Wstrict-aliasing -pedantic
CFLAGS               = -std=c++11
LDFLAGS              = 
LIBS                 = -lpthread `pkg-config --libs libfabric`

INCLUDES             = -I$(FAST_ROOT)/fast -I$(FAST_ROOT)/tests/catch2 -I$(FAST_ROOT)/3rdparty/gam/gam/include `pkg-config --cflags libfabric`
EXECS                = unit_test
TARGET_OBJ_UNIT      = basic_unit_test.o
TEST_DRIVER_OBJ      = test_driver.o

#MxNet dependecies
INCLUDES             += -I$(MXNET_INCLUDE_DIR)
LIBS                 += -L$(MXNET_LIB_DIR) -lmxnet

ifeq ($(TEST_MXNET),1)
TARGET_OBJ_UNIT      += mxnet_tests/mxnet_unit_test.o
endif

ifeq ($(USE_GAM),1)
CCFLAGS              += -DUSE_GAM
endif

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp $(FAST_ROOT)/fast/fast/*.hpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
unit_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *~ $(TARGET_OBJ_UNIT)

cleanall: clean
	rm -rf $(EXECS)
	rm -rf $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ) *.d

