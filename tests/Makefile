ifeq ($(DEBUG),1)
OPTIMIZE_FLAGS       = -O3 -DDO_DEBUG -g
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

FAST_ROOT                 = ../

INCLUDES             = -I$(FAST_ROOT) -I./include
EXECS                = unit_test perf_test
TARGET_OBJ_UNIT      = basic_unit_test.o
TARGET_OBJ_PERF      = 
DRIVER_OBJ           = test_driver.o


.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
%.o: %.cpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#executables
unit_test: $(TARGET_OBJ_UNIT) $(DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
perf_test: $(TARGET_OBJ_PERF) $(DRIVER_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)
	

clean:
	rm -rf *.o *~

cleanall: clean
	rm -rf $(EXECS)
	rm -fr $(TARGET_OBJ_BATCH) $(TARGET_OBJ_STREAM) *.d
#	rm -rf *.dSYM *.dot *.png
