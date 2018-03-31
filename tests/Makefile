CXX                  = g++
LINK_OPT             = 
VERSION              = 
OPTIMIZE_FLAGS       = -O3 -g0 -std=c++11
CCFLAGS              = -Wall #-DTRACE_FASTFLOW 
CFLAGS               =
LDFLAGS              = 
LIBS                 = -lpthread

FAST_ROOT                 = ..

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
	$(CXX) $(INCLUDES) $(CCFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

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
