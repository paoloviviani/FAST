include ../../make/config.mk
include ../../make/Makefile.global

EXECS                = mnist2D
TARGET_OBJ           = mnist2D.o

.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: $(EXECS)

#objects
mnist2D.o: $(FAST_ROOT)/src/2D_torus.cpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp 
	$(CXX) $(INCLUDES) -include mnist_logic.hpp $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
mnist2D: $(TARGET_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *~ $(TARGET_OBJ)

cleanall: clean
	rm -rf bin/$(EXECS)
	rm -rf $(TARGET_OBJ) *.d *.btr
