include ../../make/config.mk
include ../../make/Makefile.global

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

