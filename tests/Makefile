include ../make/config.mk
include ../make/Makefile.global

INCLUDES             += -I$(FAST_ROOT)/tests/catch2

EXECS                = unit_test
EXECS                += gam_unit_test
EXECS                += gff_farm
EXECS                += gff_training_mockup
EXECS                += gff_training_concurrent
EXECS                += gff_training_concurrent_2
EXECS                += gff_training_concurrent_grid
EXECS                += mxnet_aux_test
EXECS                += mxnet_worker_test

TEST_DRIVER_OBJ       = catch2/test_driver.o
TARGET_OBJ            = $(TEST_DRIVER_OBJ)

TARGET_OBJ_UNIT       = basic_unit_tests/basic_unit_test.o
TARGET_OBJ            += $(TARGET_OBJ_UNIT)

TARGET_OBJ_GAM        = gam_tests/gam_unit_test.o
TARGET_OBJ            += $(TARGET_OBJ_GAM)

TARGET_OBJ_GFF_FARM   = gff_tests/basic_farm.o
TARGET_OBJ            += $(TARGET_OBJ_GFF_FARM)

TARGET_OBJ_TRAIN_MOCK = gff_tests/gff_training_mockup.o
TARGET_OBJ            += $(TARGET_OBJ_TRAIN_MOCK)

TARGET_OBJ_TRAIN_PIPE = gff_tests/gff_training_concurrent.o
TARGET_OBJ            += $(TARGET_OBJ_TRAIN_PIPE)

TARGET_OBJ_TRAIN_MULTIPIPE = gff_tests/gff_training_concurrent_2.o
TARGET_OBJ            += $(TARGET_OBJ_TRAIN_MULTIPIPE)

TARGET_OBJ_TRAIN_GRID = gff_tests/gff_training_concurrent_grid.o
TARGET_OBJ            += $(TARGET_OBJ_TRAIN_GRID)

TARGET_OBJ_MXNET_AUX  = mxnet_test/mxnet_aux_test.o
TARGET_OBJ            += $(TARGET_OBJ_MXNET_AUX)

TARGET_OBJ_MXNET_TEST = mxnet_test/mxnet_worker_test.o
TARGET_OBJ            += $(TARGET_OBJ_MXNET_TEST)


.PHONY: all clean cleanall
.SUFFIXES: .c .cpp .o

all: bin $(EXECS)

bin:
	if [ ! -d "./bin" ];then     \
	        mkdir bin;           \
	fi

#objects
%.o: %.cpp $(FAST_ROOT)/include/fast/*.hpp $(FAST_ROOT)/include/fast/workers/mxnet_worker.hpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CFLAGS) $(OPTIMIZE_FLAGS) -c -o $@ $<

#basic unit test executable
unit_test: $(TARGET_OBJ_UNIT) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gam_unit_test: $(TARGET_OBJ_GAM) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_farm: $(TARGET_OBJ_GFF_FARM) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_training_mockup: $(TARGET_OBJ_TRAIN_MOCK) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)	
gff_training_concurrent: $(TARGET_OBJ_TRAIN_PIPE) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_training_concurrent_2: $(TARGET_OBJ_TRAIN_MULTIPIPE) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
gff_training_concurrent_grid: $(TARGET_OBJ_TRAIN_GRID) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
mxnet_aux_test: $(TARGET_OBJ_MXNET_AUX) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)
mxnet_worker_test: $(TARGET_OBJ_MXNET_TEST) $(TEST_DRIVER_OBJ)
	$(CXX) $^ -o bin/$@ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o *~ $(TARGET_OBJ)

cleanall: clean
	rm -rf bin/$(EXECS)
	rm -rf $(TARGET_OBJ) *.d *.btr

