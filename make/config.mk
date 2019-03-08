FAST_ROOT			=	$(dir $(realpath $(lastword $(MAKEFILE_LIST))))..

# Log and optimization level error (-O3, no logs), info (-O3, info messages), debug (-O0 -g, debug messages), all_debug (-O0 -g, debug messages from GAM backend too)
DEBUG				=	debug

# Location of build dependencies
MXNET_LIB_DIR		=	/opt/incubator-mxnet/lib
LIBFABRIC_ROOT		=	/opt/libfabric
#MXNET_LIB_DIR		=	/home/pviviani/pviviani/opt/magnus/mxnet/lib
#LIBFABRIC_ROOT		=	/home/pviviani/pviviani/opt/magnus/libfabric
