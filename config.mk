FAST_ROOT				=	$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
#error, info, debug
DEBUG					=	debug
MXNET_LIB_DIR			=	$(FAST_ROOT)/3rdparty/mxnet/lib
LIBFABRIC_ROOT			=	
USE_GAM					=	1
