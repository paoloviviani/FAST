OBJDIR := obj
SRCDIR := src

CC := g++
# silent by default
ifeq ($(VERBOSE),1)
    SILENCER := 
else
    SILENCER := @
endif
 
# pass this environment variable to the C source
ifeq ($(DEBUG),1)
    CFLAGS += -DDEBUG -g
    WARNINGS := -Wall -Wextra 
endif
 
# typical way to list files and build full paths
# list the sources, not the object files (nor includes)
_SRCS := basic.cpp
SRCS := $(patsubst %,$(SRCDIR)/%,$(_SRCS))
OBJS := $(patsubst %,$(OBJDIR)/%,$(_SRCS:cpp=o))
 
# generate phony deps during compilation
CFLAGS += -std=c++11
# can't include the deps before the first, default target has appeared!
 
# "all" is the classical default target
all: main
 
createdir:
        $(SILENCER)mkdir -p $(OBJDIR)
 
main: $(OBJS) 
        #@echo " LINK $^"
        $(SILENCER)$(CC) $(CFLAGS) -o $@ $^ 
 
# put object and dependency files away from the sources
# create the dir before building into it
$(OBJDIR)/%.o: $(SRCDIR)/%.c | createdir
        #@echo " CC $<"
        $(SILENCER)$(CC) $(CFLAGS) -c -o $@ $< 
 
clean:
        $(SILENCER)$(RM) -f *~ core main
        $(SILENCER)$(RM) -r $(OBJDIR)
 
.PHONY: clean all
