#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Compiler Flags
CC    = gcc
NVCC  = nvcc
LD     = nvcc
CFLAGS =  -O3 
# Targeting Compute 5.X to 9.0, disable as required
NVFLAGS = -O3 -gencode arch=compute_50,code=sm_50 \
			  -gencode arch=compute_52,code=sm_52 \
			  -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
			  -gencode arch=compute_70,code=sm_70 \
			  -gencode arch=compute_75,code=sm_75 \
			  -gencode arch=compute_80,code=sm_80 \
			  -gencode arch=compute_86,code=sm_86 \
			  -gencode arch=compute_89,code=sm_89 \
			  -gencode arch=compute_90,code=sm_90 \
			  -rdc=true

# Define objects in dependency order
SRCS = mt19937ar.c kernals.cu d_main.cu
OBJS_C = $(SRCS:.c=.o)
OBJS = $(OBJS_C:.cu=.o)
EXE  = gpu_arch_test


#
# Debug build settings
#
DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJS))
DBGCFLAGS = -g -O0 -DDEBUG
DBGNVFLAGS = -g -G

#
# Release build settings
#
RELDIR = bin
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELEXE = $(DIR)/$(EXE)



.PHONY:  clean
.PRECIOUS: $(RELDIR)/%.o
all : prep release

# Release Rules

release: $(RELEXE)
$(RELEXE) : $(RELOBJS) main.cu
	$(LD) -o $(RELEXE) $(RELOBJS) main.cu $(NVFLAGS) 

$(RELDIR)/%: %.o
$(RELDIR)/%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<
$(RELDIR)/%.o: %.cu %.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

# Debug Rules

debug: $(DBGEXE)

$(DBGEXE): $(DBGOBJS) main.cu
	$(LD) -o $(DBGEXE) $(DBGOBJS) main.cu $(NVFLAGS) $(DBGNVFLAGS)

$(DBGDIR)/%: %.o

$(DBGDIR)/%.o: %.c %.h
	$(CC) $(CFLAGS) $(DBGCFLAGS) -c -o $@ $<

$(DBGDIR)/%.o: %.cu %.h
	$(NVCC) $(NVFLAGS) $(DBGNVFLAGS) -c -o $@ $<

# Extra Rules
debug_clean: 
	rm -f $(DBGDIR)/*.o 
	rm -f $(DBGDIR)/$(EXE)
	mkdir -p $(DBGDIR) 
	make debug

prep :
	@mkdir -p $(DBGDIR) 
	@mkdir -p $(RELDIR)
clean : 
	rm -f $(RELDIR)/*.o $(RELDIR)/$(EXE)
	rm -f $(DBGDIR)/*.o $(DBGDIR)/$(EXE)
	prep
print :
	@echo "SRCS" $(SRCS)
	@echo "OBJS" $(OBJS)
	@echo "DBGOBJS" $(DBGOBJS)
	@echo "RELOBJS" $(RELOBJS)
	