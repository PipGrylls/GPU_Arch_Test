#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(HOME)
bindir     = $(prefix)/bin

# Define objects in dependency order
OBJECTS   = kernals.o d_main.o main.o

CC    = gcc
NVCC  = nvcc
LD     = nvcc
CFLAGS =  -O3 
# Targeting Compute 5.X to 9.0, disable as required
NVFLAGS = -O3 -gencode arch=compute_50,code=sm_50 \
			  -gencode arch=compute_52,code=sm_52 \
			  -gencode arch=compute_53,code=sm_53 \
			  -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
			  -gencode arch=compute_62,code=sm_62 \
			  -gencode arch=compute_70,code=sm_70 \
			  -gencode arch=compute_72,code=sm_72 \
			  -gencode arch=compute_75,code=sm_75 \
			  -gencode arch=compute_80,code=sm_80 \
			  -gencode arch=compute_86,code=sm_86 \
			  -gencode arch=compute_87,code=sm_87 \
			  -gencode arch=compute_89,code=sm_89 \
			  -gencode arch=compute_90,code=sm_90 --generate-line-info

.PRECIOUS: %.o
.PHONY:  clean

all : GPU_Arch_Test

%: %.o
%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu %.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<


GPU_2DIsing :  $(OBJECTS) ising.cu

	$(LD) -o $(bindir)/GPU_Arch_Test $(OBJECTS) main.cu $(NVFLAGS) 

clean : 
	rm -f *.mod *.d *.il *.o work.*
	rm -f $(bindir)/GPU_2DIsing_refac