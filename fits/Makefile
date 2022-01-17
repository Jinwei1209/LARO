CXX=g++
MKLDIR=/opt/intel/mkl
MKLINCFLAGS=-I$(MKLDIR)/include -I$(MKLDIR)/include/fftw
MKLLDFLAGS=-L$(MKLDIR)/lib/intel64 -L$(MKLDIR)/lib
MKLLIBS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
USRFLAGS=-L/usr/local/lib
LIBCONFIGFLAGS=-I/usr/local/include
CXXFLAGS = -W -Wall -O3 -g -gstabs $(MKLINCFLAGS) $(LIBCONFIGFLAGS) $(MKLLDFLAGS) $(MKLLIBS)

TARGET = main

.PHONY:	all
all:	$(TARGET)

main: main.o angle.o any_matrix.o fit_mm.o
		$(CXX) $(CXXFLAGS) -o main main.o angle.o any_matrix.o fit_mm.o

angle.o: angle.cc angle.h
		$(CXX) $(CXXFLAGS) -c angle.cc

any_matrix.o: any_matrix.c any_matrix.h debug.h stringmacro.h
		$(CXX) $(CXXFLAGS) -c any_matrix.c

fit_mm.o: fit_mm.cc fit_mm.h angle.h any_matrix.h util_blas.h Util.hh
		$(CXX) $(CXXFLAGS) -c fit_mm.cc

main.o: main.cc
		$(CXX) $(CXXFLAGS) -c main.cc

clean:
	rm -f *.o $(TARGET)




