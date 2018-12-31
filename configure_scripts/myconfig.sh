#!/bin/bash

export LEVEL1_DCACHE_SIZE=32768
export LEVEL1_DCACHE_LINESIZE=64
export LEVEL1_DCACHE_ASSOC=8

export MPETSC_DIR=$PWD
export MPETSC_ARCH='sclang' \

./configure \
	CC=clang \
        CXX=clang++  \
        PETSC_DIR=MPETSC_DIR \
        PETSC_ARCH=MPETSC_ARCH \
	CCFLAGS=$CCFLAGS \
	CXXFLAGS=$CXXFLAGS \
	--disable-mpi \
        --with-shared-libraries=0 \
        --enable-fortran \
        --with-avx512-kernels=1 \
        --with-memalign=64 \
        --known-level1-dcache-size=$LEVEL1_DCACHE_SIZE \
        --known-level1-dcache-linesize=$LEVEL1_DCACHE_LINESIZE \
        --known-level1-dcache-assoc=$LEVEL1_DCACHE_ASSOC \
        --with-scalar-type=real \
        --with-debugging=1 \
        make PETSC_DIR=$MPETSC_DIR PETSC_ARCH=$MPETSC_ARCH -j12 all

        # --download-mpich \
        # --with-slepc-dir=$SLEPC_DIR \
        # --with-slepc-lib=$SLEPC_LIB \
        # --with-slepc-include=$SLEPC_INCLUDE \

        # --with-hdf5=1 \
        # --with-superlu_dist=1 \
        # --with-hypre=1 \


        # --download-metis=1 \
        # --with-metis=1 \
        # --download-parmetis=1 \
        # --with-parmetis=1 \

#make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH -j12 install
#make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH -j12 check

# sourcePETSC_DIR=$HOME/.local/petsc_v393/
# sourcePETSC_ARCH=arch-linux-opt
        # --download-moab=1 \
        # --with-moab=1 \
        # --download-petsc4py \
        # --with-petsc4py=1 \
        # --download-mpi4py \
        # --with-mpi4py=1 \
        # --with-pnetcdf=1
        #--with-ptscotch=1
        #--with-zlib=1
        #--with-parmetis=1
        # --with-hypre-lib=$HYPRE/lib \
 #        --with-mpi=1 \
	# --known-mpi-shared-libraries=1 \
	# --known-mpi-shared=0 \
# python setup.py build
        # --with-zlib-lib="/usr/lib/x86_64-linux-gnu/ruby/2.5.0/zlib.so" \
        # --with-zlib-include="/usr/include" \
        # --with-ptscotch-lib="/usr/lib/x86_64-linux-gnu/libptscotch.so" \
        # --with-parmetis-dir=$PETSC_TPL \
        # --with-fblaslapack-dir=$PETSC_TPL \
        # --with-chaco-dir=$PETSC_TPL  \
        # --with-hypre-dir=$PETSC_TPL \
        # --with-metis-dir=$PETSC_TPL \
        # --with-triangle-dir=$PETSC_TPL \
        # --with-superlu-dir=$PETSC_TPL \
        # --with-superlu_dist-dir=$PETSC_TPL \
        # --with-scalapack-dir=$PETSC_TPL \
        # --with-moab-dir=$PETSC_TPL \
        # --with-hdf5-dir=$PETSC_TPL \
        # --with-mumps-dir=$PETSC_TPL \
        # --with-fftw-dir=$PETSC_TPL \
