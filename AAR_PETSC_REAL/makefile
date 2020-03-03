all:aar

CPPFLAGS = -I ./src -g -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread 
 
SOURCEC = ./main.c system.c AAR_Real.c
OBJSC = ./main.o system.o AAR_Real.o 
LIBBASE = ./AAR_Real

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

aar: ${OBJSC} chkopts
	${CLINKER} -o ${LIBBASE} ${OBJSC} ${PETSC_LIB} 
	${RM} ${OBJSC}



