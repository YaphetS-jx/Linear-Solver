all:system

CPPFLAGS = -I ./src -g -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
# CPPFLAGS = -I ./src -g -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DDEBUG
 
SOURCEC = ./main.c system.c AAR_Real.c PGR_Real.c PL2R_Real.c 
OBJSC = ./main.o system.o AAR_Real.o PGR_Real.o PL2R_Real.o 
LIBBASE = ./petsc_real

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

system: ${OBJSC} chkopts
	${CLINKER} -o ${LIBBASE} ${OBJSC} ${PETSC_LIB} 
	${RM} ${OBJSC}



