#ifndef TOOLS_H
#define TOOLS_H

#include "system.h"

void GetLocalVector(DM da, Vec res, Vec *localv, PetscInt *blockinfo, PetscInt Np, PetscScalar *local, PetscScalar ****r);

void RestoreGlobalVector(DM da, Vec global_v, Vec local_v, PetscInt *blockinfo, PetscScalar *local, PetscScalar ****r);

#endif