
#ifndef SHADER_H
#define SHADER_H


#include "variables.h"

vec3 shade(scene* scn, hit a) ;

__m128 sse_shade(sse_scene* scn, ssehit& a) ;

__m128 sse_shade_kd(sse_scene* scn, ssehit& a) ;

#endif