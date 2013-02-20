
#ifndef INTERSECT_H
#define INTERSECT_H

#include "variables.h"

hit intersect (scene* scn, vec3 S, vec3 D, unsigned int depth, float alpha, float b_amt);

hit internal_intersect (scene* scn, vec3 S, vec3 D, unsigned int depth, float alpha, float b_amt);

ssehit sse_intersect (sse_scene* scn, __m128 S, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt);

bool sse_shadow_intersect (sse_scene* scn, __m128 S, __m128 D, float ldistance, float b_amt);

ssehit sse_internal_intersect (sse_scene* scn, __m128 S, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt);

ssehit sse_intersect_kd (sse_scene* scn, __m128 S, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) ;

bool sse_shadow_intersect_kd (sse_scene* scn, __m128 S, __m128 D, float ldistance, float b_amt) ;

ssehit sse_internal_intersect_kd (sse_scene* scn, __m128 S, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) ;

#endif
