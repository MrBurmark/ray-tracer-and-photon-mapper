
#include "variables.h"
#include <random>

void shoot_photons(scene* scn);

void interact_photons(scene* scn, hit hitt, vec3 color);

vec3 photon_shade(scene* scn, hit a);

void sse_shoot_photons(sse_scene* scn);

void sse_interact_photons(sse_scene* scn, ssehit& hitt, __m128& color, mt19937& generator, uniform_real_distribution<float>& distribution);

__m128 sse_photon_shade(sse_scene* scn, ssehit& a) ;