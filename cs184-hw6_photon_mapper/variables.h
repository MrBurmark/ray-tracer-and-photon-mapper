
#ifndef VARIABLES_H
#define VARIABLES_H

#include <vector>
#include <string>
#include <omp.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "matrix_vector.h"
#include "aligned_array.h"

typedef glm::mat3 mat3 ;
typedef glm::mat4 mat4 ; 
typedef struct mat4_ex {
	mat4 mat ;
	bool non_uniform_scale ;
} mat4_ex;
typedef glm::vec3 vec3 ; 
typedef glm::vec4 vec4 ; 
const float pi = 3.14159265f ;
const unsigned int INFint = 0x7f800000 ; 
const float INF = *((float*)&INFint); 
const unsigned int TILE = 4;
const float TILEf = (float)TILE;
const float SMALL = 1e-5f ;
const float air_refracive_index = 1.0f ;

using namespace std;


enum { DIRECTIONAL, POINT };

enum { TRIANGLE, PARALLELOGRAM, SPHERE, ARBITRARY_SPHERE, KD_BOX };

enum { X, Y, Z, LEAF };


typedef struct photon {
	vec3 color;
	vec3 pos;
} photon ;


typedef struct light {
	unsigned int type ;
	vec3 pos ;
	vec3 color ;
} light;

//final
typedef struct rec_light {
	vec3 color ;
	vec3 corner ;
	vec3 width ;
	vec3 height ; 
	vector<vec3> pts;
} rec_light;


typedef struct triangle {
	vec3 pt0;
	vec3 T1;
	vec3 T2;
} triangle;

typedef struct triangle_property {
	vec3 ambient;
	vec3 emission;
	vec3 diffuse;
	float index ;
	vec3 specular;
	float shininess;
	float alpha ;
	vec3 norm0;
	vec3 norm1;
	vec3 norm2;

	vec3 norm;

	vec3 perp0;
	vec3 perp1;
	vec3 perp2;

	vector<photon> photons;

} triangle_property;

typedef struct parallelogram {
	vec3 pt0;
	vec3 T1;
	vec3 T2;
} parallelogram;

typedef struct parallelogram_property {
	vec3 ambient;
	vec3 emission;
	vec3 diffuse;
	float index ;
	vec3 specular;
	float shininess;
	float alpha ;
	vec3 norm0;
	vec3 norm1;
	vec3 norm2;
	vec3 norm3;

	vec3 norm;

	vec3 perp1;
	vec3 perp2;

	vector<photon> photons;

} parallelogram_property;

typedef struct sphere {
	vec3 pos;
	float radius2 ;
} sphere;

typedef struct arbitrary_sphere {
	vec3 pos;
	float radius2 ;
	mat4 inversetransform ;
} arbitrary_sphere;

typedef struct sphere_property {
	vec3 ambient;
	vec3 emission;
	vec3 diffuse;
	float index ;
	vec3 specular;
	float shininess;
	float alpha ;

	vector<photon> photons;

} sphere_property;


typedef struct hit {
	vec3 start ;
	vec3 dir ;
	unsigned int depth ;
	unsigned int type ;
	unsigned int obj ; 
	float u ;
	float v ;
	float t ;
	float alpha ;
} hit;


typedef struct bound_box {
	unsigned int type ;
	unsigned int obj ;
	float top_x ;
	float bot_x ;
	float top_y ;
	float bot_y ;
	float top_z ;
	float bot_z ;
} bound_box;

typedef struct kd_tree_node {
	unsigned int type ;
	union {
	float split ;
	unsigned int* items ; // first number is size, then where the parallelograms start, spheres start, and where the arbitrary spheres start, finally indices into the sets of triangles, parallelograms, spheres and arbitrary spheres
	};
} kd_tree_node;


typedef struct scene {

	vec3 eye;
	vec3 center;
	vec3 up;
	vec3 background;

	unsigned int w;
	unsigned int h;
	float fovy;

	float inc;

	vec3 attenuation;

	unsigned int maxdepth;
	unsigned int maxinternaldepth;

	unsigned int max_photon_depth;
	unsigned int num_photons;
	unsigned int num_of_diffuse;

	float loc_radius;

	unsigned int SSAA;
	unsigned int MSAA;
	unsigned int maxthreads;
	string output_filename;

	vector<light> lights;

	vector<rec_light> rec_lights;

	vector<triangle> triangles;
	vector<triangle_property> triangle_properties;

	vector<parallelogram> parallelograms;
	vector<parallelogram_property> parallelogram_properties;

	vector<sphere> spheres;
	vector<sphere_property> sphere_properties;

	vector<arbitrary_sphere> arbitrary_spheres;
	vector<sphere_property> arbitrary_sphere_properties;

	vector <bound_box> boxes;

	unsigned int max_kd_depth;
	unsigned int min_kd_leaf_size;
	kd_tree_node* kd_tree;

	unsigned char *pixels;
} scene;



typedef struct sselight {
	__m128 pos ;
	__m128 color ;
} sselight;


//final
typedef struct sserec_light {
	__m128 color ;
	__m128 corner ;
	__m128 width ;
	__m128 height ; 
	aligned_array<__m128> pts;
} sserec_light;

typedef struct ssephoton {
	float color_noitisop[6];
} ssephoton ;


typedef struct ssetriangle {
	__m128 pt0;
	__m128 T1;
	__m128 T2;
} ssetriangle;

typedef struct ssetriangle_property {
	__m128 ambient;
	__m128 emission; // index is hidden in the alpha term for emission
	__m128 diffuse; // alpha is hidden within the alpha term of diffuse
	__m128 specular; // shininess hidden in the alpha term of specular
	__m128 norm0;
	__m128 norm1;
	__m128 norm2;

	__m128 perp0;
	__m128 perp1;
	__m128 perp2;

	__m128 norm;

	float T1_lenOloc;
	float T2_lenOloc;

	float ang_cor;

	ssephoton*** photons;

	unsigned int u_max;
	unsigned int* v_max;

	//omp_lock_t** locks;

} ssetriangle_property ;

typedef struct sseparallelogram {
	__m128 pt0;
	__m128 T1;
	__m128 T2;
} sseparallelogram;

typedef struct sseparallelogram_property {
	__m128 ambient;
	__m128 emission; // index is hidden in the alpha term for emission
	__m128 diffuse; // alpha is hidden within the alpha term of diffuse
	__m128 specular; // shininess hidden in the alpha term of specular
	__m128 norm0;
	__m128 norm1;
	__m128 norm2;
	__m128 norm3;

	__m128 perp1;
	__m128 perp2;

	__m128 norm;

	float T1_lenOloc;
	float T2_lenOloc;

	float ang_cor;

	ssephoton*** photons;

	unsigned int u_max;
	unsigned int v_max;

	//omp_lock_t** locks; // changed to use atomics

} sseparallelogram_property ;


typedef struct ssesphere {
	__m128 pos; // r^2 hidden alpha component of vector
} ssesphere;

typedef struct ssearbitrary_sphere {
	__m128 pos; // r^2 hidden alpha component of vector
	fmat4 inversetransform ;
} ssearbitrary_sphere;

typedef struct ssesphere_property {
	__m128 ambient;
	__m128 emission; // index is hidden in the alpha term for emission
	__m128 diffuse; // alpha is hidden within the alpha term of diffuse
	__m128 specular; // shininess hidden in the alpha term of specular

	omp_lock_t lock;

	aligned_array<ssephoton> photons;
	//vector<ssephoton> photons; // breaks locking scheme

} ssesphere_property;


typedef struct ssehit {
	__m128 start ; 
	__m128 dir ; 
	unsigned int depth ; 
	unsigned int internaldepth ;
	unsigned int type ; 
	unsigned int obj ; 
	float u ; 
	float v ; 
	float t ; 
	float alpha ; 
} ssehit;


typedef struct sse_scene {

	__m128 eye;
	__m128 center;
	__m128 up;
	__m128 background;

	unsigned int w;
	unsigned int h;
	float fovy;

	float inc;

	float attenuation[3];

	unsigned int maxdepth;
	unsigned int maxinternaldepth;

	unsigned int max_photon_depth;
	unsigned int num_photons;
	unsigned int num_of_diffuse;

	float loc_radius;

	unsigned int SSAA;
	unsigned int MSAA;
	unsigned int maxthreads;
	string output_filename;

	aligned_array<sselight> lights;

	aligned_array<sserec_light> rec_lights;

	aligned_array<ssetriangle> triangles;
	aligned_array<ssetriangle_property> triangle_properties;

	aligned_array<sseparallelogram> parallelograms;
	aligned_array<sseparallelogram_property> parallelogram_properties;

	aligned_array<ssesphere> spheres;
	aligned_array<ssesphere_property> sphere_properties;

	aligned_array<ssearbitrary_sphere> arbitrary_spheres;
	aligned_array<ssesphere_property> arbitrary_sphere_properties;

	vector <bound_box> boxes;

	unsigned int max_kd_depth;
	unsigned int min_kd_leaf_size;
	kd_tree_node* kd_tree;

	unsigned char *pixels;

} sse_scene;


#ifdef MAIN

scene make_scene(void) {

	scene scn = {};

	scn.eye = vec3(0.0f, 0.0f, 0.0f);
	scn.center = vec3(1.0f, 0.0f, 0.0f);
	scn.up = vec3(0.0f, 0.0f, 1.0f);
	scn.background = vec3(0.0f, 0.0f, 0.0f);

	scn.w = 128;
	scn.h = 128;
	scn.fovy = 55.0f;

	scn.inc = 0.1f;

	scn.attenuation[0] = 1.0f;
	scn.attenuation[1] = 0.0f;
	scn.attenuation[2] = 0.0f;

	scn.maxdepth = 5;
	scn.maxinternaldepth = 3;

	scn.max_photon_depth = 3;
	scn.num_photons = 1000;
	scn.num_of_diffuse = 8;
	scn.loc_radius = 0.1f;

	scn.SSAA = 1;
	scn.MSAA = 0;
	scn.maxthreads = omp_get_max_threads();
	scn.output_filename = "raytrace_picture";

	/*
	scn.lights.clear();

	scn.rec_lights.clear();

	scn.triangles.clear();
	scn.triangle_properties.clear();

	scn.parallelograms.clear();
	scn.parallelogram_properties.clear();

	scn.spheres.clear();
	scn.sphere_properties.clear();

	scn.arbitrary_spheres.clear();
	scn.arbitrary_sphere_properties.clear();

	scn.boxes.clear();
	*/

	scn.max_kd_depth = 9;
	scn.min_kd_leaf_size = 32;
	scn.kd_tree = NULL;

	scn.pixels = NULL;

	return scn;
}

void init_pixels(scene* scn) {

	scn->pixels = (unsigned char*) malloc(3 * scn->w * scn->h * sizeof(unsigned char));
	if (scn->pixels == NULL) {
		printf("Memory allocation failure\n");
		exit(1);
	}
}

void destroy_scene(scene* scn) {

	scn->eye = vec3(0.0f, 0.0f, 0.0f);
	scn->center = vec3(1.0f, 0.0f, 0.0f);
	scn->up = vec3(0.0f, 0.0f, 1.0f);
	scn->background = vec3(0.0f, 0.0f, 0.0f);

	scn->w = 128;
	scn->h = 128;
	scn->fovy = 55.0f;

	scn->inc = 0.1f;

	scn->attenuation[0] = 1.0f;
	scn->attenuation[1] = 0.0f;
	scn->attenuation[2] = 0.0f;

	scn->maxdepth = 5;
	scn->maxinternaldepth = 3;

	scn->max_photon_depth = 3;
	scn->num_photons = 1000;
	scn->num_of_diffuse = 8;
	scn->loc_radius = 0.1f;

	scn->SSAA = 1;
	scn->MSAA = 0;
	scn->maxthreads = omp_get_max_threads();
	scn->output_filename = "raytrace_picture";

	scn->lights.clear();

	scn->rec_lights.clear();

	scn->triangles.clear();
	scn->triangle_properties.clear();

	scn->parallelograms.clear();
	scn->parallelogram_properties.clear();

	scn->spheres.clear();
	scn->sphere_properties.clear();

	scn->arbitrary_spheres.clear();
	scn->arbitrary_sphere_properties.clear();

	scn->boxes.clear();

	scn->max_kd_depth = 9;
	scn->min_kd_leaf_size = 32;
	if (scn->kd_tree) free(scn->kd_tree);
	scn->kd_tree = NULL;

	if (scn->pixels) free(scn->pixels);
	scn->pixels = NULL;
}


sse_scene make_sse_scene(void) {

	sse_scene scn = {};

	scn.eye = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
	scn.center = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
	scn.up = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
	scn.background = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

	scn.w = 128;
	scn.h = 128;
	scn.fovy = 55.0f;

	scn.inc = 0.1f;

	scn.attenuation[0] = 1.0f;
	scn.attenuation[1] = 0.0f;
	scn.attenuation[2] = 0.0f;

	scn.maxdepth = 5;
	scn.maxinternaldepth = 3;

	scn.max_photon_depth = 3;
	scn.num_photons = 1000;
	scn.num_of_diffuse = 8;
	scn.loc_radius = 0.1f;

	scn.SSAA = 1;
	scn.MSAA = 0;
	scn.maxthreads = omp_get_max_threads();
	scn.output_filename = "raytrace_picture";

	scn.lights = aligned_array<sselight> (16);

	scn.rec_lights = aligned_array<sserec_light> (16);

	scn.triangles = aligned_array<ssetriangle> (16);
	scn.triangle_properties = aligned_array<ssetriangle_property> (16);

	scn.parallelograms = aligned_array<sseparallelogram> (16);
	scn.parallelogram_properties = aligned_array<sseparallelogram_property> (16);

	scn.spheres = aligned_array<ssesphere> (16);
	scn.sphere_properties = aligned_array<ssesphere_property> (16);

	scn.arbitrary_spheres = aligned_array<ssearbitrary_sphere> (16);
	scn.arbitrary_sphere_properties = aligned_array<ssesphere_property> (16);

	//scn.boxes.clear();

	scn.max_kd_depth = 9;
	scn.min_kd_leaf_size = 32;
	scn.kd_tree = NULL;

	scn.pixels = NULL;

	return scn;
}

void init_pixels(sse_scene* scn) {

	scn->pixels = (unsigned char*) malloc(3 * scn->w * scn->h * sizeof(unsigned char));
	if (scn->pixels == NULL) {
		printf("Memory allocation failure\n");
		exit(1);
	}
}

void destroy_ssetriangle_photons(ssetriangle_property* triprop) {

	unsigned int i, k;

	// destroy the first padding column and the first normal column
	for (k = 0; k < triprop->v_max[0]; ++k){
		free(triprop->photons[0][k]);
		//omp_destroy_lock(&triprop->locks[0][k]);
	}

	triprop->v_max[0] = 0;
	free(triprop->photons[0]);
	//free(triprop->locks[0]);
	

	for (k = 0; k < triprop->v_max[1]; ++k){
		free(triprop->photons[1][k]);
		//omp_destroy_lock(&triprop->locks[1][k]);
	}

	triprop->v_max[1] = 0;
	free(triprop->photons[1]);
	//free(triprop->locks[1]);
	

	// destroy the remaining rows
	for (i = 2; i < triprop->u_max; ++i) {

		for (k = 0; k < triprop->v_max[i]; ++k) {
			free(triprop->photons[i][k]);
			//omp_destroy_lock(&triprop->locks[i][k]);
		}

		triprop->v_max[i] = 0;
		free(triprop->photons[i]);
		//free(triprop->locks[i]);
	}

	free(triprop->photons);
	//free(triprop->locks);
}

void destroy_ssetriangles(aligned_array<ssetriangle>* triangles, aligned_array<ssetriangle_property>* triangle_properties, float loc_radius) {

	unsigned int i;

	for (i = 0; i < triangles->size(); ++i)
		destroy_ssetriangle_photons(&(*triangle_properties)[i]);

	triangles->clear();
	triangle_properties->clear();
}


void destroy_sseparallelogram_photons(sseparallelogram_property* parprop) {

	unsigned int i, k;

	// destroy the rows
	for (i = 0; i < parprop->u_max; ++i) {

		for (k = 0; k < parprop->v_max; ++k) {
			free(parprop->photons[i][k]);
			//omp_destroy_lock(&parprop->locks[i][k]);
		}

		free(parprop->photons[i]);
		//free(parprop->locks[i]);
	}

	parprop->u_max = 0;
	parprop->v_max = 0;
	free(parprop->photons);
	//free(parprop->locks);
}

void destroy_sseparallelograms(aligned_array<sseparallelogram>* parallelograms, aligned_array<sseparallelogram_property>* parallelogram_properties, float loc_radius) {

	unsigned int i;

	for (i = 0; i < parallelograms->size(); ++i)
		destroy_sseparallelogram_photons(&(*parallelogram_properties)[i]);

	parallelograms->clear();
	parallelogram_properties->clear();
}

void destroy_ssespheres(aligned_array<ssesphere>* spheres, aligned_array<ssesphere_property>* sphere_properties) {

	unsigned int i;

	for (i = 0; i < sphere_properties->size(); ++i) {
		(*sphere_properties)[i].photons.clear();
		omp_destroy_lock(&(*sphere_properties)[i].lock);
	}

	spheres->clear();
	sphere_properties->clear();
}

void destroy_ssearbitrary_spheres(aligned_array<ssearbitrary_sphere>* spheres, aligned_array<ssesphere_property>* sphere_properties) {

	unsigned int i;

	for (i = 0; i < sphere_properties->size(); ++i) {
		(*sphere_properties)[i].photons.clear();
		omp_destroy_lock(&(*sphere_properties)[i].lock);
	}

	spheres->clear();
	sphere_properties->clear();
}


void destroy_sse_scene(sse_scene* scn) {

	scn->eye = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
	scn->center = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
	scn->up = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
	scn->background = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

	scn->w = 128;
	scn->h = 128;
	scn->fovy = 55.0f;

	scn->inc = 0.1f;

	scn->attenuation[0] = 1.0f;
	scn->attenuation[1] = 0.0f;
	scn->attenuation[2] = 0.0f;

	scn->maxdepth = 5;
	scn->maxinternaldepth = 3;

	scn->max_photon_depth = 3;
	scn->num_photons = 1000;
	scn->num_of_diffuse = 8;
	scn->loc_radius = 0.1f;

	scn->SSAA = 1;
	scn->MSAA = 0;
	scn->maxthreads = omp_get_max_threads();
	scn->output_filename = "raytrace_picture";

	scn->lights.clear();

	scn->rec_lights.clear();

	destroy_ssetriangles(&scn->triangles, &scn->triangle_properties, scn->loc_radius);

	destroy_sseparallelograms(&scn->parallelograms, &scn->parallelogram_properties, scn->loc_radius);

	destroy_ssespheres(&scn->spheres, &scn->sphere_properties);

	destroy_ssearbitrary_spheres(&scn->arbitrary_spheres, &scn->arbitrary_sphere_properties);

	scn->boxes.clear();

	scn->max_kd_depth = 9;
	scn->min_kd_leaf_size = 32;
	if (scn->kd_tree) free(scn->kd_tree);
	scn->kd_tree = NULL;

	if (scn->pixels) free(scn->pixels);
	scn->pixels = NULL;
}

#else

scene make_scene(void) ;

void init_pixels(scene* scn) ;

void destroy_scene(scene* scn) ;


sse_scene make_sse_scene(void) ;

void init_pixels(sse_scene* scn) ;

void destroy_sse_scene(sse_scene* scn) ;


#endif

#endif
