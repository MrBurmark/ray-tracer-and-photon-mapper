
//#include "kd_tree.h"
#include "intersect.h"
#include "photon.h"
#include <stdlib.h>
#include <math.h>
#include <random>
#include <omp.h>

inline vec3 random_shoot(void){

	float x1, x2;

	do{
		x1 = 1.0f - 2.0f * rand()/(float)RAND_MAX;
		x2 = 1.0f - 2.0f * rand()/(float)RAND_MAX;
	} while(x1*x1 + x2*x2 >= 1.0f);

	return vec3(2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2), 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2), 1.0f - 2.0f*(x1*x1 + x2*x2));
}

void shoot_photons(scene* scn)
{
	printf("Shooting Photons\n");


	for(unsigned int k = 0; k< scn->rec_lights.size(); k++)
	{
		rec_light* rec = &scn->rec_lights[k];

		vec3 norm = glm::cross(rec->width, rec->height);

		vec3 color = rec->color * 30.0f/(scn->loc_radius*scn->loc_radius*scn->loc_radius*(float)scn->num_photons);

		vec3 start;
		vec3 direction;
		hit hitt;

		double prev_t = omp_get_wtime();
		srand((unsigned int)(prev_t*1.18281828459045235360287471352e7));

		for(unsigned int i = 0; i < scn->num_photons; i++)
		{
			if (i%6484 == 0) {
				double new_t = omp_get_wtime();
				if (new_t != prev_t){
					srand((unsigned int)(new_t*1.18281828459045235360287471352e7));
					prev_t = new_t;
				}
			}
			
			start = rec->corner + rand()/(float)RAND_MAX * rec->width + rand()/(float)RAND_MAX * rec->height;

			direction = glm::normalize(random_shoot());

			if (glm::dot(norm, direction) < 0.0f)
				direction = -direction;

			hitt = intersect (scn, start, direction, 0, 1.0f, 0.0f);

			interact_photons(scn, hitt, color);

		}
	}

	for(unsigned int k = 0; k < scn->lights.size(); k++)
	{
		light* l = &scn->lights[k];

		unsigned int type = l->type;

		if (type == POINT) { // point light

			vec3 color = l->color * 30.0f/(scn->loc_radius*scn->loc_radius*scn->loc_radius*(float)scn->num_photons);

			vec3 start = l->pos;

			vec3 direction(0.0f);
			hit hitt;

			double prev_t = 0.0;

			for(unsigned int i = 0; i < scn->num_photons; i++) {

				if (i%6484 == 0) {
					double new_t = omp_get_wtime();
					if (new_t != prev_t){
						srand((unsigned int)(new_t*1.18281828459045235360287471352e7));
						prev_t = new_t;
					}
				}

				direction = glm::normalize(random_shoot());

				hitt = intersect (scn, start, direction, 0, 1.0f, 0.0f);

				interact_photons(scn, hitt, color);
			}
		}
	}
}

void mirror_side(vec3 V0, vec3 T1, vec3 T2, vec3 perp, triangle_property* triprop, photon pht, vec3 position, scene* scn) {

	vec3 V = glm::cross(T1, perp);

	float r = glm::dot(glm::cross(T2, position - V0), T1) / glm::dot(V, T2);

	if (r < scn->loc_radius) {

		pht.pos = position + 2.0f*r*perp;

		triprop->photons.push_back(pht);
	}
}

void mirror_point(vec3 pt, triangle_property* triprop, photon pht, vec3 position, scene* scn) {

	float r = glm::length(pt - position);

	if (r < scn->loc_radius) {

		pht.pos = position + 2.0f*(pt - position);

		triprop->photons.push_back(pht);
	}
}

void mirror_side_parallelogram(vec3 V0, vec3 T1, vec3 T2, vec3 perp, parallelogram_property* parprop, photon pht, vec3 position, scene* scn) {

	vec3 V = glm::cross(T1, perp);

	float r = glm::dot(glm::cross(T2, position - V0), T1) / glm::dot(V, T2);

	if (r < scn->loc_radius) {

		pht.pos = position + 2.0f*r*perp;

		parprop->photons.push_back(pht);
	}
}

void mirror_point_parallelogram(vec3 pt, parallelogram_property* parprop, photon pht, vec3 position, scene* scn) {

	float r = glm::length(pt - position);

	if (r < scn->loc_radius) {

		pht.pos = position + 2.0f*(pt - position);

		parprop->photons.push_back(pht);
	}
}

void interact_photons(scene* scn, hit hitt, vec3 color)
{
	if (hitt.obj == UINT_MAX) return;

	vec3 position = hitt.start + hitt.dir*hitt.t;

	vec3 photondirection = -hitt.dir;

	vec3 diffuse(0.0f);
	vec3 specular(0.0f);
	float index = 1.0f;
	float obj_alpha = 1.0f;
	float Dd;

	vec3 normal(0.0f);
	if (hitt.type == TRIANGLE) {

		triangle_property* triprop = &scn->triangle_properties[hitt.obj];

		normal = triprop->norm0*(1.0f - hitt.u - hitt.v) + triprop->norm1*hitt.u + triprop->norm2*hitt.v;

		diffuse = triprop->diffuse;
		specular = triprop->specular;

		obj_alpha = triprop->alpha;

		index = triprop->index;

		Dd = glm::dot(photondirection, normal);

		if (Dd > 0.0f) { // avoid points not facing the light

			photon pht = {color * diffuse*Dd * obj_alpha, position};

			vec3 triT1 = scn->triangles[hitt.obj].T1;
			vec3 triT2 = scn->triangles[hitt.obj].T2;
			vec3 tript0 = scn->triangles[hitt.obj].pt0;
			vec3 trinorm = scn->triangle_properties[hitt.obj].norm;


			mirror_side(tript0 + triT1, trinorm, triT2 - triT1, scn->triangle_properties[hitt.obj].perp0, triprop, pht, position, scn);

			mirror_side(tript0, trinorm, triT2, scn->triangle_properties[hitt.obj].perp1, triprop, pht, position, scn);

			mirror_side(tript0, triT1, trinorm, scn->triangle_properties[hitt.obj].perp2, triprop, pht, position, scn);

			mirror_point(tript0, triprop, pht, position, scn);

			mirror_point(tript0 + triT1, triprop, pht, position, scn);

			mirror_point(tript0 + triT2, triprop, pht, position, scn);


			pht.pos = position;

			triprop->photons.push_back(pht);
		}
	}
	else if (hitt.type == PARALLELOGRAM) {

		parallelogram_property* parprop = &scn->parallelogram_properties[hitt.obj];

		normal = parprop->norm0*(1.0f - hitt.u)*(1.0f - hitt.v) + parprop->norm1*hitt.u*(1.0f - hitt.v) + parprop->norm2*(1.0f - hitt.u)*hitt.v + parprop->norm3*hitt.u*hitt.v;

		diffuse = parprop->diffuse;
		specular = parprop->specular;

		obj_alpha = parprop->alpha;

		index = parprop->index;

		Dd = glm::dot(photondirection, normal);

		if (Dd > 0.0f) { // avoid points not facing the light

			photon pht = {color * diffuse*Dd * obj_alpha, position};

			vec3 parT1 = scn->parallelograms[hitt.obj].T1;
			vec3 parT2 = scn->parallelograms[hitt.obj].T2;
			vec3 parpt0 = scn->parallelograms[hitt.obj].pt0;
			vec3 parnorm = scn->parallelogram_properties[hitt.obj].norm;


			mirror_side_parallelogram(parpt0, parnorm, parT2, scn->parallelogram_properties[hitt.obj].perp2, parprop, pht, position, scn);

			mirror_side_parallelogram(parpt0, parT1, parnorm, scn->parallelogram_properties[hitt.obj].perp1, parprop, pht, position, scn);

			mirror_side_parallelogram(parpt0 + parT1, parT2, parnorm, -scn->parallelogram_properties[hitt.obj].perp2, parprop, pht, position, scn);

			mirror_side_parallelogram(parpt0 + parT2, parnorm, parT1, -scn->parallelogram_properties[hitt.obj].perp1, parprop, pht, position, scn);
			

			mirror_point_parallelogram(parpt0, parprop, pht, position, scn);

			mirror_point_parallelogram(parpt0 + parT1, parprop, pht, position, scn);

			mirror_point_parallelogram(parpt0 + parT2, parprop, pht, position, scn);

			mirror_point_parallelogram(parpt0 + parT1 + parT2, parprop, pht, position, scn);


			pht.pos = position;

			parprop->photons.push_back(pht);
		}
	}
	else if (hitt.type == SPHERE) {

		sphere sph = scn->spheres[hitt.obj];

		// this might be able to be done faster
		normal = glm::normalize( position - sph.pos );

		sphere_property* sphprop = &scn->sphere_properties[hitt.obj];

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		obj_alpha = sphprop->alpha;

		index = sphprop->index;

		Dd = glm::dot(photondirection, normal);

		if (Dd > 0.0f) { // avoid points not facing the light

			photon pht = {color * diffuse*Dd * obj_alpha, position};

			sphprop->photons.push_back(pht);
		}
	}
	else if (hitt.type == ARBITRARY_SPHERE) {

		arbitrary_sphere sph = scn->arbitrary_spheres[hitt.obj];

		// this might be able to be done faster
		normal = glm::normalize( glm::transpose(mat3(sph.inversetransform)) * (vec3(sph.inversetransform * vec4(position, 1.0f)) - sph.pos) );

		sphere_property* sphprop = &scn->arbitrary_sphere_properties[hitt.obj];

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		obj_alpha = sphprop->alpha;

		index = sphprop->index;

		Dd = glm::dot(photondirection, normal);

		if (Dd > 0.0f) { // avoid points not facing the light

			photon pht = {color * diffuse*Dd * obj_alpha, position};

			sphprop->photons.push_back(pht);
		}
	}

	if (hitt.depth < scn->max_photon_depth) {

		// mirror reflections
		if (glm::length(specular) > 0.0f) {
			vec3 mirror = glm::normalize( 2.0f*glm::dot(photondirection, normal)*normal - photondirection );

			hit hitm = intersect(scn, position, mirror, hitt.depth + 1, 1.0f, -SMALL);

			interact_photons(scn, hitm, color * specular);
		}

		//diffuse
		if (glm::length(diffuse) > 0.0f) {
			for (unsigned int l = 0; l < scn->num_of_diffuse; l++)
			{
				vec3 diff = glm::normalize(random_shoot());

				if (glm::dot(diff, normal) < 0.0f)
					diff = -diff;

				hit hitm = intersect(scn, position, diff, hitt.depth + 1, 1.0f, -SMALL);

				interact_photons(scn, hitm, color * diffuse*Dd * obj_alpha / (float)scn->num_of_diffuse);
			}
		}

	}
}



vec3 photon_shade(scene* scn, hit a) {

	unsigned int i, k;

	if (a.obj == UINT_MAX) return scn->background;

	vec3  color(0.0f); 

	vec3 myposition = a.start + a.dir*a.t;

	vec3 eyedirection = -a.dir;

	float ray_alpha = a.alpha;

	vector<photon> photons;

	vec3 specular(0.0f);
	float shininess = 0.0f;
	float index = 1.0f;
	float obj_alpha = 1.0f;

	vec3 normal(0.0f);
	if (a.type == TRIANGLE) {

		triangle_property triprop = scn->triangle_properties[a.obj];

		normal = triprop.norm0*(1.0f - a.u - a.v) + triprop.norm1*a.u + triprop.norm2*a.v;

		photons = triprop.photons;

		specular = triprop.specular;
		shininess = triprop.shininess;

		obj_alpha = triprop.alpha;

		index = triprop.index;
	}
	else if (a.type == PARALLELOGRAM) {

		parallelogram_property parprop = scn->parallelogram_properties[a.obj];

		normal = parprop.norm0*(1.0f - a.u)*(1.0f - a.v) + parprop.norm1*a.u*(1.0f - a.v) + parprop.norm2*(1.0f - a.u)*a.v + parprop.norm3*a.u*a.v;

		photons = parprop.photons;

		specular = parprop.specular;
		shininess = parprop.shininess;

		obj_alpha = parprop.alpha;

		index = parprop.index;
	}
	else if (a.type == SPHERE) {

		sphere sph = scn->spheres[a.obj];

		// this might be able to be done faster
		normal = glm::normalize( myposition - sph.pos );

		sphere_property sphprop = scn->sphere_properties[a.obj];

		photons = sphprop.photons;

		specular = sphprop.specular;
		shininess = sphprop.shininess;

		obj_alpha = sphprop.alpha;

		index = sphprop.index;
	}
	else if (a.type == ARBITRARY_SPHERE) {

		arbitrary_sphere sph = scn->arbitrary_spheres[a.obj];

		normal = glm::normalize( glm::transpose(mat3(sph.inversetransform)) * (vec3(sph.inversetransform * vec4(myposition, 1.0f)) - sph.pos) );

		sphere_property sphprop = scn->arbitrary_sphere_properties[a.obj];

		photons = sphprop.photons;

		specular = sphprop.specular;
		shininess = sphprop.shininess;

		obj_alpha = sphprop.alpha;

		index = sphprop.index;
	} else return color;


	vec3 diffuse(0.0f);

	for (i = 0; i < photons.size(); ++i) {

		float len = glm::length(myposition - (photons[i].pos));

		if (len < scn->loc_radius)
			diffuse += photons[i].color * (scn->loc_radius - len);
	}

	color += ray_alpha * diffuse;


	for (i = 0; i < scn->lights.size(); ++i) {

		light l = scn->lights[i];

		float latten = 1.0f;

		float ldist = INF;

		vec3 ldirection;

		if (l.type == POINT){
			ldirection = l.pos - myposition;

			ldist = glm::length(ldirection);

			latten = 1.0f/(scn->attenuation.x + scn->attenuation.y*ldist + scn->attenuation.z*ldist*ldist);
		} else if (l.type == DIRECTIONAL){
			ldirection = l.pos;
		} else continue;

		ldirection = glm::normalize(ldirection);

		float Dd = glm::dot(ldirection, normal) > 0.0f ? glm::dot(ldirection, normal) : 0.0f;

		if (Dd > 0.0f) { // avoid points not facing the light

			hit shadow = intersect(scn, myposition, ldirection, 0, 0.0f, -SMALL); // rays do not intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

			if (ldist <= shadow.t) {

				vec3 halfvec = glm::normalize(ldirection + eyedirection);

				float Sd = powf( glm::dot(halfvec, normal) > 0.0f ? glm::dot(halfvec, normal) : 0.0f, shininess );

				color += latten * l.color * (specular*Sd);
			}
		}
	}

	for (i = 0; i < scn->rec_lights.size(); ++i) {

		rec_light* l = &scn->rec_lights[i];

		float latten = 1.0f;

		float ldist = INF;

		vec3 ldirection;

		vec3 colorrl(0.0f);

		for (k = 0; k < l->pts.size(); ++k) {

			ldirection = l->pts[k] - myposition;

			ldist = glm::length(ldirection);

			latten = 1.0f/(scn->attenuation.x + scn->attenuation.y*ldist + scn->attenuation.z*ldist*ldist);

			ldirection = glm::normalize(ldirection);

			float Dd = glm::dot(ldirection, normal) > 0.0f ? glm::dot(ldirection, normal) : 0.0f;

			if (Dd > 0.0f) { // avoid points not facing the light

				hit shadow = intersect(scn, myposition, ldirection, 0, 0.0f, -SMALL); // rays do not intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

				if (ldist <= shadow.t) {

					vec3 halfvec = glm::normalize(ldirection + eyedirection);

					float Sd = powf( glm::dot(halfvec, normal) > 0.0f ? glm::dot(halfvec, normal) : 0.0f, shininess );

					colorrl += latten * l->color * (specular*Sd);
				}
			}
		}

		if (l->pts.size() > 0)
			color += colorrl * pi/(float)l->pts.size();
	}

	// mirror reflections
	if (a.depth < scn->maxdepth && glm::length(specular) > 0.0f) {

		vec3 mirror = glm::normalize( 2.0f*glm::dot(eyedirection, normal)*normal - eyedirection );

		hit m = intersect(scn, myposition, mirror, a.depth + 1, 1.0f, -SMALL);

		vec3 colorm = photon_shade(scn, m);

		color += colorm*specular;
	}
	

	
	/*
	
	// refractions
	if (a.depth < scn->maxdepth && obj_alpha*ray_alpha < ray_alpha && glm::length(diffuse) > 0.0f) {

		vec3 colorn(0.0f);

		// entering material, if if fails there is no transmittance
		if (1 - (1 - glm::dot(eyedirection, normal)*glm::dot(eyedirection, normal))*air_refracive_index*air_refracive_index/(index*index) > 0.0f) {
			
			vec3 in_refract = (glm::dot(eyedirection, normal)*normal - eyedirection) * air_refracive_index/index;
			in_refract += -normal*sqrtf(1 - (1 - glm::dot(eyedirection, normal)*glm::dot(eyedirection, normal))*air_refracive_index*air_refracive_index/(index*index));
			in_refract = glm::normalize(in_refract);
			hit m = internal_intersect(scn, myposition, in_refract, a.depth + 1, 0.0f, SMALL);			

			// internal ray
			if (m.obj >= 0) {

				vec3 eyedirm = -m.dir;

				vec3 mypositionm = m.start + m.dir*m.t;

				vec3 normalm(0.0f);

				if (m.type == TRIANGLE) {

					triangle_property triprop = scn->triangle_properties[m.obj];

					normalm = triprop.norm0*(1.0f - m.u - m.v) + triprop.norm1*m.u + triprop.norm2*m.v;
				}
				else if (m.type = SPHERE) {

					sphere sph = scn->spheres[m.obj];

					// this might be able to be done faster
					normalm = glm::normalize( mypositionm- sph.pos );
				}
				else if (m.type = ARBITRARY_SPHERE) {

					arbitrary_sphere sph = scn->arbitrary_spheres[m.obj];

					// this might be able to be done faster
					normalm = glm::normalize( glm::transpose(mat3(sph.inversetransform)) * (vec3(sph.inversetransform * vec4(mypositionm, 1.0f)) - sph.pos) );
				}
				// as hiting the back of the object
				normalm = -normalm;

				// if if fails there is no transmittance, only reflectance
				if (1 - (1 - glm::dot(eyedirm, normalm)*glm::dot(eyedirm, normalm)) * index*index/(air_refracive_index*air_refracive_index) > 0.0f) {
					
					vec3 out_refract = (glm::dot(eyedirm, normalm)*normalm - eyedirm) * index/air_refracive_index;
					out_refract += -normalm*sqrtf(1 - (1 - glm::dot(eyedirm, normalm)*glm::dot(eyedirm, normalm)) * index*index/(air_refracive_index*air_refracive_index));
					out_refract = glm::normalize(in_refract);

					hit n = intersect(scn, mypositionm, out_refract, a.depth + 1, ray_alpha - obj_alpha*ray_alpha, -SMALL);

					colorn = shade(scn, n);
				}
				// internal reflections
				// not implemented in glm version



			}

		} else colorn =  scn->background;

		color += colorn*diffuse;
	}
	*/
	
	

	return color;
}


inline __m128 sse_random_shoot(mt19937& generator, uniform_real_distribution<float>& distribution){

	float x1, x2;

	do{
		x1 = 1.0f - 2.0f * distribution(generator);
		x2 = 1.0f - 2.0f * distribution(generator);
	} while(x1*x1 + x2*x2 >= 1.0f);
    
	return _mm_set_ps(2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2), 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2), 1.0f - 2.0f*(x1*x1 + x2*x2), 0.0f);
}


void sse_shoot_photons(sse_scene* scn)
{
	printf("Shooting Photons\n");

	unsigned int m = 0;
	float work = (float)((scn->rec_lights.size())*scn->num_photons);

	for(unsigned int k = 0; k < scn->lights.size(); k++)
	{
		float type;
		_mm_store_ss(&type, scn->lights[k].pos);

		if (type != 0.0f)  // point light
			work += (float)scn->num_photons;
	}

	float n = scn->inc;

	unsigned int II = scn->num_photons/1000 ? scn->num_photons/1000 : 1;

	for(unsigned int k = 0; k < scn->rec_lights.size(); k++)
	{
		sserec_light rec = scn->rec_lights[k];

		__m128 norm = __m128_CROSS(rec.width, rec.height);

		__m128 color = __m128_MUL_float_set(rec.color, 30.0f/(scn->loc_radius*scn->loc_radius*scn->loc_radius*(float)scn->num_photons));


		#pragma omp parallel num_threads(scn->maxthreads)
		{

			unsigned long seed1 = (unsigned long)( omp_get_wtime() * 1.8491651947497106e6 * (double)(omp_get_thread_num() + 1) );

			//printf("seed %d\n", seed1);

			//default_random_engine generator = default_random_engine( (unsigned long)((omp_get_wtime() - floor(omp_get_wtime()))*(double)(omp_get_thread_num() + 1)) );
			//random_device rd;
			mt19937 generator( seed1 );
			uniform_real_distribution<float> distribution = uniform_real_distribution<float>(0.0f,1.0f);

			#pragma omp for schedule(dynamic, 1)
			for (int ii = 0; ii < scn->num_photons; ii += II) {

				for(unsigned int i = ii; i < ii + II && i < scn->num_photons; i++)
				{

					__m128 start;
					__m128 direction;
					ssehit hitt;
					
					start = _mm_add_ps(rec.corner, _mm_add_ps(__m128_MUL_float_set(rec.width, distribution(generator)), __m128_MUL_float_set(rec.height, distribution(generator))));

					direction = sse_random_shoot(generator, distribution);
					__m128_NORM3(direction);
			
					float x1;
					__m128_DOT3(norm, direction, x1);
					if (x1 < 0.0f)
						direction = __m128_NEG(direction);

					hitt = sse_intersect (scn, start, direction, 0, 0, 1.0f, 0.0f);

					sse_interact_photons(scn, hitt, color, generator, distribution);

				}

				#pragma omp critical
				{
					#pragma omp flush (m, n)
					m += II; // approximate
					if ((float)m > n*(float)work) {
						printf("photon execution %.0f%% complete\n", 100.0f*n);
						n += scn->inc;
					}
					#pragma omp flush (m, n)
				}
			}
		}
	}

	

	for(unsigned int k = 0; k < scn->lights.size(); k++)
	{
		sselight* l = &scn->lights[k];

		float type;
		_mm_store_ss(&type, l->pos);

		if (type != 0.0f) { // point light

			__m128 color = __m128_MUL_float_set(l->color, 30.0f/(scn->loc_radius*scn->loc_radius*scn->loc_radius*(float)scn->num_photons));

			__m128 start = l->pos;

			__m128 direction;
			ssehit hitt;

			#pragma omp parallel num_threads(scn->maxthreads)
			{

				unsigned long seed1 = (unsigned long)( omp_get_wtime() * 1.8491651947497106e6 * (double)(omp_get_thread_num() + 1) );

				//printf("seed %d\n", seed1);

				//default_random_engine generator = default_random_engine( seed1 );
				//random_device rd;
				mt19937 generator( seed1 );
				uniform_real_distribution<float> distribution = uniform_real_distribution<float>(0.0f,1.0f);

				#pragma omp for schedule(dynamic, 1)
				for (int ii = 0; ii < scn->num_photons; ii += II) {

					for(unsigned int i = ii; i < ii + II && i < scn->num_photons; i++) {

						direction = sse_random_shoot(generator, distribution);
						__m128_NORM3(direction);

						hitt = sse_intersect (scn, start, direction, 0, 0, 1.0f, 0.0f);

						sse_interact_photons(scn, hitt, color, generator, distribution);

					}

					#pragma omp critical
					{
						#pragma omp flush (m, n)
						m += II; // approximate
						if ((float)m > n*work) {
							printf("photon execution %.0f%% complete\n", 100.0f*n);
							n += scn->inc;
						}
						#pragma omp flush (m, n)
					}
				}
			}
		}
	}

	printf("Photons Shot\n");
}

inline void add_photon_triangle(const __m128& color, ssetriangle_property* triprop, float u, float v) {

	unsigned int u1 = (unsigned int)floorf(u*triprop->T1_lenOloc + 1.0f);

	if (u1 >= triprop->u_max) {
		//printf("bad u1");
		return;
	}

	unsigned int v1 = (unsigned int)floorf(v*triprop->T2_lenOloc + 1.0f);

	if (v1 >= triprop->v_max[u1]) {
		//printf("bad v1");
		return;
	}

	unsigned int m = (unsigned int)floorf( (u*triprop->T1_lenOloc - floorf(u*triprop->T1_lenOloc)) * TILEf );

	if (m >= TILE) {
		//printf("bad m");
		return;
	}

	unsigned int n = (unsigned int)floorf( (v*triprop->T2_lenOloc - floorf(v*triprop->T2_lenOloc)) * TILEf );
	
	if (n >= TILE) {
		//printf("bad n");
		return;
	}

	ssephoton* pht = &triprop->photons[u1][v1][m*TILE + n];

	__declspec(align(16)) float tmp[4];

	_mm_storer_ps(tmp, color);

	#pragma omp atomic
		pht->color_noitisop[0] += tmp[0];

	#pragma omp atomic
		pht->color_noitisop[1] += tmp[1];

	#pragma omp atomic
		pht->color_noitisop[2] += tmp[2];

	/*
	omp_set_lock(&triprop->locks[u1][v1]);
	{
		#pragma omp flush
		pht->color = _mm_add_ps(pht->color, color);

		#pragma omp flush
	}
	omp_unset_lock(&triprop->locks[u1][v1]);
	*/
}

inline void mirror_side_triangle(const __m128& V0, const __m128& T1, const __m128& T2, const __m128& perp, ssetriangle_property* triprop, __m128& color, const __m128& position, const sse_scene* scn, const ssetriangle* tri) {

	__m128 V = __m128_CROSS(T1, perp);

	float r, s;
	__m128_DOT3(__m128_CROSS(T2, _mm_sub_ps(position, V0)), T1, r);
	__m128_DOT3(V, T2, s);
	r /= s;

	if (r < scn->loc_radius) {

		__m128 pos = _mm_add_ps(position, __m128_MUL_float_set(perp, 2.0f*r));

		float u, v;
		__m128_DOT3(_mm_sub_ps(pos, tri->pt0), tri->T1, u) ;
		__m128_LEN3(tri->T1, r) ;
		u /= r*r;

		__m128_DOT3(_mm_sub_ps(pos, tri->pt0), tri->T2, v) ;
		__m128_LEN3(tri->T2, s) ;
		v /= s*s;

		add_photon_triangle(color, triprop, u, v);
	}
}

inline void mirror_point_triangle(const __m128& pt, ssetriangle_property* triprop, __m128& color, const __m128& position, const sse_scene* scn, const ssetriangle* tri) {

	float r, s;
	__m128_LEN3(_mm_sub_ps(pt, position), r);

	if (r < scn->loc_radius) {

		__m128 pos = _mm_add_ps(position, __m128_MUL_float_set(_mm_sub_ps(pt, position), 2.0f));

		float u, v;
		__m128_DOT3(_mm_sub_ps(pos, tri->pt0), tri->T1, u) ;
		__m128_LEN3(tri->T1, r) ;
		u /= r*r;

		__m128_DOT3(_mm_sub_ps(pos, tri->pt0), tri->T2, v) ;
		__m128_LEN3(tri->T2, s) ;
		v /= s*s;

		add_photon_triangle(color, triprop, u, v);
	}
}

inline void add_photon_parallelogram(const __m128& color, sseparallelogram_property* parprop, float u, float v) {

	unsigned int u1 = (unsigned int)floorf(u*parprop->T1_lenOloc + 1.0f);

	if (u1 >= parprop->u_max) {
		//printf("bad u1");
		return;
	}

	unsigned int v1 = (unsigned int)floorf(v*parprop->T2_lenOloc + 1.0f);

	if (v1 >= parprop->v_max) {
		//printf("bad v1");
		return;
	}

	unsigned int m = (unsigned int)floorf( (u*parprop->T1_lenOloc - floorf(u*parprop->T1_lenOloc)) * TILEf );

	if (m >= TILE) {
		//printf("bad m");
		return;
	}

	unsigned int n = (unsigned int)floorf( (v*parprop->T2_lenOloc - floorf(v*parprop->T2_lenOloc)) * TILEf );
	
	if (n >= TILE) {
		//printf("bad n");
		return;
	}

	ssephoton* pht = &parprop->photons[u1][v1][m*TILE + n];

	__declspec(align(16)) float tmp[4];

	_mm_storer_ps(tmp, color);

	#pragma omp atomic
		pht->color_noitisop[0] += tmp[0];

	#pragma omp atomic
		pht->color_noitisop[1] += tmp[1];

	#pragma omp atomic
		pht->color_noitisop[2] += tmp[2];
	
	/*
	omp_set_lock(&parprop->locks[u1][v1]);
	{
		#pragma omp flush
		pht->color = _mm_add_ps(pht->color, color);

		#pragma omp flush
	}
	omp_unset_lock(&parprop->locks[u1][v1]);
	*/
}

inline void mirror_side_parallelogram(const __m128& V0, const __m128& T1, const __m128& T2, const __m128& perp, sseparallelogram_property* parprop, __m128& color, const __m128& position, const sse_scene* scn, const sseparallelogram* par) {

	__m128 V = __m128_CROSS(T1, perp);

	float r, s;
	__m128_DOT3(__m128_CROSS(T2, _mm_sub_ps(position, V0)), T1, r);
	__m128_DOT3(V, T2, s);
	r /= s;

	if (r < scn->loc_radius) {

		__m128 pos = _mm_add_ps(position, __m128_MUL_float_set(perp, 2.0f*r));

		float u, v;
		__m128_DOT3(_mm_sub_ps(pos, par->pt0), par->T1, u) ;
		__m128_LEN3(par->T1, r) ;
		u /= r*r;

		__m128_DOT3(_mm_sub_ps(pos, par->pt0), par->T2, v) ;
		__m128_LEN3(par->T2, s) ;
		v /= s*s;

		add_photon_parallelogram(color, parprop, u, v);
	}
}

inline void mirror_point_parallelogram(const __m128& pt, sseparallelogram_property* parprop, __m128& color, const __m128& position, const sse_scene* scn, const sseparallelogram* par) {

	float r, s;
	__m128_LEN3(_mm_sub_ps(pt, position), r);

	if (r < scn->loc_radius) {

		__m128 pos = _mm_add_ps(position, __m128_MUL_float_set(_mm_sub_ps(pt, position), 2.0f));

		float u, v;
		__m128_DOT3(_mm_sub_ps(pos, par->pt0), par->T1, u) ;
		__m128_LEN3(par->T1, r) ;
		u /= r*r;

		__m128_DOT3(_mm_sub_ps(pos, par->pt0), par->T2, v) ;
		__m128_LEN3(par->T2, s) ;
		v /= s*s;

		add_photon_parallelogram(color, parprop, u, v);
	}
}

void sse_internal_interact_photons(sse_scene* scn, ssehit& hitt, __m128& color, mt19937& generator, uniform_real_distribution<float>& distribution) {

	if (hitt.obj == UINT_MAX) return;

	__m128 position = _mm_add_ps(hitt.start, __m128_MUL_float_set(hitt.dir, hitt.t));

	__m128 photondirection = __m128_NEG(hitt.dir);

	__m128 normal = _mm_setzero_ps();

	float index = 1.0f; // index is hidden in the alpha term for emission
	float obj_alpha = 1.0f;


	if (hitt.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[hitt.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - hitt.u - hitt.v), __m128_MUL_float_set(triprop->norm1, hitt.u)), __m128_MUL_float_set(triprop->norm2, hitt.v));

		_mm_store_ss(&index,  triprop->emission);
		_mm_store_ss(&obj_alpha, triprop->diffuse);
	}
	else if (hitt.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[hitt.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - hitt.u)*(1.0f - hitt.v)), __m128_MUL_float_set(parprop->norm3, hitt.u*hitt.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - hitt.u)*hitt.v), __m128_MUL_float_set(parprop->norm1, hitt.u*(1.0f - hitt.v))));

		_mm_store_ss(&index,  parprop->emission);
		_mm_store_ss(&obj_alpha, parprop->diffuse);
	}
	else if (hitt.type == SPHERE) {

		ssesphere* sph = &scn->spheres[hitt.obj];

		normal = _mm_sub_ps( position, sph->pos);
		normal = __m128_NORM3(normal);

		ssesphere_property* sphprop = &scn->sphere_properties[hitt.obj];

		_mm_store_ss(&index,  sphprop->emission);
		_mm_store_ss(&obj_alpha, sphprop->diffuse);
	}
	else if (hitt.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[hitt.obj];

		// this might be able to be done faster
		normal = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, position), sph->pos); // sph->pos has r^2 in w
		normal = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normal );
		normal = __m128_NORM3(normal);

		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[hitt.obj];

		_mm_store_ss(&index,  sphprop->emission);
		_mm_store_ss(&obj_alpha, sphprop->diffuse);
	} else return;

	// as hiting the back of the object
	normal = __m128_NEG(normal);

	// if if fails there is no transmittance, this does not model internal reflection
	float tmp6;
	__m128_DOT3(photondirection, normal, tmp6);
	float tmp7 = 1 - (1 - tmp6*tmp6) * index*index/(air_refracive_index*air_refracive_index);

	if (tmp7 > 1e-6f && (1.0f - obj_alpha) > 1e-6f) {

		__m128 out_refract = __m128_MUL_float_set(_mm_sub_ps(_mm_mul_ps(__m128_DOT3___m128(photondirection, normal), normal), photondirection), index/air_refracive_index);

		out_refract = _mm_sub_ps(out_refract, _mm_mul_ps(normal, _mm_sqrt_ps(_mm_set1_ps(tmp7))));

		out_refract = _mm_blend_ps(__m128_NORM3(out_refract), _mm_setzero_ps(), 0x1);

		ssehit n = sse_intersect(scn, position, out_refract, hitt.depth, hitt.internaldepth + 1, hitt.alpha, -SMALL);

		sse_interact_photons(scn, n, color, generator, distribution);
	}
}

void sse_interact_photons(sse_scene* scn, ssehit& hitt, __m128& color, mt19937& generator, uniform_real_distribution<float>& distribution)
{
	if (hitt.obj == UINT_MAX) return;

	__m128 position = _mm_add_ps(hitt.start, __m128_MUL_float_set(hitt.dir, hitt.t));

	__m128 photondirection = __m128_NEG(hitt.dir);

	__m128 diffuse;
	__m128 specular;
	float index = 1.0f;
	__m128 obj_alpha;

	float Dd;

	int cmp[4];

	__m128 normal;
	if (hitt.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[hitt.obj];

		normal = _mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - hitt.u - hitt.v), _mm_add_ps(__m128_MUL_float_set(triprop->norm1, hitt.u), __m128_MUL_float_set(triprop->norm2, hitt.v)));

		diffuse = triprop->diffuse;
		specular = triprop->specular;

		_mm_store_ss(&index, triprop->emission);

		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		__m128_DOT3(photondirection, normal, Dd);

		if (Dd > 0.0f) { // avoid points not facing the light

			ssetriangle* tri = &scn->triangles[hitt.obj];

			__m128 colort = _mm_mul_ps(color, _mm_mul_ps(__m128_MUL_float_set(diffuse, Dd), obj_alpha));

			_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), colort));

			if (cmp[1] || cmp[2] || cmp[3]) {

				__m128 triT1 = tri->T1;
				__m128 triT2 = tri->T2;
				__m128 tript0 = tri->pt0;
				__m128 trinorm = triprop->norm;

				add_photon_triangle(colort, triprop, hitt.u, hitt.v);

				// partial solution to intensity drop at edges and vertices
				mirror_side_triangle(_mm_add_ps(tript0, triT1), _mm_sub_ps(triT2, triT1), trinorm, scn->triangle_properties[hitt.obj].perp0, triprop, colort, position, scn, tri);

				if (hitt.u*triprop->T1_lenOloc < 1.0f) { // close to pt0 - T2 side
					mirror_side_triangle(tript0, trinorm, triT2, scn->triangle_properties[hitt.obj].perp1, triprop, colort, position, scn, tri);

					mirror_point_triangle(_mm_add_ps(tript0, triT2), triprop, colort, position, scn, tri);

					if (hitt.v*triprop->T2_lenOloc < 1.0f) // close to p0 - T1 side
						mirror_point_triangle(tript0, triprop, colort, position, scn, tri);
				}

				if (hitt.v*triprop->T2_lenOloc < 1.0f) { // close to p0 - T1 side
					mirror_side_triangle(tript0, triT1, trinorm, scn->triangle_properties[hitt.obj].perp2, triprop, colort, position, scn, tri);

					mirror_point_triangle(_mm_add_ps(tript0, triT1), triprop, colort, position, scn, tri);
				}
			}
		} else return;
	}
	else if (hitt.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[hitt.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - hitt.u)*(1.0f - hitt.v)), __m128_MUL_float_set(parprop->norm3, hitt.u*hitt.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - hitt.u)*hitt.v), __m128_MUL_float_set(parprop->norm1, hitt.u*(1.0f - hitt.v))));

		diffuse = parprop->diffuse;
		specular = parprop->specular;

		_mm_store_ss(&index, parprop->emission);

		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		__m128_DOT3(photondirection, normal, Dd);

		if (Dd > 0.0f) { // avoid points not facing the light

			sseparallelogram* par = &scn->parallelograms[hitt.obj];

			__m128 colorp = _mm_mul_ps(color, _mm_mul_ps(__m128_MUL_float_set(diffuse, Dd), obj_alpha));

			_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), colorp));

			if (cmp[1] || cmp[2] || cmp[3]) {

				__m128 parT1 = par->T1;
				__m128 parT2 = par->T2;
				__m128 parpt0 = par->pt0;
				__m128 parnorm = parprop->norm;

				add_photon_parallelogram(colorp, parprop, hitt.u, hitt.v);

				// partial solution to intensity drop at edges and vertices, works best for rectangles
				if (hitt.u*parprop->T1_lenOloc < 1.0f) { // close to pt0 - T2 side
					mirror_side_parallelogram(parpt0, parnorm, parT2, scn->parallelogram_properties[hitt.obj].perp2, parprop, colorp, position, scn, par);

					if (hitt.v*parprop->T2_lenOloc < 1.0f) // close to p0 - T1 side
						mirror_point_parallelogram(parpt0, parprop, colorp, position, scn, par);
				}

				if (hitt.v*parprop->T2_lenOloc < 1.0f) { // close to p0 - T1 side
					mirror_side_parallelogram(parpt0, parT1, parnorm, scn->parallelogram_properties[hitt.obj].perp1, parprop, colorp, position, scn, par);

					if ((1.0f - hitt.u)*parprop->T1_lenOloc < 1.0f) // close to T1 - op side
						mirror_point_parallelogram(_mm_add_ps(parpt0, parT1), parprop, colorp, position, scn, par);
				}

				if ((1.0f - hitt.u)*parprop->T1_lenOloc < 1.0f) { // close to T1 - op side
					mirror_side_parallelogram(_mm_add_ps(parpt0, parT1), parT2, parnorm, __m128_NEG(scn->parallelogram_properties[hitt.obj].perp2), parprop, colorp, position, scn, par);

					if ((1.0f - hitt.v)*parprop->T2_lenOloc < 1.0f) // close to T2 - op side
						mirror_point_parallelogram(_mm_add_ps(parpt0, _mm_add_ps(parT1, parT2)), parprop, colorp, position, scn, par);
				}

				if ((1.0f - hitt.v)*parprop->T2_lenOloc < 1.0f) { // close to T2 - op side
					mirror_side_parallelogram(_mm_add_ps(parpt0, parT2), parnorm, parT1, __m128_NEG(scn->parallelogram_properties[hitt.obj].perp1), parprop, colorp, position, scn, par);

					if (hitt.u*parprop->T1_lenOloc < 1.0f) // close to pt0 - T2 side
						mirror_point_parallelogram(_mm_add_ps(parpt0, parT2), parprop, colorp, position, scn, par);
				}
			}
		} else return;
	}
	else if (hitt.type == SPHERE) {

		ssesphere* sph = &scn->spheres[hitt.obj];

		normal = __m128_NORM3(_mm_sub_ps(position, sph->pos) );

		ssesphere_property* sphprop = &scn->sphere_properties[hitt.obj];

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&index, sphprop->emission);

		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		__m128_DOT3(photondirection, normal, Dd);

		if (Dd > 0.0f) { // avoid points not facing the light

			__declspec(align(16)) float tmp[8];

			_mm_store_ps(tmp+4, position);

			_mm_storer_ps(tmp, _mm_mul_ps(color, _mm_mul_ps(__m128_MUL_float_set(diffuse, Dd), obj_alpha)));

			ssephoton pht = {{tmp[0], tmp[1], tmp[2], tmp[5], tmp[6], tmp[7]}};

			if (tmp[5] || tmp[6] || tmp[7]) {

				//omp_set_lock(&sphprop->lock);
				#pragma omp critical (sph)
				{
					#pragma omp flush
					sphprop->photons.push_back(pht);

					#pragma omp flush
				}
				//omp_unset_lock(&sphprop->lock);

			}
		} else return;
	}
	else if (hitt.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[hitt.obj];

		// this might be able to be done faster
		normal = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, position), sph->pos);
		normal = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normal );
		normal = __m128_NORM3(normal);

		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[hitt.obj];

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&index, sphprop->emission);

		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		__m128_DOT3(photondirection, normal, Dd);

		if (Dd > 0.0f) { // avoid points not facing the light

			__declspec(align(16)) float tmp[8];

			_mm_store_ps(tmp+4, position);

			_mm_storer_ps(tmp, _mm_mul_ps(color, _mm_mul_ps(__m128_MUL_float_set(diffuse, Dd), obj_alpha)));

			if (tmp[0] || tmp[1] || tmp[2]) {

				ssephoton pht = {{tmp[0], tmp[1], tmp[2], tmp[5], tmp[6], tmp[7]}};

				//omp_set_lock(&sphprop->lock);
				#pragma omp critical (arb_sph)
				{
					#pragma omp flush
					sphprop->photons.push_back(pht);

					#pragma omp flush
				}
				//omp_unset_lock(&sphprop->lock);

			}
		} else return;
	} else return;

	
	if (hitt.depth < scn->max_photon_depth) {

		// mirror reflections
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specular));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(photondirection, normal), normal)), photondirection);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			ssehit mir = sse_intersect(scn, position, mirror, hitt.depth + 1, hitt.internaldepth, 1.0f, -SMALL);

			sse_interact_photons(scn, mir, _mm_mul_ps(color, specular), generator, distribution);
		}

		
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), diffuse));

		if (cmp[1] || cmp[2] || cmp[3]) {

			// diffuse
			for (unsigned int l = 0; l < scn->num_of_diffuse; l++)
			{
				__m128 diff = sse_random_shoot(generator, distribution);
				diff = __m128_NORM3(diff);

				float x1;
				__m128_DOT3(diff, normal, x1);
				if (x1 < 0.0f)
					diff = __m128_NEG(diff);

				ssehit hitm = sse_intersect(scn, position, diff, hitt.depth + 1, hitt.internaldepth, 1.0f, -SMALL);

				sse_interact_photons(scn, hitm, __m128_MUL_float_set(_mm_mul_ps(color, _mm_mul_ps(diffuse, obj_alpha)),  Dd/ (float)scn->num_of_diffuse), generator, distribution);
			}


			// refractions
			float tmp1;
			_mm_store_ss(&tmp1, obj_alpha);
			
			if ( (1.0f - tmp1) > 1e-6f && hitt.internaldepth < scn->max_photon_depth) {

				// entering material, if it fails there is no transmittance
				float tmp4;
				__m128_DOT3(photondirection, normal, tmp4);
				float tmp5 = 1 - (1 - tmp4*tmp4)*air_refracive_index*air_refracive_index/(index*index);

				if (tmp5 > 1e-6f) {
			
					__m128 in_refract = __m128_MUL_float_set(_mm_sub_ps(_mm_mul_ps(__m128_DOT3___m128(photondirection, normal), normal), photondirection), air_refracive_index/index);

					in_refract = _mm_sub_ps(in_refract, _mm_mul_ps(normal, _mm_sqrt_ps(_mm_set1_ps(tmp5))));

					in_refract = _mm_blend_ps(__m128_NORM3(in_refract), _mm_setzero_ps(), 0x1);

					// assumes photon starts outside of all objects
					// assumes first internal hit after entering an object corresponds to leaving that object and returning to the air
					ssehit mi = sse_internal_intersect(scn, position, in_refract, hitt.depth, hitt.internaldepth, hitt.alpha, SMALL); // internal depth increased in internal interact
	
					sse_internal_interact_photons(scn, mi, _mm_mul_ps(_mm_mul_ps(color, diffuse), _mm_sub_ps(_mm_set1_ps(1.0f), obj_alpha)), generator, distribution);
				}
			}
		}
	}
}


__m128 sse_internal_photon_shade(sse_scene* scn, ssehit& m) {

	unsigned int i, k;

	if (m.obj == UINT_MAX) return scn->background;

	__m128 colorn = _mm_setzero_ps();

	// internal ray
	__m128 eyedirectionm = __m128_NEG(m.dir);

	__m128 mypositionm = _mm_add_ps(m.start, __m128_MUL_float_set(m.dir, m.t));

	__m128 normalm = _mm_setzero_ps();

	__m128 specularm = _mm_setzero_ps(); 

	float indexm = 1.0f; // index is hidden in the alpha term for emission
	float obj_alpham = 1.0f;
	float shininessm = 1.0f; // shininess hidden in the alpha term of specular

	if (m.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[m.obj];

		normalm = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - m.u - m.v), __m128_MUL_float_set(triprop->norm1, m.u)), __m128_MUL_float_set(triprop->norm2, m.v));
				
		specularm = triprop->specular;
		_mm_store_ss(&shininessm,  specularm);

		_mm_store_ss(&indexm,  triprop->emission);

		_mm_store_ss(&obj_alpham, triprop->diffuse);
	}
	else if (m.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[m.obj];

		normalm = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - m.u)*(1.0f - m.v)), __m128_MUL_float_set(parprop->norm3, m.u*m.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - m.u)*m.v), __m128_MUL_float_set(parprop->norm1, m.u*(1.0f - m.v))));


		specularm = parprop->specular;
		_mm_store_ss(&shininessm,  specularm);

		_mm_store_ss(&indexm,  parprop->emission);

		_mm_store_ss(&obj_alpham, parprop->diffuse);
	}
	else if (m.type == SPHERE) {

		ssesphere* sph = &scn->spheres[m.obj];

		normalm = _mm_sub_ps( mypositionm, sph->pos);
		normalm = __m128_NORM3(normalm);

		ssesphere_property* sphprop = &scn->sphere_properties[m.obj];

		specularm = sphprop->specular;
		_mm_store_ss(&shininessm,  specularm);

		_mm_store_ss(&indexm,  sphprop->emission);

		_mm_store_ss(&obj_alpham, sphprop->diffuse);
	}
	else if (m.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[m.obj];

		// this might be able to be done faster
		normalm = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, mypositionm), sph->pos); // sph->pos has r^2 in w
		normalm = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normalm );
		normalm = __m128_NORM3(normalm);

		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[m.obj];

		specularm = sphprop->specular;
		_mm_store_ss(&shininessm,  specularm);

		_mm_store_ss(&indexm,  sphprop->emission);

		_mm_store_ss(&obj_alpham, sphprop->diffuse);
	} else return colorn;
	// as hiting the back of the object
	normalm = __m128_NEG(normalm);

	// if if fails there is no transmittance, only reflectance
	float tmp6;
	__m128_DOT3(eyedirectionm, normalm, tmp6);
	float tmp7 = 1 - (1 - tmp6*tmp6) * indexm*indexm/(air_refracive_index*air_refracive_index);

	float tmp8 = m.alpha - obj_alpham * m.alpha;

	if (tmp7 > 1e-6f) {

		__m128 out_refract = __m128_MUL_float_set(_mm_sub_ps(_mm_mul_ps(__m128_DOT3___m128(eyedirectionm, normalm), normalm), eyedirectionm), indexm/air_refracive_index);

		out_refract = _mm_sub_ps(out_refract, _mm_mul_ps(normalm, _mm_sqrt_ps(_mm_set1_ps(tmp7))));

		out_refract = _mm_blend_ps(__m128_NORM3(out_refract), _mm_setzero_ps(), 0x1);

		ssehit n = sse_intersect(scn, mypositionm, out_refract, m.depth, m.internaldepth, tmp8, -SMALL);

		colorn = _mm_add_ps(colorn, sse_photon_shade(scn, n));

		// shade using out_refract as a normal to get internal specular reflections

		// revert back to outward pointing normalm
		normalm = __m128_NEG(normalm);

		for (i = 0; i < scn->lights.size(); ++i) {

			sselight* l = &scn->lights[i];

			__m128 sselatten = _mm_set1_ps(1.0f);

			float ldist = INF;

			__m128 ldirection;

			float type;
			_mm_store_ss(&type, l->pos);

			if (type != 0.0f){
				ldirection = _mm_sub_ps(l->pos, mypositionm);

				__m128_DOT3(ldirection, ldirection, ldist);
				ldist = sqrtf(ldist);

				sselatten = _mm_set1_ps(1.0f/(scn->attenuation[0] + scn->attenuation[1]*ldist + scn->attenuation[2]*ldist*ldist));
			} else if (type == 0.0f){
				ldirection = l->pos; // here pos is really a direction, w = 0
			} else continue;

			ldirection = __m128_NORM3(ldirection);

			float Dd;
			__m128_DOT3(ldirection, normalm, Dd);

			if (Dd >= 0.0f) { // avoid points not facing the light

				bool shadow = sse_shadow_intersect(scn, mypositionm, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

				if (!shadow) {

					float Sd;
					__m128_DOT3(ldirection, out_refract, Sd);

					if (Sd > 0.0f) {
						
						__m128 sseSd = _mm_set1_ps(powf( Sd, shininessm ));

						colorn = _mm_add_ps(colorn, _mm_mul_ps(_mm_set1_ps(tmp8), _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_mul_ps(specularm, sseSd))));

					}
				}
			}
		}

		for (i = 0; i < scn->rec_lights.size(); ++i) {

			sserec_light* l = &scn->rec_lights[i];

			__m128 sselatten = _mm_set1_ps(1.0f);

			float ldist = INF;

			__m128 ldirection;

			__m128 colorrl = _mm_setzero_ps();

			for (k = 0; k < l->pts.size(); ++k) {

				ldirection = _mm_sub_ps(l->pts[k], mypositionm);

				__m128_DOT3(ldirection, ldirection, ldist);
				ldist = sqrtf(ldist);

				sselatten = _mm_set1_ps(1.0f/(scn->attenuation[0] + scn->attenuation[1]*ldist + scn->attenuation[2]*ldist*ldist));

				

				ldirection = __m128_NORM3(ldirection);

				float Dd;
				__m128_DOT3(ldirection, normalm, Dd);

				if (Dd >= 0.0f) { // avoid points not facing the light

					bool shadow = sse_shadow_intersect(scn, mypositionm, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

					if (!shadow) {

						float Sd;
						__m128_DOT3(ldirection, out_refract, Sd);

						if (Sd > 0.0f) {
						
							__m128 sseSd = _mm_set1_ps(powf( Sd, shininessm ));

							colorrl = _mm_add_ps(colorrl, _mm_mul_ps(_mm_set1_ps(tmp8), _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_mul_ps(specularm, sseSd))));

						}
					}
				}
			}

			if (l->pts.size() > 0)
				colorn = _mm_add_ps(colorn, __m128_MUL_float_set(colorrl, pi/(float)l->pts.size()));
		}
	}

	// internal reflections
	if (m.depth < scn->maxdepth && m.internaldepth < scn->maxinternaldepth) {
		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specularm));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirectionm, normalm), normalm)), eyedirectionm);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			// to avoid hitting close geometry start internal reflection rays a bit forward
			ssehit mir = sse_internal_intersect(scn, mypositionm, mirror, m.depth + 1, m.internaldepth + 1, tmp8, SMALL);

			__m128 colorm = sse_internal_photon_shade(scn, mir);

			colorn = _mm_add_ps(colorn, _mm_mul_ps(colorm, specularm));
		}
	}
	return colorn;
}

__m128 sse_photon_shade(sse_scene* scn, ssehit& a) {

	unsigned int i, k, u, v;

	if (a.obj == UINT_MAX) return scn->background;

	__m128 color = _mm_setzero_ps(); 

	__m128 myposition = _mm_add_ps(a.start, __m128_MUL_float_set(a.dir, a.t));

	__m128 eyedirection = __m128_NEG(a.dir);

	__m128 ray_alpha = _mm_set1_ps(a.alpha);

	__m128 specular;
	float shininess = 0.0f;
	float index = 1.0f;
	__m128 obj_alpha;

	__m128 normal;

	__m128 diffuse = _mm_setzero_ps();
	__m128 obj_diffuse;

	__m128 loc_rad = _mm_set1_ps(scn->loc_radius);
	
	__m128 len;
	float len_ft;

	if (a.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[a.obj];

		normal = _mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - a.u - a.v), _mm_add_ps(__m128_MUL_float_set(triprop->norm1, a.u), __m128_MUL_float_set(triprop->norm2, a.v)));

		specular = triprop->specular;
		_mm_store_ss(&shininess,  specular);

		obj_diffuse = triprop->diffuse;

		_mm_store_ss(&index, triprop->emission);

		obj_alpha = _mm_shuffle_ps(triprop->diffuse, triprop->diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		ssephoton*** photons_arr = triprop->photons;

		ssephoton* photons;

		u = (unsigned int)(floorf(a.u*triprop->T1_lenOloc+1.0f));
		v = (unsigned int)(floorf(a.v*triprop->T2_lenOloc+1.0f));

		__m128 pht_pos;
		__m128 pht_color;

		{
			photons = photons_arr[u-1][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u-1][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u-1][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}


			color = _mm_add_ps(color, _mm_mul_ps(ray_alpha, diffuse));
		}

	}
	else if (a.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[a.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - a.u)*(1.0f - a.v)), __m128_MUL_float_set(parprop->norm3, a.u*a.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - a.u)*a.v), __m128_MUL_float_set(parprop->norm1, a.u*(1.0f - a.v))));

		specular = parprop->specular;
		_mm_store_ss(&shininess,  specular);

		obj_diffuse = parprop->diffuse;

		_mm_store_ss(&index, parprop->emission);

		obj_alpha = _mm_shuffle_ps(parprop->diffuse, parprop->diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		ssephoton*** photons_arr = parprop->photons;

		ssephoton* photons;

		u = (unsigned int)(floorf(a.u*parprop->T1_lenOloc+1.0f));
		v = (unsigned int)(floorf(a.v*parprop->T2_lenOloc+1.0f));

		__m128 pht_pos;
		__m128 pht_color;

		{
			photons = photons_arr[u-1][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v-1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u-1][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u-1][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}

			photons = photons_arr[u+1][v+1];

			for (i = 0; i < TILE*TILE; ++i) {

				pht_pos = _mm_loadu_ps(photons[i].color_noitisop + 2);

				len = _mm_sub_ps(myposition, pht_pos);
				len = __m128_LEN3___m128(len);
				_mm_store_ss(&len_ft, len);

				if (len_ft < scn->loc_radius) {
					pht_color = __m128_REVERSE(_mm_loadu_ps(photons[i].color_noitisop));

					diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
				}
			}


			color = _mm_add_ps(color, _mm_mul_ps(ray_alpha, diffuse));
		}
	}
	else if (a.type == SPHERE) {

		ssesphere* sph = &scn->spheres[a.obj];

		// this might be able to be done faster
		normal = __m128_NORM3(_mm_sub_ps(myposition, sph->pos) );

		ssesphere_property* sphprop = &scn->sphere_properties[a.obj];

		specular = sphprop->specular;
		_mm_store_ss(&shininess,  specular);

		obj_diffuse = sphprop->diffuse;

		_mm_store_ss(&index, sphprop->emission);

		obj_alpha = _mm_shuffle_ps(sphprop->diffuse, sphprop->diffuse, _MM_SHUFFLE(0, 0, 0, 0));

		aligned_array<ssephoton>* photons = &sphprop->photons;
		//vector<ssephoton>* photons = &sphprop->photons;

		__m128 pht_pos;
		__m128 pht_color;

		for (i = 0; i < photons->size(); ++i) {

			pht_pos = _mm_loadu_ps((*photons)[i].color_noitisop + 2);

			len = _mm_sub_ps(myposition, pht_pos);
			len = __m128_LEN3___m128(len);
			_mm_store_ss(&len_ft, len);

			if (len_ft < scn->loc_radius) {
				pht_color = __m128_REVERSE(_mm_loadu_ps((*photons)[i].color_noitisop));

				diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
			}
		}

		color = _mm_add_ps(color, _mm_mul_ps(ray_alpha, diffuse));
	}
	else if (a.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[a.obj];

		normal = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, myposition), sph->pos);
		normal = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normal );
		normal = __m128_NORM3(normal);

		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[a.obj];

		specular = sphprop->specular;
		_mm_store_ss(&shininess,  specular);

		obj_diffuse = sphprop->diffuse;

		_mm_store_ss(&index, sphprop->emission);

		obj_alpha = _mm_shuffle_ps(sphprop->diffuse, sphprop->diffuse, _MM_SHUFFLE(0, 0, 0, 0));


		aligned_array<ssephoton>* photons = &sphprop->photons;
		//vector<ssephoton>* photons = &sphprop->photons;

		__m128 pht_pos;
		__m128 pht_color;

		for (i = 0; i < photons->size(); ++i) {

			pht_pos = _mm_loadu_ps((*photons)[i].color_noitisop + 2);

			len = _mm_sub_ps(myposition, pht_pos);
			len = __m128_LEN3___m128(len);
			_mm_store_ss(&len_ft, len);

			if (len_ft < scn->loc_radius) {
				pht_color = __m128_REVERSE(_mm_loadu_ps((*photons)[i].color_noitisop));

				diffuse = _mm_add_ps(diffuse, _mm_mul_ps(pht_color, _mm_sub_ps(loc_rad, len)));
			}
		}

		color = _mm_add_ps(color, _mm_mul_ps(ray_alpha, diffuse));

	} else return color;

	for (i = 0; i < scn->lights.size(); ++i) {

		sselight* l = &scn->lights[i];

		__m128 sselatten = _mm_set1_ps(1.0f);

		float ldist = INF;

		__m128 ldirection;

		float type;
		_mm_store_ss(&type, l->pos);

		if (type != 0.0f){
			ldirection = _mm_sub_ps(l->pos, myposition);

			__m128_DOT3(ldirection, ldirection, ldist);
			ldist = sqrtf(ldist);

			sselatten = _mm_set1_ps(1.0f/(scn->attenuation[0] + scn->attenuation[1]*ldist + scn->attenuation[2]*ldist*ldist));
		} else if (type == 0.0f){
			ldirection = l->pos; // here pos is really a direction, w = 0
		} else continue;

		ldirection = __m128_NORM3(ldirection);

		float Dd;
		__m128 sseDd = __m128_DOT3___m128(ldirection, normal);
		_mm_store_ss(&Dd, sseDd);

		if (Dd >= 0.0f) { // avoid points not facing the light

			bool shadow = sse_shadow_intersect(scn, myposition, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

			if (!shadow) {

				__m128 halfvec = __m128_NORM3(_mm_add_ps(ldirection, eyedirection));

				float Sd;
				__m128_DOT3(halfvec, normal, Sd);
				if (Sd > 0.0f) Sd = powf( Sd, shininess );
				else Sd = 0.0f;

				color = _mm_add_ps(color, _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_mul_ps(specular, _mm_set1_ps(Sd))));
			}
		}
	}

	for (i = 0; i < scn->rec_lights.size(); ++i) {

		sserec_light* l = &scn->rec_lights[i];

		__m128 sselatten = _mm_set1_ps(1.0f);

		float ldist = INF;

		__m128 ldirection;

		__m128 colorrl = _mm_setzero_ps();

		for (k = 0; k < l->pts.size(); ++k) {

			ldirection = _mm_sub_ps(l->pts[k], myposition);

			__m128_DOT3(ldirection, ldirection, ldist);
			ldist = sqrtf(ldist);

			sselatten = _mm_set1_ps(1.0f/(scn->attenuation[0] + scn->attenuation[1]*ldist + scn->attenuation[2]*ldist*ldist));

			ldirection = __m128_NORM3(ldirection);

			float Dd;
			__m128 sseDd = __m128_DOT3___m128(ldirection, normal);
			_mm_store_ss(&Dd, sseDd);

			if (Dd >= 0.0f) { // avoid points not facing the light

				bool shadow = sse_shadow_intersect(scn, myposition, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

				if (!shadow) {

					__m128 halfvec = __m128_NORM3(_mm_add_ps(ldirection, eyedirection));

					float Sd;
					__m128_DOT3(halfvec, normal, Sd);
					if (Sd > 0.0f) Sd = powf( Sd, shininess );
					else Sd = 0.0f;

					colorrl = _mm_add_ps(colorrl, _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_mul_ps(specular, _mm_set1_ps(Sd))));
				}
			}
		}

		if (l->pts.size() > 0)
			color = _mm_add_ps(color, __m128_MUL_float_set(colorrl, pi/(float)l->pts.size()));
	}

	// mirror reflections
	if (a.depth < scn->maxdepth) {

		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specular));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirection, normal), normal)), eyedirection);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			ssehit mir = sse_intersect(scn, myposition, mirror, a.depth + 1, a.internaldepth, a.alpha, -SMALL);

			__m128 colorm = sse_photon_shade(scn, mir);

			color = _mm_add_ps(color, _mm_mul_ps(colorm, specular));
		}

	
		// refractions
		float tmp1;
		_mm_store_ss(&tmp1, _mm_mul_ps(obj_alpha, ray_alpha));

		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), obj_diffuse)); // use the objects original difuse, not the one calculated using the photon map

		if (a.internaldepth < scn->maxinternaldepth && tmp1 < a.alpha && (cmp[1] || cmp[2] || cmp[3])) {

			__m128 colorn = _mm_setzero_ps();

			// entering material, if it fails there is no transmittance
			float tmp4;
			__m128_DOT3(eyedirection, normal, tmp4);
			float tmp5 = 1 - (1 - tmp4*tmp4)*air_refracive_index*air_refracive_index/(index*index);

			if (tmp5 > 1e-6f) {
			
				__m128 in_refract = __m128_MUL_float_set(_mm_sub_ps(_mm_mul_ps(__m128_DOT3___m128(eyedirection, normal), normal), eyedirection), air_refracive_index/index);

				in_refract = _mm_sub_ps(in_refract, _mm_mul_ps(normal, _mm_sqrt_ps(_mm_set1_ps(tmp5))));

				in_refract = _mm_blend_ps(__m128_NORM3(in_refract), _mm_setzero_ps(), 0x1);

				// assumes eye starts outside of all objects
				// assumes first internal hit after entering an object corresponds to leaving that object and returning to the air
				ssehit mi = sse_internal_intersect(scn, myposition, in_refract, a.depth + 1, a.internaldepth + 1, a.alpha - tmp1, SMALL);

				colorn = sse_internal_photon_shade(scn, mi);

				color = _mm_add_ps(color, _mm_mul_ps(colorn, obj_diffuse)); // use the objects original difuse, not the one calculated using the photon map
			}
		}
	}
	
	return color;
}
