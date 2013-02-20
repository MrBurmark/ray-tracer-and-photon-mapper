
#include "variables.h"
#include "intersect.h"
#include "kd_tree.h"
#include "shader.h"

vec3 shade(scene* scn, hit a) {

	unsigned int i;

	if (a.obj == UINT_MAX) return scn->background;

	vec3  color(0.0f); 

	vec3 myposition = a.start + a.dir*a.t;

	vec3 eyedirection = -a.dir;

	float ray_alpha = a.alpha;

	vec3 diffuse(0.0f);
	vec3 specular(0.0f);
	float shininess = 0.0f;
	float index = 1.0f;
	float obj_alpha = 1.0f;

	vec3 normal(0.0f);
	if (a.type == TRIANGLE) {

		triangle_property triprop = scn->triangle_properties[a.obj];

		normal = triprop.norm0*(1.0f - a.u - a.v) + triprop.norm1*a.u + triprop.norm2*a.v;

		color += triprop.ambient + triprop.emission;

		diffuse = triprop.diffuse;
		specular = triprop.specular;
		shininess = triprop.shininess;

		obj_alpha = triprop.alpha;

		index = triprop.index;
	}
	else if (a.type == PARALLELOGRAM) {

		parallelogram_property parprop = scn->parallelogram_properties[a.obj];

		normal = glm::normalize(parprop.norm0*(1.0f - a.u)*(1.0f - a.v) + parprop.norm2*(1.0f - a.u)*a.v + parprop.norm1*a.u*(1.0f - a.v) + parprop.norm3*a.u*a.v);

		color += parprop.ambient + parprop.emission;

		diffuse = parprop.diffuse;
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

		color += sphprop.ambient + sphprop.emission;

		diffuse = sphprop.diffuse;
		specular = sphprop.specular;
		shininess = sphprop.shininess;

		obj_alpha = sphprop.alpha;

		index = sphprop.index;
	}
	else if (a.type == ARBITRARY_SPHERE) {

		arbitrary_sphere sph = scn->arbitrary_spheres[a.obj];

		// this might be able to be done faster
		normal = glm::normalize( glm::transpose(mat3(sph.inversetransform)) * (vec3(sph.inversetransform * vec4(myposition, 1.0f)) - sph.pos) );

		sphere_property sphprop = scn->arbitrary_sphere_properties[a.obj];

		color += sphprop.ambient + sphprop.emission;

		diffuse = sphprop.diffuse;
		specular = sphprop.specular;
		shininess = sphprop.shininess;

		obj_alpha = sphprop.alpha;

		index = sphprop.index;
	} else return color;

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

				color += latten * l.color * (obj_alpha*ray_alpha * diffuse*Dd + specular*Sd);
			}
		}
	}

	// mirror reflections
	if (a.depth < scn->maxdepth && glm::length(specular) > 0.0f) {

		vec3 mirror = glm::normalize( 2.0f*glm::dot(eyedirection, normal)*normal - eyedirection );

		hit m = intersect(scn, myposition, mirror, a.depth + 1, 1.0f, -SMALL);

		vec3 colorm = shade(scn, m);

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

__m128 sse_internal_shade(sse_scene* scn, ssehit& m) {

	unsigned int i;

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
		normalm = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, mypositionm), sph->pos);
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

		colorn = _mm_add_ps(colorn, sse_shade(scn, n));

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
	}

	// internal reflections
	if (m.depth < scn->maxdepth && m.internaldepth < scn->maxinternaldepth) {
		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specularm));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirectionm, normalm), normalm)), eyedirectionm);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			ssehit mir = sse_internal_intersect(scn, mypositionm, mirror, m.depth + 1, m.internaldepth + 1, tmp8, -SMALL);

			__m128 colorm = sse_internal_shade(scn, mir);

			colorn = _mm_add_ps(colorn, _mm_mul_ps(colorm, specularm));
		}
	}
	return colorn;
}


__m128 sse_shade(sse_scene* scn, ssehit& a) {

	unsigned int i;

	if (a.obj == UINT_MAX) return scn->background;

	__m128  color = _mm_setzero_ps(); 

	__m128 myposition = _mm_add_ps(a.start, __m128_MUL_float_set(a.dir, a.t));

	__m128 eyedirection = __m128_NEG(a.dir);

	__m128 ray_alpha = _mm_set1_ps(a.alpha);

	__m128 diffuse = _mm_setzero_ps(); 
	__m128 specular = _mm_setzero_ps(); 
	float shininess = 0.0f; // shininess hidden in the alpha term of specular
	float index = 1.0f; // index is hidden in the alpha term for emission
	__m128 obj_alpha = _mm_set1_ps(1.0f); // alpha is hidden within the alpha term of diffuse

	__m128 normal = _mm_setzero_ps(); 
	if (a.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[a.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - a.u - a.v), __m128_MUL_float_set(triprop->norm1, a.u)), __m128_MUL_float_set(triprop->norm2, a.v));

		color = _mm_add_ps(color, _mm_add_ps(triprop->ambient, triprop->emission));

		diffuse = triprop->diffuse;
		specular = triprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  triprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	}
	else if (a.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[a.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - a.u)*(1.0f - a.v)), __m128_MUL_float_set(parprop->norm3, a.u*a.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - a.u)*a.v), __m128_MUL_float_set(parprop->norm1, a.u*(1.0f - a.v))));

		color = _mm_add_ps(color, _mm_add_ps(parprop->ambient, parprop->emission));

		diffuse = parprop->diffuse;
		specular = parprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  parprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	}
	else if (a.type == SPHERE) {

		ssesphere* sph = &scn->spheres[a.obj];
		ssesphere_property* sphprop = &scn->sphere_properties[a.obj];

		normal = _mm_sub_ps(myposition, sph->pos);
		normal = __m128_NORM3(normal);

		color = _mm_add_ps(color, _mm_add_ps(sphprop->ambient, sphprop->emission));

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  sphprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	} else if (a.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[a.obj];
		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[a.obj];

		// this might be able to be done faster
		normal = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, myposition), sph->pos);
		normal = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normal );
		normal = __m128_NORM3(normal);

		color = _mm_add_ps(color, _mm_add_ps(sphprop->ambient, sphprop->emission));

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  sphprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	}else return color;

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

				color = _mm_add_ps(color, _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_add_ps(_mm_mul_ps(_mm_mul_ps(obj_alpha, ray_alpha), _mm_mul_ps(diffuse, sseDd)), _mm_mul_ps(specular, _mm_set1_ps(Sd)))));
			}

		}
	}

	// mirror reflections
	if (a.depth < scn->maxdepth) {
		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specular));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirection, normal), normal)), eyedirection);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			ssehit mir = sse_intersect(scn, myposition, mirror, a.depth + 1, a.internaldepth, a.alpha, -SMALL);

			__m128 colorm = sse_shade(scn, mir);

			color = _mm_add_ps(color, _mm_mul_ps(colorm, specular));
		}
	}

	// refractions
	float tmp1;
	float tmp2;
	_mm_store_ss(&tmp1, _mm_mul_ps(obj_alpha, ray_alpha));
	_mm_store_ss(&tmp2, ray_alpha);

	int cmp[4];
	_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), diffuse));

	if (a.depth < scn->maxdepth && a.internaldepth < scn->maxinternaldepth && tmp1 < tmp2 && (cmp[1] || cmp[2] || cmp[3])) {

		__m128 colorn = _mm_setzero_ps();

		// entering material, if it fails there is no transmittance
		float tmp4;
		__m128_DOT3(eyedirection, normal, tmp4);
		float tmp5 = 1 - (1 - tmp4*tmp4)*air_refracive_index*air_refracive_index/(index*index);

		if (tmp5 > 1e-6f) {
			
			__m128 in_refract = __m128_MUL_float_set(_mm_sub_ps(_mm_mul_ps(__m128_DOT3___m128(eyedirection, normal), normal), eyedirection), air_refracive_index/index);

			in_refract = _mm_sub_ps(in_refract, _mm_mul_ps(normal, _mm_sqrt_ps(_mm_set1_ps(tmp5))));

			in_refract = _mm_blend_ps(__m128_NORM3(in_refract), _mm_setzero_ps(), 0x1);

			ssehit m = sse_internal_intersect(scn, myposition, in_refract, a.depth + 1, a.internaldepth + 1, a.alpha - tmp1, SMALL);

			colorn = sse_internal_shade(scn, m);

		} else colorn = scn->background;

		color = _mm_add_ps(color, _mm_mul_ps(colorn, diffuse));
	}


	return color;
}




__m128 sse_internal_shade_kd(sse_scene* scn, ssehit& m) {

	unsigned int i;

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

		ssehit n = sse_intersect_kd(scn, mypositionm, out_refract, m.depth, m.internaldepth, tmp8, -SMALL);

		colorn = _mm_add_ps(colorn, sse_shade_kd(scn, n));

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

				bool shadow = sse_shadow_intersect_kd(scn, mypositionm, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

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
	}

	// internal reflections
	if (m.depth < scn->maxdepth && m.internaldepth < scn->maxinternaldepth) {
		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specularm));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirectionm, normalm), normalm)), eyedirectionm);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			// to avoid hitting close geometry start internal reflection rays a bit forward
			ssehit mir = sse_internal_intersect_kd(scn, mypositionm, mirror, m.depth + 1, m.internaldepth + 1, tmp8, SMALL);

			__m128 colorm = sse_internal_shade_kd(scn, mir);

			colorn = _mm_add_ps(colorn, _mm_mul_ps(colorm, specularm));
		}
	}
	return colorn;
}


__m128 sse_shade_kd(sse_scene* scn, ssehit& a) {

	unsigned int i;

	if (a.obj == UINT_MAX) return scn->background;

	__m128  color = _mm_setzero_ps(); 

	__m128 myposition = _mm_add_ps(a.start, __m128_MUL_float_set(a.dir, a.t));

	__m128 eyedirection = __m128_NEG(a.dir);

	__m128 ray_alpha = _mm_set1_ps(a.alpha);

	__m128 diffuse = _mm_setzero_ps(); 
	__m128 specular = _mm_setzero_ps(); 
	float shininess = 0.0f; // shininess hidden in the alpha term of specular
	float index = 1.0f; // index is hidden in the alpha term for emission
	__m128 obj_alpha = _mm_set1_ps(1.0f); // alpha is hidden within the alpha term of diffuse

	__m128 normal = _mm_setzero_ps(); 
	if (a.type == TRIANGLE) {

		ssetriangle_property* triprop = &scn->triangle_properties[a.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(triprop->norm0, 1.0f - a.u - a.v), __m128_MUL_float_set(triprop->norm1, a.u)), __m128_MUL_float_set(triprop->norm2, a.v));

		color = _mm_add_ps(color, _mm_add_ps(triprop->ambient, triprop->emission));

		diffuse = triprop->diffuse;
		specular = triprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  triprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	}
	else if (a.type == PARALLELOGRAM) {

		sseparallelogram_property* parprop = &scn->parallelogram_properties[a.obj];

		normal = _mm_add_ps(_mm_add_ps(__m128_MUL_float_set(parprop->norm0, (1.0f - a.u)*(1.0f - a.v)), __m128_MUL_float_set(parprop->norm3, a.u*a.v)), _mm_add_ps(__m128_MUL_float_set(parprop->norm2, (1.0f - a.u)*a.v), __m128_MUL_float_set(parprop->norm1, a.u*(1.0f - a.v))));

		color = _mm_add_ps(color, _mm_add_ps(parprop->ambient, parprop->emission));

		diffuse = parprop->diffuse;
		specular = parprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  parprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	}
	else if (a.type == SPHERE) {

		ssesphere* sph = &scn->spheres[a.obj];
		ssesphere_property* sphprop = &scn->sphere_properties[a.obj];

		normal = _mm_sub_ps(myposition, sph->pos);
		normal = __m128_NORM3(normal);

		color = _mm_add_ps(color, _mm_add_ps(sphprop->ambient, sphprop->emission));

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  sphprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
	} else if (a.type == ARBITRARY_SPHERE) {

		ssearbitrary_sphere* sph = &scn->arbitrary_spheres[a.obj];
		ssesphere_property* sphprop = &scn->arbitrary_sphere_properties[a.obj];

		// this might be able to be done faster
		normal = _mm_sub_ps(fmat4_MUL3___m128(sph->inversetransform, myposition), sph->pos);
		normal = fmat4_MUL3___m128(fmat4_transp(sph->inversetransform), normal );
		normal = __m128_NORM3(normal);

		color = _mm_add_ps(color, _mm_add_ps(sphprop->ambient, sphprop->emission));

		diffuse = sphprop->diffuse;
		specular = sphprop->specular;

		_mm_store_ss(&shininess,  specular);
		_mm_store_ss(&index,  sphprop->emission);
		obj_alpha = _mm_shuffle_ps(diffuse, diffuse, _MM_SHUFFLE(0, 0, 0, 0));
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

			bool shadow = sse_shadow_intersect_kd(scn, myposition, ldirection, ldist, -SMALL); // rays do not sse_intersect the rear of faces, so there is no problem if the ray starts below a surface by a little bit

			if (!shadow) {

				__m128 halfvec = __m128_NORM3(_mm_add_ps(ldirection, eyedirection));

				float Sd;
				__m128_DOT3(halfvec, normal, Sd);
				if (Sd > 0.0f) Sd = powf( Sd, shininess );
				else Sd = 0.0f;

				color = _mm_add_ps(color, _mm_mul_ps(_mm_mul_ps(l->color, sselatten), _mm_add_ps(_mm_mul_ps(_mm_mul_ps(obj_alpha, ray_alpha), _mm_mul_ps(diffuse, sseDd)), _mm_mul_ps(specular, _mm_set1_ps(Sd)))));
			}

		}
	}
	
	// mirror reflections
	if (a.depth < scn->maxdepth) {
		int cmp[4];
		_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), specular));

		if (cmp[1] || cmp[2] || cmp[3]) {

			__m128 mirror = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(__m128_DOT3___m128(eyedirection, normal), normal)), eyedirection);
			mirror = _mm_blend_ps(__m128_NORM3(mirror), _mm_setzero_ps(), 0x1);

			ssehit mir = sse_intersect_kd(scn, myposition, mirror, a.depth + 1, a.internaldepth, a.alpha, -SMALL);

			__m128 colorm = sse_shade_kd(scn, mir);

			color = _mm_add_ps(color, _mm_mul_ps(colorm, specular));
		}
	}
	
	// refractions
	float tmp1;
	float tmp2;
	_mm_store_ss(&tmp1, _mm_mul_ps(obj_alpha, ray_alpha));
	_mm_store_ss(&tmp2, ray_alpha);

	int cmp[4];
	_mm_storeu_ps((float*)cmp, _mm_cmpneq_ps(_mm_setzero_ps(), diffuse));

	if (a.depth < scn->maxdepth && a.internaldepth < scn->maxinternaldepth && tmp1 < tmp2 && (cmp[1] || cmp[2] || cmp[3])) {

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
			ssehit mi = sse_internal_intersect_kd(scn, myposition, in_refract, a.depth + 1, a.internaldepth + 1, a.alpha - tmp1, SMALL);
	
			
			/*

			// make refractions for objects within objects work properly

			// mi.dir = _mm_blend_ps(mi.dir, _mm_set1_ps(index), 0x1); // hide the index of the hit item in the w component of the direction

			*/

			colorn = sse_internal_shade_kd(scn, mi);

		}

		color = _mm_add_ps(color, _mm_mul_ps(colorn, diffuse));
	}
	
	return color;
}

