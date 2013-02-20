
#include <stdio.h>
#include <math.h>

#include "variables.h"
#include "kd_tree.h"


hit intersect_objs(scene* scn, vec3 S, vec3 D) {

	unsigned int i;

	hit out = {S, D, UINT_MAX, UINT_MAX, UINT_MAX,  INF, INF, INF, 1.0f};

	for (i = 0; i < scn->triangles.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

		vec3 T1 = scn->triangles[i].T1;
		vec3 T2 = scn->triangles[i].T2;
		
		vec3 V = glm::cross(T1, D);

		float denom = glm::dot(V, T2);

		// avoid hitting the backs of triangles or triangles too close to parallel with the ray
		if (denom > 1e-6f) {

			vec3 V0 = scn->triangles[i].pt0;
		
			float denomInv = 1.0f/denom;

			vec3 DT = S - V0;

			float v = glm::dot(V, DT) * denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				vec3 W = glm::cross(T2, DT);

				float u = glm::dot(W, D) * denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t = glm::dot(W, T1) * denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = TRIANGLE;
						out.obj = i;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->parallelograms.size(); ++i) {
		
		vec3 T1 = scn->parallelograms[i].T1;
		vec3 T2 = scn->parallelograms[i].T2;
		
		vec3 V = glm::cross(T1, D);

		float denom = glm::dot(V, T2);

		// avoid hitting the backs of parallelograms or parallelograms too close to parallel with the ray
		if (denom > 1e-6f) {

			vec3 V0 = scn->parallelograms[i].pt0;
		
			float denomInv = 1.0f/denom;

			vec3 DT = S - V0;

			float v = glm::dot(V, DT) * denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				vec3 W = glm::cross(T2, DT);

				float u = glm::dot(W, D) * denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t = glm::dot(W, T1) * denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = PARALLELOGRAM;
						out.obj = i;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->spheres.size(); ++i) {

		vec3 pos = scn->spheres[i].pos;
		float r2 = scn->spheres[i].radius2;

		float a = glm::dot(D, D);
		float b = 2.0f*glm::dot(D, S-pos);
		float c = glm::dot(S-pos, S-pos) - r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}

	for (i = 0; i < scn->arbitrary_spheres.size(); ++i) {

		vec3 S0 = vec3(scn->arbitrary_spheres[i].inversetransform * vec4(S, 1.0f));
		vec3 D0 = vec3(scn->arbitrary_spheres[i].inversetransform * vec4(D, 0.0f));

		vec3 pos = scn->arbitrary_spheres[i].pos;
		float r2 = scn->arbitrary_spheres[i].radius2;

		float a = glm::dot(D0, D0);
		float b = 2.0f*glm::dot(D0, S0-pos);
		float c = glm::dot(S0-pos, S0-pos) - r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = ARBITRARY_SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}
	return out;
}

hit intersect (scene* scn, vec3 S0, vec3 D, unsigned int depth, float alpha, float b_amt) {
	
	vec3 S = S0 +  D*b_amt;

	hit hitt = intersect_objs (scn, S, D);

	hitt.depth = depth;
	hitt.alpha = alpha;
	return hitt;
}

hit internal_intersect_objs(scene* scn, vec3 S, vec3 D) {

	unsigned int i;

	hit out = {S, D, UINT_MAX, UINT_MAX, UINT_MAX,  INF, INF, INF, 1.0f};

	for (i = 0; i < scn->triangles.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

		vec3 T1 = scn->triangles[i].T1;
		vec3 T2 = scn->triangles[i].T2;
		
		vec3 V = glm::cross(T1, D);

		float denom = glm::dot(V, T2);

		// avoid hitting the fronts of triangles or triangles too close to parallel with the ray
		if (denom < 1e-6f) {

			vec3 V0 = scn->triangles[i].pt0;
		
			float denomInv = 1.0f/denom;

			vec3 DT = S - V0;

			float v = glm::dot(V, DT) * denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				vec3 W = glm::cross(T2, DT);

				float u = glm::dot(W, D) * denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t = glm::dot(W, T1) * denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = TRIANGLE;
						out.obj = i;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->parallelograms.size(); ++i) {
		
		vec3 T1 = scn->parallelograms[i].T1;
		vec3 T2 = scn->parallelograms[i].T2;
		
		vec3 V = glm::cross(T1, D);

		float denom = glm::dot(V, T2);

		// avoid hitting the fronts of parallelograms or parallelograms too close to parallel with the ray
		if (denom < 1e-6f) {

			vec3 V0 = scn->parallelograms[i].pt0;
		
			float denomInv = 1.0f/denom;

			vec3 DT = S - V0;

			float v = glm::dot(V, DT) * denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				vec3 W = glm::cross(T2, DT);

				float u = glm::dot(W, D) * denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t = glm::dot(W, T1) * denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = PARALLELOGRAM;
						out.obj = i;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->spheres.size(); ++i) {

		vec3 pos = scn->spheres[i].pos;
		float r2 = scn->spheres[i].radius2;

		float a = glm::dot(D, D);
		float b = 2.0f*glm::dot(D, S-pos);
		float c = glm::dot(S-pos, S-pos) - r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, rules out hitting front-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}

	for (i = 0; i < scn->arbitrary_spheres.size(); ++i) {

		vec3 S0 = vec3(scn->arbitrary_spheres[i].inversetransform * vec4(S, 1.0f));
		vec3 D0 = vec3(scn->arbitrary_spheres[i].inversetransform * vec4(D, 0.0f));

		vec3 pos = scn->arbitrary_spheres[i].pos;
		float r2 = scn->arbitrary_spheres[i].radius2;

		float a = glm::dot(D0, D0);
		float b = 2.0f*glm::dot(D0, S0-pos);
		float c = glm::dot(S0-pos, S0-pos) - r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, rules out hitting front-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = ARBITRARY_SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}
	return out;
}

hit internal_intersect (scene* scn, vec3 S0, vec3 D, unsigned int depth, float alpha, float b_amt) {
	
	vec3 S = S0 +  D*b_amt;

	hit hitt = internal_intersect_objs (scn, S, D);

	hitt.depth = depth;
	hitt.alpha = alpha;
	return hitt;
}



ssehit sse_intersect_objs(sse_scene* scn, __m128 S0, __m128 D0) {

	unsigned int i;

	ssehit out = { S0, D0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};

	for (i = 0; i < scn->triangles.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[i].T1;
		__m128 T2 = scn->triangles[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of triangles or triangles too close to parallel with the ray
		if (denom > 1e-6f) {
		
			__m128 V0 = scn->triangles[i].pt0;

			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.obj = i;
						out.type = TRIANGLE;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->parallelograms.size(); ++i) {
		
		__m128 T1 = scn->parallelograms[i].T1;
		__m128 T2 = scn->parallelograms[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of parallelograms or parallelograms too close to parallel with the ray
		if (denom > 1e-6f) {
		
			__m128 V0 = scn->parallelograms[i].pt0;

			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest object yet found
					if (t > 0.0f && t < out.t) {

						out.obj = i;
						out.type = PARALLELOGRAM;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->spheres.size(); ++i) {

		__m128 pos = scn->spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}

	for (i = 0; i < scn->arbitrary_spheres.size(); ++i) {

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = ARBITRARY_SPHERE;
				out.obj = i;
				out.t = t;
			}
		}
	}
	return out;
}


ssehit sse_intersect (sse_scene* scn, __m128 S0, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));

	ssehit hitt = sse_intersect_objs (scn, S, D);

	hitt.depth = depth;
	hitt.alpha = alpha;
	hitt.internaldepth = internaldepth;
	return hitt;
}

bool sse_shadow_intersect_objs(sse_scene* scn, __m128 S0, __m128 D0, float ldist) {

	unsigned int i;

	for (i = 0; i < scn->triangles.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[i].T1;
		__m128 T2 = scn->triangles[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of triangles or triangles too close to parallel with the ray
		if (denom > 1e-6f) { // || denom < 1e-6f) {

			__m128 V0 = scn->triangles[i].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than distance to the light
					if (t > 0.0f && t < ldist) {

						return true;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->parallelograms.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->parallelograms[i].T1;
		__m128 T2 = scn->parallelograms[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of parallelograms or parallelograms too close to parallel with the ray
		if (denom > 1e-6f) { // || denom < 1e-6f) {

			__m128 V0 = scn->parallelograms[i].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than distance to the light
					if (t > 0.0f && t < ldist) {

						return true;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->spheres.size(); ++i) {

		__m128 pos = scn->spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float tl = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);
			//float th = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than the distance to the light
			if ((tl > 0.0f && tl < ldist)) {// || (th > 0.0f && th < ldist)) {

				return true;
			}
		}
	}

	for (i = 0; i < scn->arbitrary_spheres.size(); ++i) {

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float tl = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);
			//float th = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than the distance to the light
			if ((tl > 0.0f && tl < ldist)) {// || (th > 0.0f && th < ldist)) {

				return true;
			}
		}
	}
	return false;
}

bool sse_shadow_intersect (sse_scene* scn, __m128 S0, __m128 D, float ldistance, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));

	float amt = 0.0f;
	
	return sse_shadow_intersect_objs (scn, _mm_add_ps(S, __m128_MUL_float_set(D, amt)), D, ldistance - amt);
}

ssehit sse_intersect_back_objs(sse_scene* scn, __m128 S0, __m128 D0) {

	unsigned int i;

	ssehit out = { S0, D0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};

	for (i = 0; i < scn->triangles.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[i].T1;
		__m128 T2 = scn->triangles[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// hit the backs of triangles and avoid triangles too close to parallel with the ray
		if (denom < 1e-6f) {

			__m128 V0 = scn->triangles[i].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.obj = i;
						out.type = TRIANGLE;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->parallelograms.size(); ++i) {
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->parallelograms[i].T1;
		__m128 T2 = scn->parallelograms[i].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// hit the backs of parallelograms and avoid parallelograms too close to parallel with the ray
		if (denom < 1e-6f) {

			__m128 V0 = scn->parallelograms[i].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.obj = i;
						out.type = PARALLELOGRAM;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = 0; i < scn->spheres.size(); ++i) {

		__m128 pos = scn->spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, hits back-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.obj = i;
				out.type = SPHERE;
				out.t = t;
			}
		}
	}

	for (i = 0; i < scn->arbitrary_spheres.size(); ++i) {

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[i].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[i].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, hits back-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.obj = i;
				out.type = ARBITRARY_SPHERE;
				out.t = t;
			}
		}
	}
	return out;
}


ssehit sse_internal_intersect (sse_scene* scn, __m128 S0, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));

	ssehit hitt = sse_intersect_back_objs (scn, S, D);

	hitt.depth = depth;
	hitt.alpha = alpha;
	hitt.internaldepth = internaldepth;
	return hitt;
}


// assumes S + tD actually hits the box, if goes to inf/INF, returns INF
__m128 sse_collide_inside_bound_box (bound_box box, __m128 S, __m128 D, float offset) {

	__m128 high = _mm_set_ps(box.top_x, box.top_y, box.top_z, 0.0f);
	__m128 low = _mm_set_ps(box.bot_x, box.bot_y, box.bot_z, 0.0f);

	high = _mm_div_ps(_mm_sub_ps(high, S), D);
	low = _mm_div_ps(_mm_sub_ps(low, S), D);

	float t[8];
	_mm_storeu_ps(t, _mm_max_ps(low, high));
	_mm_storeu_ps(t+4, _mm_cmpgt_ps(low, high));

	// in all cases hide t in the w component
	if (t[3] < t[2] ) { // x, z
		if (t[3] < t[1] ) { // x
			// if t[7] hit lower z
			return _mm_blend_ps(_mm_add_ps(_mm_set_ps(t[7] ? -offset : offset, 0.0f, 0.0f, 0.0f), _mm_add_ps(S, _mm_mul_ps(D, _mm_set1_ps(t[3]*(1 + offset))))), _mm_set1_ps(t[3]), 0x1);
		}
		else { // z
			// if t[5] hit lower z
			return _mm_blend_ps(_mm_add_ps(_mm_set_ps(0.0f, 0.0f, t[5] ? -offset : offset, 0.0f), _mm_add_ps(S, _mm_mul_ps(D, _mm_set1_ps(t[1]*(1 + offset))))), _mm_set1_ps(t[1]), 0x1);
		}
	}
	else { // y, z
		if (t[2] < t[1] ) { // y
			// if t[6] hit lower y
			return _mm_blend_ps(_mm_add_ps(_mm_set_ps(0.0f, t[6] ? -offset : offset, 0.0f, 0.0f), _mm_add_ps(S, _mm_mul_ps(D, _mm_set1_ps(t[2]*(1 + offset))))), _mm_set1_ps(t[2]), 0x1);
		}
		else { // z
			// if t[5] hit lower z
			return _mm_blend_ps(_mm_add_ps(_mm_set_ps(0.0f, 0.0f, t[5] ? -offset : offset, 0.0f), _mm_add_ps(S, _mm_mul_ps(D, _mm_set1_ps(t[1]*(1 + offset))))), _mm_set1_ps(t[1]), 0x1);
		}
	}
}


ssehit sse_intersect_kd_objs(sse_scene* scn, __m128 S0, __m128 D0, unsigned int kd_pos) {

	unsigned int i;

	ssehit out = { S0, D0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};

	kd_tree_node* node = scn->kd_tree + kd_pos;

	for (i = 4; i < node->items[1]; ++i) {

		unsigned int j = node->items[i];
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[j].T1;
		__m128 T2 = scn->triangles[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of triangles or triangles too close to parallel with the ray
		if (denom > 1e-6f) {

			__m128 V0 = scn->triangles[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = TRIANGLE;
						out.obj = j;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = node->items[1]; i < node->items[2]; ++i) {

		unsigned int j = node->items[i];

		__m128 T1 = scn->parallelograms[j].T1;
		__m128 T2 = scn->parallelograms[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of parallelograms or parallelograms too close to parallel with the ray
		if (denom > 1e-6f) {

			__m128 V0 = scn->parallelograms[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest object yet found
					if (t > 0.0f && t < out.t) {

						out.type = PARALLELOGRAM;
						out.obj = j;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = node->items[2]; i < node->items[3]; ++i) {

		unsigned int j = node->items[i];

		__m128 pos = scn->spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = SPHERE;
				out.obj = j;
				out.t = t;
			}
		}
	}

	for (i = node->items[3]; i < node->items[0]; ++i) {

		unsigned int j = node->items[i];

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = ARBITRARY_SPHERE;
				out.obj = j;
				out.t = t;
			}
		}
	}

	return out;
}


ssehit sse_intersect_kd (sse_scene* scn, __m128 S0, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));
	
	float t[4];
	ssehit hitt = { S, D, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};
	bound_box box;
	_mm_storeu_ps(t, S);
	t[0] = b_amt;

	while ( t[0] < INF && t[0] < hitt.t ) { // have to check until you know there is nothing else possibly closer to hit, may have to go to further boxes

		//printf("t = %f\n", x[0]);
		
		// use T to find the box you are in
		box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

		// intersect objects in that box
		hitt = sse_intersect_kd_objs(scn, S, D, box.obj);

		float old_t = t[0];

		// find your new box and t, the vector is nudged in the right direction a bit
		_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, SMALL));
		
		// if ray moved backwards, rare, causes the ray to spin between two boxes indefinately
		if (t[0] <= old_t) {

			float amt = SMALL;
			unsigned int old_obj = box.obj;

			// go back to the closer box
			box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

			//printf("whoops\n");

			_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));

			while (kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]).obj == old_obj) {

				// increase amt
				amt *= 1.5f;

				//printf("fix\n");
				
				// try to move to a diferent box than before by increasing the nudge
				_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));
			}
		}
	}

	hitt.start = S;
	//hitt.t += b_amt;
	hitt.depth = depth;
	hitt.alpha = alpha;
	hitt.internaldepth = internaldepth;
	return hitt;
}



bool sse_shadow_intersect_kd_objs(sse_scene* scn, __m128 S0, __m128 D0, unsigned int kd_pos, float ldist) {

	unsigned int i;

	kd_tree_node* node = scn->kd_tree + kd_pos;

	for (i = 4; i < node->items[1]; ++i) {

		unsigned int j = node->items[i];
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[j].T1;
		__m128 T2 = scn->triangles[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of triangles or triangles too close to parallel with the ray
		if (denom > 1e-6f) {

			__m128 V0 = scn->triangles[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the distance to the light
					if (t > 0.0f && t < ldist) {

						return true;
					}
				}
			}
		}
	}

	for (i = node->items[1]; i < node->items[2]; ++i) {

		unsigned int j = node->items[i];

		__m128 T1 = scn->parallelograms[j].T1;
		__m128 T2 = scn->parallelograms[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// avoid hitting the backs of parallelograms or parallelograms too close to parallel with the ray
		if (denom > 1e-6f) {

			__m128 V0 = scn->parallelograms[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the distance to the light
					if (t > 0.0f && t < ldist) {

						return true;
					}
				}
			}
		}
	}

	for (i = node->items[2]; i < node->items[3]; ++i) {

		unsigned int j = node->items[i];

		__m128 pos = scn->spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit is in the right direction and closer than the light
			if (t > 0.0f && t < ldist) {

				return true;
			}
		}
	}

	for (i = node->items[3]; i < node->items[0]; ++i) {

		unsigned int j = node->items[i];

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take smaller of intersections, rules out hitting back-faces
			float t = (-b - sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit is in the right direction and closer than the light
			if (t > 0.0f && t < ldist) {

				return true;
			}
		}
	}

	return false;
}


bool sse_shadow_intersect_kd (sse_scene* scn, __m128 S0, __m128 D, float ldistance, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));

	float t[4];
	bound_box box;
	_mm_storeu_ps(t, S);
	t[0] = b_amt;

	while (t[0] < ldistance) {

		//printf("t = %f\n", x[0]);
		
		// use T to find the box you are in
		box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

		if (sse_shadow_intersect_kd_objs(scn, S, D, box.obj, ldistance)) return true;

		float old_t = t[0];

		// find your next box, the vector is nudged in the right direction a bit
		_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, SMALL));

		// if ray moved backwards, rare, causes the ray to spin between two boxes indefinately
		if (t[0] <= old_t) {

			float amt = SMALL;
			unsigned int old_obj = box.obj;

			//printf("whoops\n");

			// go back to the closer box
			box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

			_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));

			while (kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]).obj == old_obj) {

				// increase amt
				amt *= 1.5f;

				//printf("fix\n");
				
				// try to move to a diferent box than before by increasing the nudge
				_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));
			}
		}
	}

	return false;
}


ssehit sse_internal_intersect_kd_objs(sse_scene*scn, __m128 S0, __m128 D0, unsigned int kd_pos) {

	unsigned int i;

	ssehit out = { S0, D0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};

	kd_tree_node* node = scn->kd_tree + kd_pos;

	for (i = 4; i < node->items[1]; ++i) {

		unsigned int j = node->items[i];
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->triangles[j].T1;
		__m128 T2 = scn->triangles[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// hit the backs of triangles and avoid triangles too close to parallel with the ray
		if (denom < 1e-6f) {

			__m128 V0 = scn->triangles[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the triangle
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the triangle
				if (u >= 0.0f && u <= 1.0f && u + v <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest triangle yet found
					if (t > 0.0f && t < out.t) {

						out.type = TRIANGLE;
						out.obj = j;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = node->items[1]; i < node->items[2]; ++i) {

		unsigned int j = node->items[i];
		
		// using algorithm in http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
		
		__m128 T1 = scn->parallelograms[j].T1;
		__m128 T2 = scn->parallelograms[j].T2;
		
		__m128 V = __m128_CROSS(T1, D0);

		float denom;
		__m128_DOT3(V, T2, denom);

		// hit the backs of parallelograms and avoid parallelograms too close to parallel with the ray
		if (denom < 1e-6f) {

			__m128 V0 = scn->parallelograms[j].pt0;
		
			float denomInv = 1.0f/denom;

			__m128 DT = _mm_sub_ps(S0, V0);

			float v;
			__m128_DOT3(V, DT, v);
			v *= denomInv;

			// check that v lies within the parallelogram
			if (v >= 0.0f && v <= 1.0f) {

				__m128 W = __m128_CROSS(T2, DT);

				float u;
				__m128_DOT3(W, D0, u);
				u *= denomInv;

				// check that u and u + v lie within the parallelogram
				if (u >= 0.0f && u <= 1.0f) {

					float t;
					__m128_DOT3(W, T1, t);
					t *= denomInv;

					// check that t is positive, and less than the closest object yet found
					if (t > 0.0f && t < out.t) {

						out.type = PARALLELOGRAM;
						out.obj = j;
						out.u = u;
						out.v = v;
						out.t = t;
					}
				}
			}
		}
	}

	for (i = node->items[2]; i < node->items[3]; ++i) {

		unsigned int j = node->items[i];

		__m128 pos = scn->spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D0, D0, a);
		float b;
		__m128_DOT3(D0, _mm_sub_ps(S0, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S0, pos), _mm_sub_ps(S0, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, hits back-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = SPHERE;
				out.obj = j;
				out.t = t;
			}
		}
	}

	for (i = node->items[3]; i < node->items[0]; ++i) {

		unsigned int j = node->items[i];

		__m128 S = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, S0);
		__m128 D = fmat4_MUL3___m128(scn->arbitrary_spheres[j].inversetransform, D0);

		__m128 pos = scn->arbitrary_spheres[j].pos;
		float r2;
		_mm_store_ss(&r2, pos);

		float a;
		__m128_DOT3(D, D, a);
		float b;
		__m128_DOT3(D, _mm_sub_ps(S, pos), b);
		b*= 2.0f;
		float c;
		__m128_DOT3(_mm_sub_ps(S, pos), _mm_sub_ps(S, pos), c);
		c -= r2;

		// ensure ball is hit
		if (b*b-4.0f*a*c > 0.0f) {

			// take larger of intersections, hits back-faces
			float t = (-b + sqrtf(b*b-4.0f*a*c))/(2.0f*a);

			// check sphere hit it in front of camera and closer than previous hit
			if (t > 0.0f && t < out.t) {

				out.type = ARBITRARY_SPHERE;
				out.obj = j;
				out.t = t;
			}
		}
	}

	return out;
}

ssehit sse_internal_intersect_kd (sse_scene* scn, __m128 S0, __m128 D, unsigned int depth, unsigned int internaldepth, float alpha, float b_amt) {
	
	__m128 S = _mm_add_ps(S0, _mm_mul_ps(D, _mm_set1_ps(b_amt)));
	
	float t[4];
	ssehit hitt = { S, D, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, INF, INF, INF, 1.0f};
	bound_box box;
	_mm_storeu_ps(t, S);
	t[0] = b_amt;

	while ( t[0] < INF && t[0] < hitt.t ) { // have to check until you know there is nothing else possibly closer to hit, may have to go to further boxes

		//printf("t = %f\n", x[0]);
		
		// use T to find the box you are in
		box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

		// intersect objects in that box
		hitt = sse_internal_intersect_kd_objs(scn, S, D, box.obj);

		float old_t = t[0];

		// find your next box, the vector is nudged in the right direction a bit
		_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, SMALL));

		// if ray moved backwards, rare, causes the ray to spin between two boxes indefinately
		if (t[0] <= old_t) {

			float amt = SMALL;
			unsigned int old_obj = box.obj;

			//printf("whoops\n");

			// go back to the closer box
			box = kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]);

			_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));

			while (kd_find_leaf(scn->kd_tree, t[3], t[2], t[1]).obj == old_obj) {

				// increase amt
				amt *= 1.5f;

				//printf("fix\n");
				
				// try to move to a diferent box than before by increasing the nudge
				_mm_storeu_ps(t, sse_collide_inside_bound_box(box, S, D, amt));
			}
		}
	}

	hitt.start = S;
	//hitt.t += b_amt;
	hitt.depth = depth;
	hitt.alpha = alpha;
	hitt.internaldepth = internaldepth;
	return hitt;
}

