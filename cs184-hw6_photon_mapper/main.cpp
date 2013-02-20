
#define MAIN

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <deque>
#include <stack>
#include <math.h>
#include <FreeImage.h>
#include <omp.h>

#include "variables.h"
#include "readfile.h"
#include "intersect.h"
#include "shader.h"
#include "kd_tree.h"
#include "photon.h"


using namespace std ;

void saveScreenshot(string fname, unsigned char  * pixels, unsigned int w, unsigned int h) {

	FIBITMAP *img = FreeImage_ConvertFromRawBits(pixels, w, h, w * 3, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);

	std::cout << "Saving screenshot: " << fname + ".png" << "\n";

	FreeImage_Save(FIF_PNG, img, (fname + ".png").c_str(), 0);
}


void create_image(scene* scn) {

	printf("Rendering\n");

	vec3 forward = glm::normalize(scn->center-scn->eye);
	vec3 right = glm::normalize(glm::cross(forward, scn->up));
	vec3 down_vec = glm::normalize(glm::cross(forward, right));

	int m = 0;
	float n = scn->inc;

	float J = (float)scn->h;
	float I = (float)scn->w;

	float SY = 2.0f*tanf(scn->fovy*pi/360.0f);
	float SX = scn->w/J * 2.0f*tanf(scn->fovy*pi/360.0f);

	const unsigned int JJ = 8;
	const unsigned int II = 8;

	#pragma omp parallel for schedule(dynamic, 1) num_threads(scn->maxthreads)
	for (int jj = 0; jj < scn->h; jj+=JJ)
		for (unsigned int ii = 0; ii < scn->w; ii+=II) {
			for (unsigned int j = jj; j <jj + JJ && j < scn->h; ++j) {

				// upsied down
				float sy = SY * (J*0.5f - (float)j - 0.5f)/J ;

				for (unsigned int i = ii; i < ii + II && i < scn->w; ++i) {

					// take into account field of view and aspect ratio/differing width/height
					float sx = SX * ((float)i + 0.5f - I*0.5f)/I ;


					vec3 dir = forward + sx*right + sy*down_vec;

					vec3 color = scn->background;

					if (scn->SSAA == 1) {

						hit ha = intersect(scn, scn->eye, glm::normalize(dir), 0, 1.0f, 0.0f);

						color = shade(scn, ha);

					} else if (scn->SSAA == 2) {

						vec3 dn = SY * 0.25f/J * down_vec;
						vec3 r = SX * 0.25f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + r + dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir - r - dn), 0, 1.0f, 0.0f);

						vec3 colora = shade(scn, ha);
						vec3 colorb = shade(scn, hb);

						color = (colora + colorb)/2.0f;

					} else if (scn->SSAA == 4) {

						vec3 dn = SY * 0.25f/J * down_vec;
						vec3 r = SX * 0.25f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + r + dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir - r + dn), 0, 1.0f, 0.0f);
						hit hc = intersect(scn, scn->eye, glm::normalize(dir + r - dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir - r - dn), 0, 1.0f, 0.0f);

						vec3 colora = shade(scn, ha);
						vec3 colorb = shade(scn, hb);
						vec3 colorc = shade(scn, hc);
						vec3 colord = shade(scn, hd);

						color = (colora + colorb + colorc + colord)/4.0f;

					} else if (scn->SSAA == 8) {

						vec3 dn = SY * 0.125f/J * down_vec;
						vec3 r = SX * 0.125f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hc = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);

						hit he = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hf = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hg = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hh = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);

						vec3 colora = shade(scn, ha);
						vec3 colorb = shade(scn, hb);
						vec3 colorc = shade(scn, hc);
						vec3 colord = shade(scn, hd);
						vec3 colore = shade(scn, he);
						vec3 colorf = shade(scn, hf);
						vec3 colorg = shade(scn, hg);
						vec3 colorh = shade(scn, hh);

						color = (colora + colorb + colorc + colord + colore + colorf + colorg + colorh)/8.0f;

					} else if (scn->SSAA == 16) {

						vec3 dn = SY * 0.125f/J * down_vec;
						vec3 r = SX * 0.125f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);
						hit hc = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit he = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hf = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hg = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hh = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);

						hit hi = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hj = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);
						hit hk = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hl = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hm = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hn = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit ho = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hp = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);

						vec3 colora = shade(scn, ha);
						vec3 colorb = shade(scn, hb);
						vec3 colorc = shade(scn, hc);
						vec3 colord = shade(scn, hd);
						vec3 colore = shade(scn, he);
						vec3 colorf = shade(scn, hf);
						vec3 colorg = shade(scn, hg);
						vec3 colorh = shade(scn, hh);
						vec3 colori = shade(scn, hi);
						vec3 colorj = shade(scn, hj);
						vec3 colork = shade(scn, hk);
						vec3 colorl = shade(scn, hl);
						vec3 colorm = shade(scn, hm);
						vec3 colorn = shade(scn, hn);
						vec3 coloro = shade(scn, ho);
						vec3 colorp = shade(scn, hp);

						color = (colora + colorb + colorc + colord + colore + colorf + colorg + colorh + colori + colorj + colork + colorl + colorm + colorn + coloro + colorp)/16.0f;
					}


					// correct for too high or low color values, which will overflow/underflow on typing
					if (color.x > 1.0f) color.x = 1.0f;
					else if (color.x < 0.0f) color.x = 0.0f;
					if (color.y > 1.0f) color.y = 1.0f;
					else if (color.y < 0.0f) color.y = 0.0f;
					if (color.z > 1.0f) color.z = 1.0f;
					else if (color.z < 0.0f) color.z = 0.0f;

					// rgb, reverse order of xs
					scn->pixels[3*scn->w*j + 3*i + 2] = (unsigned char)(color.x * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 1] = (unsigned char)(color.y * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 0] = (unsigned char)(color.z * 255.0f);

				}
			}

			#pragma omp critical
			{
				#pragma omp flush (m, n)
				m += II*JJ; // approximate
				if (m/(float)(scn->h*scn->w) > n) {
					printf("rendering %.0f%% complete\n", 100.0f*n);
					n += scn->inc;
				}
				#pragma omp flush (m, n)
			}
		}

	printf("rendering complete\n");
}

void create_photon_image(scene* scn) {

	printf("Rendering\n");

	vec3 forward = glm::normalize(scn->center-scn->eye);
	vec3 right = glm::normalize(glm::cross(forward, scn->up));
	vec3 down_vec = glm::normalize(glm::cross(forward, right));

	int m = 0;
	float n = scn->inc;

	float J = (float)scn->h;
	float I = (float)scn->w;

	float SY = 2.0f*tanf(scn->fovy*pi/360.0f);
	float SX = scn->w/J * 2.0f*tanf(scn->fovy*pi/360.0f);

	const unsigned int JJ = 8;
	const unsigned int II = 8;

	#pragma omp parallel for schedule(dynamic, 1) num_threads(scn->maxthreads)
	for (int jj = 0; jj < scn->h; jj+=JJ)
		for (unsigned int ii = 0; ii < scn->w; ii+=II) {
			for (unsigned int j = jj; j < jj+JJ && j < scn->h; ++j) {

				// upsied down
				float sy = SY * (J*0.5f - (float)j - 0.5f)/J ;

				for (unsigned int i = ii; i < ii+II && i < scn->w; ++i) {

					// take into account field of view and aspect ratio/differing width/height
					float sx = SX * ((float)i + 0.5f - I*0.5f)/I ;


					vec3 dir = forward + sx*right + sy*down_vec;

					vec3 color = scn->background;

					if (scn->SSAA == 1) {

						hit ha = intersect(scn, scn->eye, glm::normalize(dir), 0, 1.0f, 0.0f);

						color = photon_shade(scn, ha);

					} else if (scn->SSAA == 2) {

						vec3 dn = SY * 0.25f/J * down_vec;
						vec3 r = SX * 0.25f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + r + dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir - r - dn), 0, 1.0f, 0.0f);

						vec3 colora = photon_shade(scn, ha);
						vec3 colorb = photon_shade(scn, hb);

						color = (colora + colorb)/2.0f;

					} else if (scn->SSAA == 4) {

						vec3 dn = SY * 0.25f/J * down_vec;
						vec3 r = SX * 0.25f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + r + dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir - r + dn), 0, 1.0f, 0.0f);
						hit hc = intersect(scn, scn->eye, glm::normalize(dir + r - dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir - r - dn), 0, 1.0f, 0.0f);

						vec3 colora = photon_shade(scn, ha);
						vec3 colorb = photon_shade(scn, hb);
						vec3 colorc = photon_shade(scn, hc);
						vec3 colord = photon_shade(scn, hd);

						color = (colora + colorb + colorc + colord)/4.0f;

					} else if (scn->SSAA == 8) {

						vec3 dn = SY * 0.125f/J * down_vec;
						vec3 r = SX * 0.125f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hc = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);

						hit he = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hf = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hg = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hh = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);

						vec3 colora = photon_shade(scn, ha);
						vec3 colorb = photon_shade(scn, hb);
						vec3 colorc = photon_shade(scn, hc);
						vec3 colord = photon_shade(scn, hd);
						vec3 colore = photon_shade(scn, he);
						vec3 colorf = photon_shade(scn, hf);
						vec3 colorg = photon_shade(scn, hg);
						vec3 colorh = photon_shade(scn, hh);

						color = (colora + colorb + colorc + colord + colore + colorf + colorg + colorh)/8.0f;

					} else if (scn->SSAA == 16) {

						vec3 dn = SY * 0.125f/J * down_vec;
						vec3 r = SX * 0.125f/I * right;

						hit ha = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hb = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);
						hit hc = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hd = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit he = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hf = intersect(scn, scn->eye, glm::normalize(dir + 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit hg = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hh = intersect(scn, scn->eye, glm::normalize(dir + 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);

						hit hi = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hj = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);
						hit hk = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 3.0f*dn), 0, 1.0f, 0.0f);
						hit hl = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r + 1.0f*dn), 0, 1.0f, 0.0f);

						hit hm = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hn = intersect(scn, scn->eye, glm::normalize(dir - 3.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);
						hit ho = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 3.0f*dn), 0, 1.0f, 0.0f);
						hit hp = intersect(scn, scn->eye, glm::normalize(dir - 1.0f*r - 1.0f*dn), 0, 1.0f, 0.0f);

						vec3 colora = photon_shade(scn, ha);
						vec3 colorb = photon_shade(scn, hb);
						vec3 colorc = photon_shade(scn, hc);
						vec3 colord = photon_shade(scn, hd);
						vec3 colore = photon_shade(scn, he);
						vec3 colorf = photon_shade(scn, hf);
						vec3 colorg = photon_shade(scn, hg);
						vec3 colorh = photon_shade(scn, hh);
						vec3 colori = photon_shade(scn, hi);
						vec3 colorj = photon_shade(scn, hj);
						vec3 colork = photon_shade(scn, hk);
						vec3 colorl = photon_shade(scn, hl);
						vec3 colorm = photon_shade(scn, hm);
						vec3 colorn = photon_shade(scn, hn);
						vec3 coloro = photon_shade(scn, ho);
						vec3 colorp = photon_shade(scn, hp);

						color = (colora + colorb + colorc + colord + colore + colorf + colorg + colorh + colori + colorj + colork + colorl + colorm + colorn + coloro + colorp)/16.0f;
					}


					// correct for too high or low color values, which will overflow/underflow on typing
					if (color.x > 1.0f) color.x = 1.0f;
					else if (color.x < 0.0f) color.x = 0.0f;
					if (color.y > 1.0f) color.y = 1.0f;
					else if (color.y < 0.0f) color.y = 0.0f;
					if (color.z > 1.0f) color.z = 1.0f;
					else if (color.z < 0.0f) color.z = 0.0f;

					// rgb, reverse order of xs
					scn->pixels[3*scn->w*j + 3*i + 2] = (unsigned char)(color.x * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 1] = (unsigned char)(color.y * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 0] = (unsigned char)(color.z * 255.0f);

				}
			}

			#pragma omp critical
			{
				#pragma omp flush (m, n)
				m += II*JJ; // approximate
				if (m/(float)(scn->h*scn->w) > n) {
					printf("rendering %.0f%% complete\n", 100.0f*n);
					n += scn->inc;
				}
				#pragma omp flush (m, n)
			}
		}

	printf("rendering complete\n");
}


void sse_create_image(sse_scene* scn) {

	printf("Rendering\n");

	__m128 forward = __m128_NORM(_mm_sub_ps(scn->center, scn->eye));
	__m128 right = __m128_CROSS(forward, scn->up);
	right = __m128_NORM(right);
	__m128 down_vec = __m128_CROSS(forward, right);
	down_vec = __m128_NORM(down_vec);

	int m = 0;
	float n = scn->inc;

	__m128 J = _mm_set1_ps((float)scn->h);
	__m128 I = _mm_set1_ps((float)scn->w);

	__m128 Jo2 = __m128_MUL_float_set(J, 0.5f);
	__m128 Io2 = __m128_MUL_float_set(I, 0.5f);

	__m128 invJ = _mm_div_ps(_mm_set1_ps(1.0f), J);
	__m128 invI = _mm_div_ps(_mm_set1_ps(1.0f), I);

	__m128 SY = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f));
	__m128 SX = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f));

	__m128 SY2 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->h);
	__m128 SX2 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->w);

	__m128 SY4 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->h);
	__m128 SX4 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->w);

	const unsigned int JJ = 8;
	const unsigned int II = 8;

	#pragma omp parallel for schedule(dynamic, 1) num_threads(scn->maxthreads)
	for (int jj = 0; jj < scn->h; jj+=JJ)
		for (unsigned int ii = 0; ii < scn->w; ii+=II) {

			for (unsigned int j = jj; j < jj+JJ && j < scn->h; ++j) {

				// upsied down
				__m128 sy = _mm_mul_ps(SY, _mm_mul_ps(_mm_sub_ps(Jo2, _mm_set1_ps((float)j + 0.5f)), invJ)) ;

				for (unsigned int i = ii; i < ii+II && i < scn->w; ++i) {

					// take into account field of view and aspect ratio/differing width/height
					__m128 sx = _mm_mul_ps(SX, _mm_mul_ps(_mm_sub_ps(_mm_set1_ps((float)i + 0.5f), Io2), invI)) ;

					__m128 dir = _mm_add_ps(forward, _mm_add_ps(_mm_mul_ps(sx, right), _mm_mul_ps(sy, down_vec)));

					__m128 color = scn->background;

					if (scn->SSAA == 1) {

						ssehit ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

						color = sse_shade(scn, ha);

					} else if (scn->SSAA == 2) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						if ( !scn->MSAA || (ha.obj != hb.obj)) {

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade(scn, ha);
						}

					} else if (scn->SSAA == 4) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj) ) {

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);
							__m128 colorc = sse_shade(scn, hc);
							__m128 colord = sse_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));

						} else if (scn->MSAA == 2){

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade(scn, ha);
						}

					} else if (scn->SSAA == 8) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hg = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj || 
											he.obj != hf.obj || hg.obj != hh.obj || he.obj != hg.obj || 
											ha.obj != he.obj) ) {

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);
							__m128 colorc = sse_shade(scn, hc);
							__m128 colord = sse_shade(scn, hd);
							__m128 colore = sse_shade(scn, he);
							__m128 colorf = sse_shade(scn, hf);
							__m128 colorg = sse_shade(scn, hg);
							__m128 colorh = sse_shade(scn, hh);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);
							__m128 colorc = sse_shade(scn, hc);
							__m128 colord = sse_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade(scn, ha);
						}

					} else if (scn->SSAA == 16) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hg = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hi = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hj = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hk = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hl = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hm = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hn = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit ho = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hp = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || he.obj != hf.obj || hg.obj != hh.obj || ha.obj != hc.obj || he.obj != hg.obj || ha.obj != he.obj || 
											hi.obj != hj.obj || hk.obj != hl.obj || hm.obj != hn.obj || ho.obj != hp.obj || hi.obj != hk.obj || hm.obj != ho.obj || hi.obj != hm.obj ||
											ha.obj != hi.obj) ) {

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);
							__m128 colorc = sse_shade(scn, hc);
							__m128 colord = sse_shade(scn, hd);
							__m128 colore = sse_shade(scn, he);
							__m128 colorf = sse_shade(scn, hf);
							__m128 colorg = sse_shade(scn, hg);
							__m128 colorh = sse_shade(scn, hh);
							__m128 colori = sse_shade(scn, hi);
							__m128 colorj = sse_shade(scn, hj);
							__m128 colork = sse_shade(scn, hk);
							__m128 colorl = sse_shade(scn, hl);
							__m128 colorm = sse_shade(scn, hm);
							__m128 colorn = sse_shade(scn, hn);
							__m128 coloro = sse_shade(scn, ho);
							__m128 colorp = sse_shade(scn, hp);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
																	 _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
														  _mm_add_ps(_mm_add_ps(_mm_add_ps(colori, colorj), _mm_add_ps(colork, colorl)),
																	 _mm_add_ps(_mm_add_ps(colorm, colorn), _mm_add_ps(coloro, colorp)))), 
											   _mm_set1_ps(0.0625f));

						} else if (scn->MSAA == 8) {

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hc);
							__m128 colorc = sse_shade(scn, hf);
							__m128 colord = sse_shade(scn, hg);
							__m128 colore = sse_shade(scn, hk);
							__m128 colorf = sse_shade(scn, hj);
							__m128 colorg = sse_shade(scn, hp);
							__m128 colorh = sse_shade(scn, hm);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);
							__m128 colorc = sse_shade(scn, hc);
							__m128 colord = sse_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade(scn, ha);
							__m128 colorb = sse_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade(scn, ha);
						}
					}

					// correct for too high or low color values, which will overflow/underflow on typing
					color = _mm_min_ps(color, _mm_set1_ps(1.0f));
					color = _mm_max_ps(color, _mm_setzero_ps());

					float out_color[4];
					_mm_storeu_ps(out_color, color);

					// rgb, reverse order of xs
					scn->pixels[3*scn->w*j + 3*i + 2] = (unsigned char)(out_color[3] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 1] = (unsigned char)(out_color[2] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 0] = (unsigned char)(out_color[1] * 255.0f);

				}
			}
			
			#pragma omp critical
			{
				#pragma omp flush (m, n)
				m += II*JJ; // approximate
				if (m/(float)(scn->h*scn->w) > n) {
					printf("rendering %.0f%% complete\n", 100.0f*n);
					n += scn->inc;
				}
				#pragma omp flush (m, n)
			}
		}

	printf("rendering complete\n");
}

void sse_create_photon_image(sse_scene* scn) {

	printf("Rendering\n");

	__m128 forward = __m128_NORM(_mm_sub_ps(scn->center, scn->eye));
	__m128 right = __m128_CROSS(forward, scn->up);
	right = __m128_NORM(right);
	__m128 down_vec = __m128_CROSS(forward, right);
	down_vec = __m128_NORM(down_vec);

	int m = 0;
	float n = scn->inc;

	__m128 J = _mm_set1_ps((float)scn->h);
	__m128 I = _mm_set1_ps((float)scn->w);

	__m128 Jo2 = __m128_MUL_float_set(J, 0.5f);
	__m128 Io2 = __m128_MUL_float_set(I, 0.5f);

	__m128 invJ = _mm_div_ps(_mm_set1_ps(1.0f), J);
	__m128 invI = _mm_div_ps(_mm_set1_ps(1.0f), I);

	__m128 SY = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f));
	__m128 SX = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f));

	__m128 SY2 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->h);
	__m128 SX2 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->w);

	__m128 SY4 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->h);
	__m128 SX4 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->w);

	const unsigned int JJ = 8;
	const unsigned int II = 8;

	#pragma omp parallel for schedule(dynamic, 1) num_threads(scn->maxthreads)
	for (int jj = 0; jj < scn->h; jj+=JJ)
		for (unsigned int ii = 0; ii < scn->w; ii+=II) {

			for (unsigned int j = jj; j < jj+JJ && j < scn->h; ++j) {

				// upsied down
				__m128 sy = _mm_mul_ps(SY, _mm_mul_ps(_mm_sub_ps(Jo2, _mm_set1_ps((float)j + 0.5f)), invJ)) ;

				for (unsigned int i = ii; i < ii+II && i < scn->w; ++i) {

					// take into account field of view and aspect ratio/differing width/height
					__m128 sx = _mm_mul_ps(SX, _mm_mul_ps(_mm_sub_ps(_mm_set1_ps((float)i + 0.5f), Io2), invI)) ;

					__m128 dir = _mm_add_ps(forward, _mm_add_ps(_mm_mul_ps(sx, right), _mm_mul_ps(sy, down_vec)));

					__m128 color = scn->background;

					if (scn->SSAA == 1) {

						ssehit ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

						color = sse_photon_shade(scn, ha);

					} else if (scn->SSAA == 2) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						if ( !scn->MSAA || (ha.obj != hb.obj)) {

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_photon_shade(scn, ha);
						}

					} else if (scn->SSAA == 4) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj) ) {

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);
							__m128 colorc = sse_photon_shade(scn, hc);
							__m128 colord = sse_photon_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));

						} else if (scn->MSAA == 2){

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_photon_shade(scn, ha);
						}

					} else if (scn->SSAA == 8) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hg = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj || 
											he.obj != hf.obj || hg.obj != hh.obj || he.obj != hg.obj || 
											ha.obj != he.obj) ) {

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);
							__m128 colorc = sse_photon_shade(scn, hc);
							__m128 colord = sse_photon_shade(scn, hd);
							__m128 colore = sse_photon_shade(scn, he);
							__m128 colorf = sse_photon_shade(scn, hf);
							__m128 colorg = sse_photon_shade(scn, hg);
							__m128 colorh = sse_photon_shade(scn, hh);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);
							__m128 colorc = sse_photon_shade(scn, hc);
							__m128 colord = sse_photon_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_photon_shade(scn, ha);
						}

					} else if (scn->SSAA == 16) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hg = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hi = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hj = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hk = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hl = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hm = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hn = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit ho = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hp = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || he.obj != hf.obj || hg.obj != hh.obj || ha.obj != hc.obj || he.obj != hg.obj || ha.obj != he.obj || 
											hi.obj != hj.obj || hk.obj != hl.obj || hm.obj != hn.obj || ho.obj != hp.obj || hi.obj != hk.obj || hm.obj != ho.obj || hi.obj != hm.obj ||
											ha.obj != hi.obj) ) {

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);
							__m128 colorc = sse_photon_shade(scn, hc);
							__m128 colord = sse_photon_shade(scn, hd);
							__m128 colore = sse_photon_shade(scn, he);
							__m128 colorf = sse_photon_shade(scn, hf);
							__m128 colorg = sse_photon_shade(scn, hg);
							__m128 colorh = sse_photon_shade(scn, hh);
							__m128 colori = sse_photon_shade(scn, hi);
							__m128 colorj = sse_photon_shade(scn, hj);
							__m128 colork = sse_photon_shade(scn, hk);
							__m128 colorl = sse_photon_shade(scn, hl);
							__m128 colorm = sse_photon_shade(scn, hm);
							__m128 colorn = sse_photon_shade(scn, hn);
							__m128 coloro = sse_photon_shade(scn, ho);
							__m128 colorp = sse_photon_shade(scn, hp);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
																	 _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
														  _mm_add_ps(_mm_add_ps(_mm_add_ps(colori, colorj), _mm_add_ps(colork, colorl)),
																	 _mm_add_ps(_mm_add_ps(colorm, colorn), _mm_add_ps(coloro, colorp)))), 
											   _mm_set1_ps(0.0625f));

						} else if (scn->MSAA == 8) {

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hc);
							__m128 colorc = sse_photon_shade(scn, hf);
							__m128 colord = sse_photon_shade(scn, hg);
							__m128 colore = sse_photon_shade(scn, hk);
							__m128 colorf = sse_photon_shade(scn, hj);
							__m128 colorg = sse_photon_shade(scn, hp);
							__m128 colorh = sse_photon_shade(scn, hm);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);
							__m128 colorc = sse_photon_shade(scn, hc);
							__m128 colord = sse_photon_shade(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_photon_shade(scn, ha);
							__m128 colorb = sse_photon_shade(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_photon_shade(scn, ha);
						}
					}


					// correct for too high or low color values, which will overflow/underflow on typing
					color = _mm_min_ps(color, _mm_set1_ps(1.0f));
					color = _mm_max_ps(color, _mm_setzero_ps());

					float out_color[4];
					_mm_storeu_ps(out_color, color);

					// rgb, reverse order of xs
					scn->pixels[3*scn->w*j + 3*i + 2] = (unsigned char)(out_color[3] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 1] = (unsigned char)(out_color[2] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 0] = (unsigned char)(out_color[1] * 255.0f);

				}
			}
			
			#pragma omp critical
			{
				#pragma omp flush (m, n)
				m += II*JJ; // approximate
				if (m/(float)(scn->h*scn->w) > n) {
					printf("rendering %.0f%% complete\n", 100.0f*n);
					n += scn->inc;
				}
				#pragma omp flush (m, n)
			}
		}

	printf("rendering complete\n");
}


void sse_kd_create_image(sse_scene* scn) {

	printf("Rendering\n");

	__m128 forward = __m128_NORM(_mm_sub_ps(scn->center, scn->eye));
	__m128 right = __m128_CROSS(forward, scn->up);
	right = __m128_NORM(right);
	__m128 down_vec = __m128_CROSS(forward, right);
	down_vec = __m128_NORM(down_vec);


	__m128 J = _mm_set1_ps((float)scn->h);
	__m128 I = _mm_set1_ps((float)scn->w);

	__m128 Jo2 = __m128_MUL_float_set(J, 0.5f);
	__m128 Io2 = __m128_MUL_float_set(I, 0.5f);

	__m128 invJ = _mm_div_ps(_mm_set1_ps(1.0f), J);
	__m128 invI = _mm_div_ps(_mm_set1_ps(1.0f), I);

	__m128 SY = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f));
	__m128 SX = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f));

	__m128 SY2 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->h);
	__m128 SX2 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.25f/(float)scn->w);

	__m128 SY4 = _mm_set1_ps(2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->h);
	__m128 SX4 = _mm_set1_ps(scn->w/(float)scn->h * 2.0f*tanf(scn->fovy*pi/360.0f) * 0.125f/(float)scn->w);

	int m = 0;
	float n = scn->inc;

	const unsigned int JJ = 8;
	const unsigned int II = 8;

	#pragma omp parallel for schedule(dynamic, 1) num_threads(scn->maxthreads)
	for (int jj = 0; jj < scn->h; jj+=JJ)
		for (unsigned int ii = 0; ii < scn->w; ii+=II) {

			for (unsigned int j = jj; j < jj+JJ && j < scn->h; ++j) {

				// upsied down
				__m128 sy = _mm_mul_ps(SY, _mm_mul_ps(_mm_sub_ps(Jo2, _mm_set1_ps((float)j + 0.5f)), invJ)) ;

				for (unsigned int i = ii; i < ii+II && i < scn->w; ++i) {

					// take into account field of view and aspect ratio/differing width/height
					__m128 sx = _mm_mul_ps(SX, _mm_mul_ps(_mm_sub_ps(_mm_set1_ps((float)i + 0.5f), Io2), invI)) ;

					__m128 dir = _mm_add_ps(forward, _mm_add_ps(_mm_mul_ps(sx, right), _mm_mul_ps(sy, down_vec)));

					__m128 color = scn->background;

					if (scn->SSAA == 1) {

						ssehit ha = sse_intersect_kd(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

						color = sse_shade_kd(scn, ha);

					} else if (scn->SSAA == 2) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						if ( !scn->MSAA || (ha.obj != hb.obj)) {

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect_kd(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade_kd(scn, ha);
						}

					} else if (scn->SSAA == 4) {

						__m128 dn = _mm_mul_ps(SY2, down_vec);
						__m128 r = _mm_mul_ps(SX2, right);

						ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj) ) {

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);
							__m128 colorc = sse_shade_kd(scn, hc);
							__m128 colord = sse_shade_kd(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));

						} else if (scn->MSAA == 2){

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));

						} else {

							ha = sse_intersect_kd(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade_kd(scn, ha);
						}

					} else if (scn->SSAA == 8) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hc = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hg = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || ha.obj != hc.obj || 
											he.obj != hf.obj || hg.obj != hh.obj || he.obj != hg.obj || 
											ha.obj != he.obj) ) {

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);
							__m128 colorc = sse_shade_kd(scn, hc);
							__m128 colord = sse_shade_kd(scn, hd);
							__m128 colore = sse_shade_kd(scn, he);
							__m128 colorf = sse_shade_kd(scn, hf);
							__m128 colorg = sse_shade_kd(scn, hg);
							__m128 colorh = sse_shade_kd(scn, hh);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);
							__m128 colorc = sse_shade_kd(scn, hc);
							__m128 colord = sse_shade_kd(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect_kd(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade_kd(scn, ha);
						}

					} else if (scn->SSAA == 16) {

						__m128 dn = _mm_mul_ps(SY4, down_vec);
						__m128 r = _mm_mul_ps(SX4, right);
						__m128 dn3 = _mm_mul_ps(dn, _mm_set1_ps(3.0f));
						__m128 r3 = _mm_mul_ps(r, _mm_set1_ps(3.0f));

						ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hc = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hd = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit he = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hf = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hg = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hh = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hi = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hj = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit hk = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hl = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						ssehit hm = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hn = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r3), dn)), 0, 0, 1.0f, 0.0f);
						ssehit ho = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn3)), 0, 0, 1.0f, 0.0f);
						ssehit hp = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

						

						if ( !scn->MSAA || (ha.obj != hb.obj || hc.obj != hd.obj || he.obj != hf.obj || hg.obj != hh.obj || ha.obj != hc.obj || he.obj != hg.obj || ha.obj != he.obj || 
											hi.obj != hj.obj || hk.obj != hl.obj || hm.obj != hn.obj || ho.obj != hp.obj || hi.obj != hk.obj || hm.obj != ho.obj || hi.obj != hm.obj ||
											ha.obj != hi.obj) ) {

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);
							__m128 colorc = sse_shade_kd(scn, hc);
							__m128 colord = sse_shade_kd(scn, hd);
							__m128 colore = sse_shade_kd(scn, he);
							__m128 colorf = sse_shade_kd(scn, hf);
							__m128 colorg = sse_shade_kd(scn, hg);
							__m128 colorh = sse_shade_kd(scn, hh);
							__m128 colori = sse_shade_kd(scn, hi);
							__m128 colorj = sse_shade_kd(scn, hj);
							__m128 colork = sse_shade_kd(scn, hk);
							__m128 colorl = sse_shade_kd(scn, hl);
							__m128 colorm = sse_shade_kd(scn, hm);
							__m128 colorn = sse_shade_kd(scn, hn);
							__m128 coloro = sse_shade_kd(scn, ho);
							__m128 colorp = sse_shade_kd(scn, hp);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
																	 _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
														  _mm_add_ps(_mm_add_ps(_mm_add_ps(colori, colorj), _mm_add_ps(colork, colorl)),
																	 _mm_add_ps(_mm_add_ps(colorm, colorn), _mm_add_ps(coloro, colorp)))), 
											   _mm_set1_ps(0.0625f));

						} else if (scn->MSAA == 8) {

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hc);
							__m128 colorc = sse_shade_kd(scn, hf);
							__m128 colord = sse_shade_kd(scn, hg);
							__m128 colore = sse_shade_kd(scn, hk);
							__m128 colorf = sse_shade_kd(scn, hj);
							__m128 colorg = sse_shade_kd(scn, hp);
							__m128 colorh = sse_shade_kd(scn, hm);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(colora, colorb), _mm_add_ps(colorc, colord)),
														  _mm_add_ps(_mm_add_ps(colore, colorf), _mm_add_ps(colorg, colorh))), 
											   _mm_set1_ps(0.125f));

						} else if (scn->MSAA == 4) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hc = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hd = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);
							__m128 colorc = sse_shade_kd(scn, hc);
							__m128 colord = sse_shade_kd(scn, hd);

							color = _mm_mul_ps(_mm_add_ps(_mm_add_ps(colora, colorb), 
														  _mm_add_ps(colorc, colord)), 
											   _mm_set1_ps(0.25f));
						
						} else if (scn->MSAA == 2) {

							__m128 dn = _mm_mul_ps(SY2, down_vec);
							__m128 r = _mm_mul_ps(SX2, right);

							ssehit ha = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_add_ps(_mm_add_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);
							ssehit hb = sse_intersect_kd(scn, scn->eye,  __m128_NORM3(_mm_sub_ps(_mm_sub_ps(dir, r), dn)), 0, 0, 1.0f, 0.0f);

							__m128 colora = sse_shade_kd(scn, ha);
							__m128 colorb = sse_shade_kd(scn, hb);

							color = _mm_mul_ps(_mm_add_ps(colora, colorb), _mm_set1_ps(0.5f));
						
						} else {

							ha = sse_intersect_kd(scn, scn->eye, __m128_NORM3(dir), 0, 0, 1.0f, 0.0f);

							color = sse_shade_kd(scn, ha);
						}
					}

					// correct for too high or low color values, which will overflow/underflow on typing
					color = _mm_min_ps(color, _mm_set1_ps(1.0f));
					color = _mm_max_ps(color, _mm_setzero_ps());

					float out_color[4];
					_mm_storeu_ps(out_color, color);

					// rgb, reverse order of xs
					scn->pixels[3*scn->w*j + 3*i + 2] = (unsigned char)(out_color[3] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 1] = (unsigned char)(out_color[2] * 255.0f);
					scn->pixels[3*scn->w*j + 3*i + 0] = (unsigned char)(out_color[1] * 255.0f);

				}
			}
			
			#pragma omp critical
			{
				#pragma omp flush (m, n)
				m += II*JJ; // approximate
				if ((float)m > n*(float)(scn->h*scn->w)) {
					printf("rendering %.0f%% complete\n", 100.0f*n);
					n += scn->inc;
				}
				#pragma omp flush (m, n)
			}
		}

	printf("rendering complete\n");
}

int main(int argc, char** argv) {

	if (argc < 2) {
		cerr << "No scene file specified\n"; 
		exit(-1); 
	}

	//printf("%f\n", INF);
	//printf("%d\n", sizeof(kd_tree_node));

	FreeImage_Initialise();

	//scene scn = make_scene(); 
	sse_scene scn = make_sse_scene(); 

	double rft = omp_get_wtime() ;
	//readfile(argv[1], &scn) ; 
	sse_readfile(argv[1], &scn) ; // local_radius or photonradius must be set before any triangles or parallelograms are created
	rft = omp_get_wtime() - rft ;

	
	double spt = omp_get_wtime() ;
	//shoot_photons(&scn);
	sse_shoot_photons(&scn);
	spt = omp_get_wtime() - spt ;


	double ckdt = omp_get_wtime() ;
	//create_kd_tree(&scn) ; 
	ckdt = omp_get_wtime() - ckdt ;
	

	init_pixels(&scn);


	double rt = omp_get_wtime() ;
	//create_photon_image(&scn) ;
	//sse_create_image(&scn) ;
	//sse_kd_create_image(&scn) ;
	sse_create_photon_image(&scn) ;
	rt = omp_get_wtime() - rt ;


	stringstream s;
	s << " (readfile - " << rft << "s, shoot photons - " << spt << "s (" << scn.num_photons << "x" << scn.num_of_diffuse << "-" << scn.loc_radius << "x" << TILE << "), kd-tree - " << ckdt << "s (" << scn.max_kd_depth << "-" << scn.min_kd_leaf_size << "), render - " << rt << "s (" << scn.w << "x" << scn.h << "-" << (scn.MSAA ? "MSAA " : "SSAA ") << scn.SSAA << "-" << scn.MSAA << "x)";
	//s << " (Render - " << rt << "s)";

	saveScreenshot(scn.output_filename + s.str(), scn.pixels, scn.w, scn.h);


	//destroy_scene(&scn);
	destroy_sse_scene(&scn);


	FreeImage_DeInitialise();
	return 0;
}
