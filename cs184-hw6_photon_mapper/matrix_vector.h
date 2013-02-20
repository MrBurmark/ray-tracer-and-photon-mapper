
#ifndef MATRIX_VECTOR_H
#define MATRIX_VECTOR_H

#include <emmintrin.h>

typedef struct fmat4 {
	__m128 r1 ;
	__m128 r2 ;
	__m128 r3 ;
	__m128 r4 ;
} fmat4;

typedef struct fmat4_ex {
	fmat4 mat ;
	bool non_uniform_scale ;
} fmat4_ex;

#define __m128_SUM(a, b) _mm_store_ss(&b, _mm_hadd_ps(_mm_hadd_ps(a, a)), a)

//#define __m128_DOT(a, b, c) _mm_store_ss(&c, _mm_hadd_ps( _mm_hadd_ps(_mm_mul_ps(a, b), a), b))
//#define __m128_DOT___m128(a, b) _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, b), _mm_mul_ps(a, b)), _mm_hadd_ps(_mm_mul_ps(a, b), _mm_mul_ps(a, b)))

#define __m128_DOT(a, b, c) _mm_store_ss(&c, _mm_dp_ps(a, b, 0xff))
#define __m128_DOT___m128(a, b) _mm_dp_ps(a, b, 0xff)

#define __m128_DOT3(a, b, c) _mm_store_ss(&c, _mm_dp_ps(a, b, 0xef))
#define __m128_DOT3___m128(a, b) _mm_dp_ps(a, b, 0xef)

#define __m128_MUL_float_load(a, b) _mm_mul_ps(a, _mm_load1_ps(&b))
#define __m128_MUL_float_set(a, b) _mm_mul_ps(a, _mm_set1_ps(b))
#define float_load_MUL___m128(a, b) _mm_mul_ps(_mm_load1_ps(&a), b)
#define float_set_MUL___m128(a, b) _mm_mul_ps(_mm_set1_ps(a), b)

//#define __m128_CROSS(a, b) _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 1, 3, 0)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 3, 2, 0))), _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 3, 2, 0)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 1, 3, 0))))
#define __m128_CROSS(a, b) _mm_shuffle_ps(_mm_sub_ps(_mm_mul_ps(a, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 1, 3, 0))), _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 1, 3, 0)), b)), _mm_sub_ps(_mm_mul_ps(a, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 1, 3, 0))), _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 1, 3, 0)), b)), _MM_SHUFFLE(2, 1, 3, 0))

#define __m128_INV(a) _mm_div_ps(_mm_set1_ps(1.0f), a)
#define __m128_QINV(a) _mm_rcp_ps(a)

#define __m128_LEN(a, b) _mm_store_ss(&b, _mm_sqrt_ps(_mm_dp_ps(a, a, 0xff)))
#define __m128_LEN___m128(a) _mm_sqrt_ps(_mm_dp_ps(a, a, 0xff))

#define __m128_LEN3(a, b) _mm_store_ss(&b, _mm_sqrt_ps(_mm_dp_ps(a, a, 0xef)))
#define __m128_LEN3___m128(a) _mm_sqrt_ps(_mm_dp_ps(a, a, 0xef))

//#define __m128_NORM(a) _mm_div_ps(a, _mm_sqrt_ps(_mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, a), _mm_mul_ps(a, a)), _mm_hadd_ps(_mm_mul_ps(a, a), _mm_mul_ps(a, a)))))
//#define __m128_QNORM(a) _mm_mul_ps(a, _mm_rsqrt_ps(_mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, a), _mm_mul_ps(a, a)), _mm_hadd_ps(_mm_mul_ps(a, a), _mm_mul_ps(a, a)))))

#define __m128_NORM(a) _mm_div_ps(a, _mm_sqrt_ps(_mm_dp_ps(a, a, 0xff)))
#define __m128_QNORM(a) _mm_mul_ps(a, _mm_rsqrt_ps(_mm_dp_ps(a, a, 0xff)))

#define __m128_NORM3(a) _mm_div_ps(a, _mm_sqrt_ps(_mm_dp_ps(a, a, 0xef)))
#define __m128_QNORM3(a) _mm_mul_ps(a, _mm_rsqrt_ps(_mm_dp_ps(a, a, 0xef)))

#define __m128_NEG(a) _mm_mul_ps(_mm_set1_ps(-1.0f), a)

#define fmat4_MUL___m128(A, b) _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(A.r4, b), _mm_mul_ps(A.r3, b)), _mm_hadd_ps(_mm_mul_ps(A.r2, b), _mm_mul_ps(A.r1, b)))
#define fmat4_MUL3___m128(A, b) _mm_hadd_ps(_mm_hadd_ps(_mm_setzero_ps(), _mm_mul_ps(A.r3, b)), _mm_hadd_ps(_mm_mul_ps(A.r2, b), _mm_mul_ps(A.r1, b)))

#define __m128_MUL_fmat4(a, B) _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(a, B.r4), _mm_mul_ps(a, B.r3)), _mm_hadd_ps(_mm_mul_ps(a, B.r2), _mm_mul_ps(a, B.r1)))

#define __m128_ZERO_r0(a) _mm_insert_ps(a, a, _MM_MK_INSERTPS_NDX(0, 0, 1))

#define __m128_REVERSE(a) _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3))

#ifdef MAIN

fmat4 fmat4_transp(fmat4& A) {

	__m128 tmp1 = _mm_shuffle_ps(A.r2, A.r1, _MM_SHUFFLE(3, 2, 3, 2));
	__m128 tmp2 = _mm_shuffle_ps(A.r4, A.r3, _MM_SHUFFLE(3, 2, 3, 2));

	__m128 tmp3 = _mm_shuffle_ps(A.r2, A.r1, _MM_SHUFFLE(1, 0, 1, 0));
	__m128 tmp4 = _mm_shuffle_ps(A.r4, A.r3, _MM_SHUFFLE(1, 0, 1, 0));

	fmat4 B = {_mm_shuffle_ps(tmp2, tmp1, _MM_SHUFFLE(3, 1, 3, 1)),
			   _mm_shuffle_ps(tmp2, tmp1, _MM_SHUFFLE(2, 0, 2, 0)),
			   _mm_shuffle_ps(tmp4, tmp3, _MM_SHUFFLE(3, 1, 3, 1)),
			   _mm_shuffle_ps(tmp4, tmp3, _MM_SHUFFLE(2, 0, 2, 0))};
	
	return B;
}


fmat4 fmat4_add_fmat4(fmat4& A, fmat4& B) {

	fmat4 C = {_mm_add_ps(A.r1, B.r1),
			   _mm_add_ps(A.r2, B.r2),
			   _mm_add_ps(A.r3, B.r3),
			   _mm_add_ps(A.r4, B.r4)};
	return C;
}

fmat4 fmat4_mul_fmat4(fmat4& A, fmat4& B) {

	fmat4 C = fmat4_transp(B);

	fmat4 D = {__m128_MUL_fmat4(A.r1, C),
			   __m128_MUL_fmat4(A.r2, C),
			   __m128_MUL_fmat4(A.r3, C),
			   __m128_MUL_fmat4(A.r4, C)};
	return D;
}

fmat4 fmat4_mul_float(fmat4& A, float b) {

	fmat4 C = {_mm_mul_ps(A.r1, _mm_set1_ps(b)),
			   _mm_mul_ps(A.r2, _mm_set1_ps(b)),
			   _mm_mul_ps(A.r3, _mm_set1_ps(b)),
			   _mm_mul_ps(A.r4, _mm_set1_ps(b))};
	return C;
}

fmat4 make_fmat4(float a) {
	
	fmat4 A = {_mm_set_ps(a, 0.0f, 0.0f, 0.0f),
			   _mm_set_ps(0.0f, a, 0.0f, 0.0f),
			   _mm_set_ps(0.0f, 0.0f, a, 0.0f),
			   _mm_set_ps(0.0f, 0.0f, 0.0f, a)};
	return A;
}

fmat4 make_fmat4(float a, float b, float c, float d) {
	
	fmat4 A = {_mm_set_ps(a, 0.0f, 0.0f, 0.0f),
			   _mm_set_ps(0.0f, b, 0.0f, 0.0f),
			   _mm_set_ps(0.0f, 0.0f, c, 0.0f),
			   _mm_set_ps(0.0f, 0.0f, 0.0f, d)};
	return A;
}

fmat4 make_fmat4(float a0, float a1, float a2, float a3, 
				 float b0, float b1, float b2, float b3, 
				 float c0, float c1, float c2, float c3, 
				 float d0, float d1, float d2, float d3) {
	
	fmat4 A = {_mm_set_ps(a0, a1, a2, a3),
			   _mm_set_ps(b0, b1, b2, b3),
			   _mm_set_ps(c0, c1, c2, c3),
			   _mm_set_ps(d0, d1, d2, d3)};
	return A;
}

#else

fmat4 fmat4_transp(fmat4& A) ;

fmat4 fmat4_add_fmat4(fmat4& A, fmat4& B) ;

fmat4 fmat4_mul_fmat4(fmat4& A, fmat4& B) ;

fmat4 fmat4_mul_float(fmat4& A, float b) ;

fmat4 make_fmat4(float a) ;

fmat4 make_fmat4(float a, float b, float c, float d) ;

fmat4 make_fmat4(float a0, float a1, float a2, float a3, 
				 float b0, float b1, float b2, float b3, 
				 float c0, float c1, float c2, float c3, 
				 float d0, float d1, float d2, float d3) ;
#endif

#endif
