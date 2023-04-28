#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 fxvec = _mm256_load_ps(fx);
    __m256 fyvec = _mm256_load_ps(fy);
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 mvec = _mm256_load_ps(m);

    __m256 dxvec = _mm256_sub_ps(xivec, xvec);
    __m256 dx2vec = _mm256_mul_ps(dxvec, dxvec);
    __m256 dyvec = _mm256_sub_ps(yivec, yvec);
    __m256 dy2vec = _mm256_mul_ps(dyvec, dyvec);
    __m256 r2vec = _mm256_add_ps(dx2vec, dy2vec);
    __m256 rinvvec = _mm256_rsqrt_ps(r2vec);
    __m256 mask = _mm256_cmp_ps(rinvvec, _mm256_set1_ps(INFINITY), _CMP_NEQ_OQ);
    rinvvec = _mm256_blendv_ps(_mm256_setzero_ps(), rinvvec, mask);
    __m256 rinv3vec = _mm256_mul_ps(rinvvec, rinvvec);
    rinv3vec = _mm256_mul_ps(rinv3vec, rinvvec);

    __m256 subfxvec = _mm256_mul_ps(dxvec, mvec);
    subfxvec = _mm256_mul_ps(subfxvec, rinv3vec);
    __m256 subfyvec = _mm256_mul_ps(dyvec, mvec);
    subfyvec = _mm256_mul_ps(subfyvec, rinv3vec);

    __m256 redvec = _mm256_permute2f128_ps(subfxvec, subfxvec, 1);
    subfxvec = _mm256_add_ps(subfxvec, redvec);
    subfxvec = _mm256_hadd_ps(subfxvec, subfxvec);
    subfxvec = _mm256_hadd_ps(subfxvec, subfxvec);

    redvec = _mm256_permute2f128_ps(subfyvec, subfyvec, 1);
    subfyvec = _mm256_add_ps(subfyvec, redvec);
    subfyvec = _mm256_hadd_ps(subfyvec, subfyvec);
    subfyvec = _mm256_hadd_ps(subfyvec, subfyvec);

    float subfx[N], subfy[N];

    _mm256_store_ps(subfx, subfxvec);
    _mm256_store_ps(subfy, subfyvec);

    fx[i] -= subfx[0];
    fy[i] -= subfy[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

