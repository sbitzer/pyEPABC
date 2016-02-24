#include <math.h>
#include "brownian_motion_simulation.h"

int sample_from_DDM_C(double *parsamples, int N, int P, double dt,
		double maxRT, int *seed, double *choices, double *RTs)
{
	int n;
	double s, v, sv, a, z, sz, ndt, sndt, RT;

	if (P != 8) return 1;

	for (n = 0; n < N; n++){
		/* unpack parameters */
		v    = parsamples[n*P];
		sv   = parsamples[n*P+1];
		a    = parsamples[n*P+2];
		z    = parsamples[n*P+3] * a;
		sz   = parsamples[n*P+4];
		ndt  = parsamples[n*P+5];
		sndt = parsamples[n*P+6];
		s    = parsamples[n*P+7];

		/* sample drift, non-decision time and starting point for this run */
		if (sv > 0){
			v = r8_normal_01(seed) * sv + v;
		}
		if (sndt > 0) {
			ndt = (r8_uniform_01(seed) - 0.5) * sndt + ndt;
			/* non-decision time can never be negative */
			ndt = (ndt > 0) ? ndt : 0;
		}
		if (sz > 0){
			z = (r8_uniform_01(seed) - 0.5) * sz * a + z;
		}

		/* simulate drift-diffusion process */
		RT = ndt;
		while (((0 < z) & (z < a)) & (RT < maxRT)){
			z += v * dt + sqrt(dt) * s * r8_normal_01(seed);
			RT += dt;
		}

		/* record response */
		if (RT > maxRT){
			choices[n] = 0;
			RTs[n] = maxRT + 1.0;
		} else if (z > a){
			choices[n] = 1.0;
			RTs[n] = RT;
		} else {
			choices[n] = 2.0;
			RTs[n] = RT;
		}
	}

	return 0;
}
