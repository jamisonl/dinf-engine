/* Public Apple math.h API, available since macOS 10.9. Keep its implementation
 * struct-return ABI behind this header-defined function and a plain pointer ABI.
 */
#include <math.h>

void dinf_rope_sincosf(float angle, float *sine, float *cosine) {
    __sincosf(angle, sine, cosine);
}
