/*
 * Argon2 POWER8/VSX optimized implementation
 * Ported by Scott (Scottcjn) - 2025
 *
 * VSX/AltiVec SIMD implementation for IBM POWER8 and later
 * Uses vec_perm for shuffle operations - perfect 1:1 mapping from SSE2
 */

#ifndef BLAKE_ROUND_MKA_VSX_H
#define BLAKE_ROUND_MKA_VSX_H

#include "blake2-impl.h"

#if defined(__VSX__) || defined(__ALTIVEC__)

#include <altivec.h>

/* Type definitions */
typedef vector unsigned long long v2du;  /* 2x64-bit unsigned */
typedef vector unsigned int v4su;        /* 4x32-bit unsigned */
typedef vector unsigned char v16qu;      /* 16x8-bit unsigned */

/* Permutation patterns for rotations */
static const v16qu ROT32_PERM = {4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11};
static const v16qu ROT24_PERM = {3,4,5,6,7,0,1,2, 11,12,13,14,15,8,9,10};
static const v16qu ROT16_PERM = {2,3,4,5,6,7,0,1, 10,11,12,13,14,15,8,9};

/* Permutation patterns for DIAGONALIZE/UNDIAGONALIZE */
/* alignr_epi8(B1, B0, 8) = take high 8 bytes of B0, low 8 bytes of B1 */
static const v16qu ALIGNR8_01 = {8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23};
static const v16qu ALIGNR8_10 = {8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23};

/*
 * fBlaMka: The core Argon2 mixing function
 * result = x + y + 2 * trunc(x) * trunc(y)
 * where trunc() takes lower 32 bits of each 64-bit element
 */
static BLAKE2_INLINE v2du fBlaMka_vsx(v2du x, v2du y) {
    /* Multiply lower 32 bits: need vec_mule for even lanes */
    v4su x32 = (v4su)x;
    v4su y32 = (v4su)y;

    /* vec_mule multiplies elements 0 and 2, giving 64-bit results */
    v2du z = vec_mule(x32, y32);

    /* result = x + y + 2*z */
    return vec_add(vec_add(x, y), vec_add(z, z));
}

/* Rotation macros using vec_perm */
#define VSX_ROTI_EPI64(x, c) \
    ((-(c) == 32) ? (v2du)vec_perm((v16qu)(x), (v16qu)(x), ROT32_PERM) : \
     (-(c) == 24) ? (v2du)vec_perm((v16qu)(x), (v16qu)(x), ROT24_PERM) : \
     (-(c) == 16) ? (v2du)vec_perm((v16qu)(x), (v16qu)(x), ROT16_PERM) : \
     (-(c) == 63) ? vec_xor(vec_sr((x), (v2du){63,63}), vec_add((x), (x))) : \
                    vec_xor(vec_sr((x), (v2du){-(c), -(c)}), \
                            vec_sl((x), (v2du){64-(-(c)), 64-(-(c))})))

/* G1 mixing function - first half of quarter round */
#define G1_VSX(A0, B0, C0, D0, A1, B1, C1, D1)                              \
    do {                                                                     \
        A0 = fBlaMka_vsx(A0, B0);                                           \
        A1 = fBlaMka_vsx(A1, B1);                                           \
                                                                             \
        D0 = vec_xor(D0, A0);                                               \
        D1 = vec_xor(D1, A1);                                               \
                                                                             \
        D0 = VSX_ROTI_EPI64(D0, -32);                                       \
        D1 = VSX_ROTI_EPI64(D1, -32);                                       \
                                                                             \
        C0 = fBlaMka_vsx(C0, D0);                                           \
        C1 = fBlaMka_vsx(C1, D1);                                           \
                                                                             \
        B0 = vec_xor(B0, C0);                                               \
        B1 = vec_xor(B1, C1);                                               \
                                                                             \
        B0 = VSX_ROTI_EPI64(B0, -24);                                       \
        B1 = VSX_ROTI_EPI64(B1, -24);                                       \
    } while ((void)0, 0)

/* G2 mixing function - second half of quarter round */
#define G2_VSX(A0, B0, C0, D0, A1, B1, C1, D1)                              \
    do {                                                                     \
        A0 = fBlaMka_vsx(A0, B0);                                           \
        A1 = fBlaMka_vsx(A1, B1);                                           \
                                                                             \
        D0 = vec_xor(D0, A0);                                               \
        D1 = vec_xor(D1, A1);                                               \
                                                                             \
        D0 = VSX_ROTI_EPI64(D0, -16);                                       \
        D1 = VSX_ROTI_EPI64(D1, -16);                                       \
                                                                             \
        C0 = fBlaMka_vsx(C0, D0);                                           \
        C1 = fBlaMka_vsx(C1, D1);                                           \
                                                                             \
        B0 = vec_xor(B0, C0);                                               \
        B1 = vec_xor(B1, C1);                                               \
                                                                             \
        B0 = VSX_ROTI_EPI64(B0, -63);                                       \
        B1 = VSX_ROTI_EPI64(B1, -63);                                       \
    } while ((void)0, 0)

/*
 * DIAGONALIZE - Permute vectors for diagonal mixing
 * Using vec_perm for alignr_epi8 equivalent
 */
#define DIAGONALIZE_VSX(A0, B0, C0, D0, A1, B1, C1, D1)                     \
    do {                                                                     \
        /* alignr_epi8(B1, B0, 8): high 8 of B0 || low 8 of B1 */          \
        v16qu t0 = vec_perm((v16qu)B0, (v16qu)B1, ALIGNR8_01);             \
        v16qu t1 = vec_perm((v16qu)B1, (v16qu)B0, ALIGNR8_01);             \
        B0 = (v2du)t0;                                                      \
        B1 = (v2du)t1;                                                      \
                                                                             \
        v2du tmp = C0;                                                      \
        C0 = C1;                                                            \
        C1 = tmp;                                                           \
                                                                             \
        t0 = vec_perm((v16qu)D0, (v16qu)D1, ALIGNR8_01);                   \
        t1 = vec_perm((v16qu)D1, (v16qu)D0, ALIGNR8_01);                   \
        D0 = (v2du)t1;                                                      \
        D1 = (v2du)t0;                                                      \
    } while ((void)0, 0)

/*
 * UNDIAGONALIZE - Reverse the diagonal permutation
 */
#define UNDIAGONALIZE_VSX(A0, B0, C0, D0, A1, B1, C1, D1)                   \
    do {                                                                     \
        v16qu t0 = vec_perm((v16qu)B1, (v16qu)B0, ALIGNR8_01);             \
        v16qu t1 = vec_perm((v16qu)B0, (v16qu)B1, ALIGNR8_01);             \
        B0 = (v2du)t0;                                                      \
        B1 = (v2du)t1;                                                      \
                                                                             \
        v2du tmp = C0;                                                      \
        C0 = C1;                                                            \
        C1 = tmp;                                                           \
                                                                             \
        t0 = vec_perm((v16qu)D1, (v16qu)D0, ALIGNR8_01);                   \
        t1 = vec_perm((v16qu)D0, (v16qu)D1, ALIGNR8_01);                   \
        D0 = (v2du)t1;                                                      \
        D1 = (v2du)t0;                                                      \
    } while ((void)0, 0)

/*
 * Full BLAKE2 round using VSX
 */
#define BLAKE2_ROUND_VSX(A0, A1, B0, B1, C0, C1, D0, D1)                    \
    do {                                                                     \
        G1_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                             \
        G2_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                             \
                                                                             \
        DIAGONALIZE_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                    \
                                                                             \
        G1_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                             \
        G2_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                             \
                                                                             \
        UNDIAGONALIZE_VSX(A0, B0, C0, D0, A1, B1, C1, D1);                  \
    } while ((void)0, 0)

/* Load/Store macros for VSX */
#define VSX_LOADU(ptr) vec_xl(0, (const unsigned long long*)(ptr))
#define VSX_STOREU(ptr, val) vec_xst((val), 0, (unsigned long long*)(ptr))

#endif /* __VSX__ || __ALTIVEC__ */

#endif /* BLAKE_ROUND_MKA_VSX_H */
