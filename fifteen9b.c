#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <immintrin.h> 
#include <pthread.h>

// gcc fifteen9b.c -o fifteen9b.bin -O3 -march=native -Wall -lpthread


#define CMN15_ONEDBNUM 8
#define BUILTIN_PREFETCH_ARG2_FOR_READ 0
#define BUILTIN_PREFETCH_ARG2_FOR_WRITE 1
#define BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_LOWEST 0 // Don't keep in cache for long
#define BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_LOW 1
#define BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_MEDIUM 2
#define BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_HIGH 3 // Keep in cache for long time
// e.g. __builtin_prefetch(ptr, BUILTIN_PREFETCH_ARG2_FOR_WRITE, BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_LOW);


typedef struct {
  uint64_t goalboard;
  uint64_t arraysize;
  uint8_t *array[2];
  uint8_t onedbx;
  uint8_t onedby;
} cmn15_db7710_t;

uint64_t cmn15_removezeronibble(uint64_t b) {
  uint64_t res = 0;
  uint64_t n;  
  uint8_t zeropassed = 0;
  for(uint32_t i=0; i<16; i++) {
    n = b & (0xfULL << (4*i));
    if (n == 0) zeropassed = 4;
    res += n >> zeropassed;
  }
  return res;
}

void cmn15_removezeronibbles(uint64_t a, uint64_t b, uint64_t *aout, uint64_t *bout) {
  uint64_t maska = (a & 0x5555555555555555ULL) | ((a & 0xaaaaaaaaaaaaaaaaULL) >> 1);
  uint64_t maskb = (b & 0x5555555555555555ULL) | ((b & 0xaaaaaaaaaaaaaaaaULL) >> 1);
  maska = (maska & 0x3333333333333333ULL) | ((maska & 0xccccccccccccccccULL) >> 2);
  maskb = (maskb & 0x3333333333333333ULL) | ((maskb & 0xccccccccccccccccULL) >> 2);
  maska |= (maska << 1);
  maskb |= (maskb << 1);
  maska |= (maska << 2);
  maskb |= (maskb << 2);
  int32_t ctza = __builtin_ctzll(~maska);
  int32_t ctzb = __builtin_ctzll(~maskb);
  if (ctza == 60) {
    *aout = (((1ULL << ctza) - 1) & a);
  } else {
    *aout = (((1ULL << ctza) - 1) & a) | ((~((1ULL << (4 + ctza)) - 1) & a) >> 4);
  }
  if (ctzb == 60) {
    *bout = (((1ULL << ctzb) - 1) & b);
  } else {
    *bout = (((1ULL << ctzb) - 1) & b) | ((~((1ULL << (4 + ctzb)) - 1) & b) >> 4);
  }
  return;
}
#ifndef __SSSE3__
uint32_t cmn15_linearconflicts(uint64_t boardA, uint64_t boardB) {
  uint8_t p[15] = {0};
  uint64_t temp, A, B, lc;
  cmn15_removezeronibbles(boardA, boardB, &A, &B);
  uint8_t nibbleA;
  for(uint32_t k=0; k<15; k++) {
    nibbleA = A & 0xf;
    temp = B;
    for(uint32_t j=0; j<15; j++) {
      if (nibbleA == (temp & 0xf)){
        p[k] = j;
        break;
      }
      temp >>= 4;
    }
    A >>= 4;
  }
  lc = 0;
  for(uint32_t m=0; m<15; m++) {
    for(uint32_t k=0; k<m; k++) {
      lc += (p[k] > p[m]);
    }
  }
  return lc;
}
#else
void cmn15_linearconflicts(uint64_t boardA1, uint32_t *nibbleB1pos, uint64_t boardA2, uint32_t *nibbleB2pos, uint32_t *V, uint32_t *H) {
  uint8_t p1[16] = {0};
  uint8_t p2[16] = {0};
  uint64_t A1, A2;
  cmn15_removezeronibbles(boardA1, boardA2, &A1, &A2);
  __m128i shuff = _mm_set_epi8(7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8);
  __m128i vA1 = _mm_set_epi64x(A1 & 0x0f0f0f0f0f0f0f0fULL, (A1 & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  vA1 = _mm_shuffle_epi8(vA1, shuff);
  __m128i vA2 = _mm_set_epi64x(A2 & 0x0f0f0f0f0f0f0f0fULL, (A2 & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  vA2 = _mm_shuffle_epi8(vA2, shuff);
  __m128i vk;
  uint32_t nibbleA1pos, nibbleA2pos;
  for(uint32_t k=1; k<16; k++) {
    vk = _mm_set1_epi8(k);
    nibbleA1pos = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, vA1)));
    p1[nibbleA1pos] = nibbleB1pos[k];
    nibbleA2pos = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, vA2)));
    p2[nibbleA2pos] = nibbleB2pos[k];
  }
  *V = 0;
  *H = 0;
  __m128i vpm, vp1, vp2, vgt;
  shuff = _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  vp1 = _mm_lddqu_si128((__m128i const *)p1);
  vp1 = _mm_shuffle_epi8(vp1, shuff);
  vp2 = _mm_lddqu_si128((__m128i const *)p2);
  vp2 = _mm_shuffle_epi8(vp2, shuff);
  for(uint32_t m=0; m<15; m++) {
    vpm = _mm_set1_epi8(p1[m]);
    vgt = _mm_cmpgt_epi8(vp1, vpm);
    *V += __builtin_popcount(_mm_movemask_epi8(vgt) >> (16 - m));
    vpm = _mm_set1_epi8(p2[m]);
    vgt = _mm_cmpgt_epi8(vp2, vpm);
    *H += __builtin_popcount(_mm_movemask_epi8(vgt) >> (16 - m));
  }
  return; 
}

uint32_t cmn15_linearconflicts2(uint64_t boardA, uint64_t boardB) {
  uint8_t p[16] = {0};
  uint64_t A, B, lc;
  cmn15_removezeronibbles(boardA, boardB, &A, &B);
  __m128i shuff1 = _mm_set_epi8(7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8);
  __m128i vA = _mm_set_epi64x(A & 0x0f0f0f0f0f0f0f0fULL, (A & 0x00f0f0f0f0f0f0f0ULL) >> 4); // 14,12,10,8,6,4,2,0,-,13,11,9,7,5,3,1
  vA = _mm_shuffle_epi8(vA, shuff1);
  __m128i vB = _mm_set_epi64x(B & 0x0f0f0f0f0f0f0f0fULL, (B & 0x00f0f0f0f0f0f0f0ULL) >> 4);
  vB = _mm_shuffle_epi8(vB, shuff1);
  __m128i vk;
  int nibbleApos, nibbleBpos;
  for(uint32_t k=1; k<16; k++) {
    vk = _mm_set1_epi8(k);
    nibbleApos = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, vA)));
    nibbleBpos = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, vB)));
    p[nibbleApos] = nibbleBpos;
  }
  lc = 0;
  __m128i vpm, vp, vgt;
  vp = _mm_lddqu_si128((__m128i const *)p);
  shuff1 = _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  vp = _mm_shuffle_epi8(vp, shuff1);
  for(uint32_t m=0; m<15; m++) {
    vpm = _mm_set1_epi8(p[m]);
    vgt = _mm_cmpgt_epi8(vp, vpm);
    lc += __builtin_popcount(_mm_movemask_epi8(vgt) >> (16 - m));
  }
  return lc; 
}
#endif

uint32_t cmn15_boardzeropos(uint64_t board) {
  board |= board >> 1;
  board |= board >> 2;
  return __builtin_ctzll(~board & 0x1111111111111111ULL) / 4;
}

_Bool cmn15_pathleneven(uint64_t boardA, uint64_t boardB) {
  uint32_t parityA = cmn15_boardzeropos(boardA);
  parityA += parityA / 4;
  uint32_t parityB = cmn15_boardzeropos(boardB);
  parityB += parityB / 4;
  return (parityA & 1) == (parityB & 1);
}

uint64_t cmn15_boardset(uint8_t *boardarray) {
  uint64_t board = 0;
  for (int32_t i=0; i<16; i++) {
    board = (board << 4) + boardarray[i];
  }
  return board;
}

/*
uint32_t cmn15_boardset7710(uint8_t *boardarray, uint8_t *map7) {
  // map7 should map 7 non-zero numbers to 1-7, zero to 0, and the rest to 0xf;
  // Output format is nibbles xabcdefg where a to g are positions of 1 to 7 respectively.
  // x is the number of fs to the left of 0.
  // e.g. f4f37ff1ff06f5f2 -> 0x6b42ed08
  // Internal range is 0x00123456 to 0x8fedcba9 for 0ffffffff7654321 and 12345670ffffffff
  // so output has 0x123456 subtracted.
  // Output range is 0 - 2413533011 inclusive.
  uint8_t db7710pos[7] = {0};
  uint8_t x = 0;
  _Bool xset = false;
  uint32_t res = 0;
  for (uint32_t i=0; i<16; i++) {
    if (!xset && (map7[boardarray[i]] == 0xf)) x++;
    if ((map7[boardarray[i]] != 0) && (map7[boardarray[i]] != 0xf)) {
      db7710pos[map7[boardarray[i]]-1] = i;
    }
  }    
  for (uint32_t i=0; i<7; i++) {
    res <<= 4;
    res += db771pos[i];
  }
  return res + (x << 28) - 0x123456;
}
*/
uint8_t cmn15_map7710lo[16] = {0,1,2,3,4,5,6,7,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf};
uint8_t cmn15_map7710hi[16] = {0,0xf,0xf,0xf,0xf,0xf,0xf,0xf,1,2,3,4,5,6,7,0xf};
/*
uint32_t cmn15_boardindex7710(uint64_t board) {
  // Output format is nibbles xabcdefg where a to g are positions of 1 to 7 respectively.
  // x is the zero's position minus number of 1-7 to right of it.
  // e.g. f4f37ff1ff06f5f2 -> 0x680ce24b
  // Internal range is 0x00123456 to 0x8fedcba9 for 0ffffffff7654321 and 12345670ffffffff
  // so output has 0x123456 subtracted.
  // Output range is 0 - 2413533011 inclusive.
  uint8_t db7710pos[7] = {0};
  uint8_t nibble;
  uint8_t x = 0;
  uint8_t count = 0;
  uint32_t res = 0;
  for (uint32_t i=0; i<16; i++) {
    nibble = board & 0xf;
    if ((nibble != 0) && (nibble != 0xf)) {
      count++;
      db7710pos[nibble-1] = i;
    }
    if (nibble == 0) x = i - count;
    board >>= 4;
  }    
  for (uint32_t i=0; i<7; i++) {
    res <<= 4;
    res += db7710pos[i];
  }
  return res + (x << 28) - 0x123456;
}
*/

uint64_t cmn15_factorial[16] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800ULL,87178291200ULL,1307674368000ULL};

void cmn15_boardindex7710x2(uint64_t *board, uint32_t *rank1, uint32_t *rank2) {
  uint8_t nibblefreq1[16] = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,8};
  uint8_t nibblefreq2[16] = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,8};
#ifdef __SSE4_2__
  __m128i vfreq1 = _mm_lddqu_si128((void*)nibblefreq1);
  __m128i vfreq2 = _mm_lddqu_si128((void*)nibblefreq2);
  __m128i vj = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
#endif
  uint16_t nibble1, nibble2, shift = 60;
  uint64_t prodfn1, sumf1;
  uint64_t prodfn2, sumf2;
  uint64_t prank1 = 0;
  uint64_t prank2 = 0;
  prodfn1 = 1;
  prodfn2 = 1;
  for (int32_t i=15; i>=0; i--) {
    nibble1 = (board[0] >> shift) & 0xf;
    nibble2 = (board[1] >> shift) & 0xf;
    sumf1 = 0;
    sumf2 = 0;
#ifndef __SSE4_2__
    for (uint32_t j=0; j<nibble1; j++) sumf1 += nibblefreq1[j];
    for (uint32_t j=0; j<nibble2; j++) sumf2 += nibblefreq2[j];
#else
    __m128i vmask1, vdec1;
    __m128i vmask2, vdec2;
    vmask1 = _mm_set1_epi8(nibble1);
    vdec1 = _mm_cmpeq_epi8(vj,vmask1);
    vmask2 = _mm_set1_epi8(nibble2);
    vdec2 = _mm_cmpeq_epi8(vj,vmask2);
    vmask1 = _mm_cmplt_epi8(vj,vmask1);
    vmask2 = _mm_cmplt_epi8(vj,vmask2);
    vmask1 = _mm_and_si128(vfreq1,vmask1);
    vmask2 = _mm_and_si128(vfreq2,vmask2);
    vmask1 = _mm_sad_epu8(vmask1, _mm_setzero_si128());
    vmask2 = _mm_sad_epu8(vmask2, _mm_setzero_si128());
    vmask1 = _mm_add_epi64(vmask1, _mm_bslli_si128(vmask1, 8));
    vmask2 = _mm_add_epi64(vmask2, _mm_bslli_si128(vmask2, 8));
    sumf1 = _mm_extract_epi64(vmask1,0);
    vfreq1 = _mm_add_epi8(vfreq1,vdec1);
    sumf2 = _mm_extract_epi64(vmask2,0);
    vfreq2 = _mm_add_epi8(vfreq2,vdec2);
#endif
    prank1 += (cmn15_factorial[i]*sumf1*prodfn1);
    prank2 += (cmn15_factorial[i]*sumf2*prodfn2);
    prodfn1 *= nibblefreq1[nibble1];
    prodfn2 *= nibblefreq2[nibble2];
    nibblefreq1[nibble1]--;
    nibblefreq2[nibble2]--;
    shift -= 4;
  }    
  *rank1 = prank1/cmn15_factorial[8];
  *rank2 = prank2/cmn15_factorial[8];
  return;
}

uint32_t cmn15_boardindex7710(uint64_t board) {
  // Output permutation rank of the nibbles of the input.
  // e.g. f4f37ff1ff06f5f2 -> 339613353
  uint8_t nibblefreq[16] = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,8};
#ifdef __SSE4_2__
  __m128i vfreq = _mm_lddqu_si128((void*)nibblefreq);
  __m128i vj = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
#endif
  uint16_t nibble, shift = 60;
  uint64_t prodfn, sumf;
  uint64_t prank = 0;
  prodfn = 1;
  for (int32_t i=15; i>=0; i--) {
    nibble = (board >> shift) & 0xf;
    sumf = 0;
#ifndef __SSE4_2__
    for (uint32_t j=0; j<nibble; j++) sumf += nibblefreq[j];
#else
    __m128i vmask, vdec;
    vmask = _mm_set1_epi8(nibble);
    vdec = _mm_cmpeq_epi8(vj,vmask);
    vmask = _mm_cmplt_epi8(vj,vmask);
    vmask = _mm_and_si128(vfreq,vmask);
    vmask = _mm_sad_epu8(vmask, _mm_setzero_si128());
    vmask = _mm_add_epi64(vmask, _mm_bslli_si128(vmask, 8));
    sumf = _mm_extract_epi64(vmask,0);
    vfreq = _mm_add_epi8(vfreq,vdec);
#endif
    prank += (cmn15_factorial[i]*sumf*prodfn);
    prodfn *= nibblefreq[nibble];
    nibblefreq[nibble]--;
    shift -= 4;
  }    
  return prank/cmn15_factorial[8];
}

_Bool cmn15_nextboard(uint64_t board, uint64_t *up, uint64_t *down, uint64_t *left, uint64_t *right) {
  uint64_t square;
  uint64_t pos;
  if ((board & 0xf) == 0) {
    *right = 0;
    *down = 0;
    pos = 0xf0000;
    square = (board & pos) >> 16;
    *up = (board & ~pos) + square;
    pos = 0xf0;
    square = (board & pos) >> 4;
    *left = (board & ~pos) + square;
    return true;
  }
  if ((board & 0xf0) == 0) {
    pos = 0xf;
    square = (board & pos);
    *right = (board & ~pos) + (square << 4);
    *down = 0;
    pos = 0xf00000;
    square = (board & pos) >> 20;
    *up = (board & ~pos) + (square << 4);
    pos = 0xf00;
    square = (board & pos) >> 8;
    *left = (board & ~pos) + (square << 4);
    return true;
  }
  if ((board & 0xf00) == 0) {
    pos = 0xf0;
    square = (board & pos) >> 4;
    *right = (board & ~pos) + (square << 8);
    *down = 0;
    pos = 0xf000000;
    square = (board & pos) >> 24;
    *up = (board & ~pos) + (square << 8);
    pos = 0xf000;
    square = (board & pos) >> 12;
    *left = (board & ~pos) + (square << 8);
    return true;
  }
  if ((board & 0xf000) == 0) {
    pos = 0xf00;
    square = (board & pos) >> 8;
    *right = (board & ~pos) + (square << 12);
    *down = 0;
    pos = 0xf0000000;
    square = (board & pos) >> 28;
    *up = (board & ~pos) + (square << 12);
    *left = 0;
    return true;
  }
  if ((board & 0xf0000) == 0) {
    *right = 0;
    pos = 0xf;
    square = (board & pos);
    *down = (board & ~pos) + (square << 16);
    pos = 0xf00000000ULL;
    square = (board & pos) >> 32;
    *up = (board & ~pos) + (square << 16);
    pos = 0xf00000;
    square = (board & pos) >> 20;
    *left = (board & ~pos) + (square << 16);
    return true;
  }
  if ((board & 0xf00000) == 0) {
    pos = 0xf0000;
    square = (board & pos) >> 16;
    *right = (board & ~pos) + (square << 20);
    pos = 0xf0;
    square = (board & pos) >> 4;
    *down = (board & ~pos) + (square << 20);
    pos = 0xf000000000ULL;
    square = (board & pos) >> 36;
    *up = (board & ~pos) + (square << 20);
    pos = 0xf000000;
    square = (board & pos) >> 24;
    *left = (board & ~pos) + (square << 20);
    return true;
  }
  if ((board & 0xf000000) == 0) {
    pos = 0xf00000;
    square = (board & pos) >> 20;
    *right = (board & ~pos) + (square << 24);
    pos = 0xf00;
    square = (board & pos) >> 8;
    *down = (board & ~pos) + (square << 24);
    pos = 0xf0000000000ULL;
    square = (board & pos) >> 40;
    *up = (board & ~pos) + (square << 24);
    pos = 0xf0000000;
    square = (board & pos) >> 28;
    *left = (board & ~pos) + (square << 24);
    return true;
  }
  if ((board & 0xf0000000) == 0) {
    pos = 0xf000000;
    square = (board & pos) >> 24;
    *right = (board & ~pos) + (square << 28);
    pos = 0xf000;
    square = (board & pos) >> 12;
    *down = (board & ~pos) + (square << 28);
    pos = 0xf00000000000ULL;
    square = (board & pos) >> 44;
    *up = (board & ~pos) + (square << 28);
    *left = 0;
    return true;
  }
  if ((board & 0xf00000000ULL) == 0) {
    *right = 0;
    pos = 0xf0000;
    square = (board & pos) >> 16;
    *down = (board & ~pos) + (square << 32);
    pos = 0xf000000000000ULL;
    square = (board & pos) >> 48;
    *up = (board & ~pos) + (square << 32);
    pos = 0xf000000000ULL;
    square = (board & pos) >> 36;
    *left = (board & ~pos) + (square << 32);
    return true;
  }
  if ((board & 0xf000000000ULL) == 0) {
    pos = 0xf00000000ULL;
    square = (board & pos) >> 32;
    *right = (board & ~pos) + (square << 36);
    pos = 0xf00000;
    square = (board & pos) >> 20;
    *down = (board & ~pos) + (square << 36);
    pos = 0xf0000000000000ULL;
    square = (board & pos) >> 52;
    *up = (board & ~pos) + (square << 36);
    pos = 0xf0000000000ULL;
    square = (board & pos) >> 40;
    *left = (board & ~pos) + (square << 36);
    return true;
  }
  if ((board & 0xf0000000000ULL) == 0) {
    pos = 0xf000000000ULL;
    square = (board & pos) >> 36;
    *right = (board & ~pos) + (square << 40);
    pos = 0xf000000;
    square = (board & pos) >> 24;
    *down = (board & ~pos) + (square << 40);
    pos = 0xf00000000000000ULL;
    square = (board & pos) >> 56;
    *up = (board & ~pos) + (square << 40);
    pos = 0xf00000000000ULL;
    square = (board & pos) >> 44;
    *left = (board & ~pos) + (square << 40);
    return true;
  }
  if ((board & 0xf00000000000ULL) == 0) {
    pos = 0xf0000000000ULL;
    square = (board & pos) >> 40;
    *right = (board & ~pos) + (square << 44);
    pos = 0xf0000000;
    square = (board & pos) >> 28;
    *down = (board & ~pos) + (square << 44);
    pos = 0xf000000000000000ULL;
    square = (board & pos) >> 60;
    *up = (board & ~pos) + (square << 44);
    *left = 0;
    return true;
  }
  if ((board & 0xf000000000000ULL) == 0) {
    *right = 0;
    pos = 0xf00000000ULL;
    square = (board & pos) >> 32;
    *down = (board & ~pos) + (square << 48);
    *up = 0;
    pos = 0xf0000000000000ULL;
    square = (board & pos) >> 52;
    *left = (board & ~pos) + (square << 48);
    return true;
  }
  if ((board & 0xf0000000000000ULL) == 0) {
    pos = 0xf000000000000ULL;
    square = (board & pos) >> 48;
    *right = (board & ~pos) + (square << 52);
    pos = 0xf000000000ULL;
    square = (board & pos) >> 36;
    *down = (board & ~pos) + (square << 52);
    *up = 0;
    pos = 0xf00000000000000ULL;
    square = (board & pos) >> 56;
    *left = (board & ~pos) + (square << 52);
    return true;
  }
  if ((board & 0xf00000000000000ULL) == 0) {
    pos = 0xf0000000000000ULL;
    square = (board & pos) >> 52;
    *right = (board & ~pos) + (square << 56);
    pos = 0xf0000000000ULL;
    square = (board & pos) >> 40;
    *down = (board & ~pos) + (square << 56);
    *up = 0;
    pos = 0xf000000000000000ULL;
    square = (board & pos) >> 60;
    *left = (board & ~pos) + (square << 56);
    return true;
  }
  if ((board & 0xf000000000000000ULL) == 0) {
    pos = 0xf00000000000000ULL;
    square = (board & pos) >> 56;
    *right = (board & ~pos) + (square << 60);
    pos = 0xf00000000000ULL;
    square = (board & pos) >> 44;
    *down = (board & ~pos) + (square << 60);
    *up = 0;
    *left = 0;
    return true;
  }
  return false;
}


_Bool cmn15_insolutionboards(uint64_t board, uint64_t *solutionboards, uint32_t depth) {
  for (int32_t i=depth; i>=0; i--) {
    if (solutionboards[i] == board) return true;
  }
  return false;
}

uint32_t cmn15_heuristicMD(uint64_t boardA, uint8_t *Bx, uint8_t *By) {
  uint32_t nibble, md = 0;
  int8_t diffx, diffy, Ax[16], Ay[16];
  for (uint32_t i=0; i<16; i++) {
    nibble = boardA & 0xf;
    Ax[nibble] = i & 0x3;
    Ay[nibble] = i >> 2;
    boardA >>= 4;
  }
  for (uint32_t i=1; i<16; i++) {
    diffx = Ax[i] - Bx[i];
    diffx = (diffx > 0 ? diffx : -diffx);
    diffy = Ay[i] - By[i];
    diffy = (diffy > 0 ? diffy : -diffy);
    md += diffx + diffy;
  }
  return md;
}

uint32_t cmn15_heuristicMD2(uint64_t boardA, uint64_t boardB) {
  uint32_t nibble, md = 0;
  int8_t diffx, diffy, Ax[16], Ay[16], Bx[16], By[16];
  for (uint32_t i=0; i<16; i++) {
    nibble = boardA & 0xf;
    Ax[nibble] = i & 0x3;
    Ay[nibble] = i >> 2;
    nibble = boardB & 0xf;
    Bx[nibble] = i & 0x3;
    By[nibble] = i >> 2;
    boardA >>= 4;
    boardB >>= 4;
  }
  for (uint32_t i=1; i<16; i++) {
    diffx = Ax[i] - Bx[i];
    diffx = (diffx > 0 ? diffx : -diffx);
    diffy = Ay[i] - By[i];
    diffy = (diffy > 0 ? diffy : -diffy);
    md += diffx + diffy;
  }
  return md;
}

#ifdef __SSSE3__
static inline uint64_t cmn15_vnibblestou64(__m128i vnibbles) {
  uint8_t nibbles[16];
  _mm_storeu_si128((void*)nibbles, vnibbles);
  return cmn15_boardset(nibbles);
}

_Bool cmn15_boards7710(uint64_t board, uint64_t *boards7710) {
  __m128i shuff = _mm_setr_epi8(7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8);
  __m128i vnibbles = _mm_set_epi64x(board, board >> 4); 
  vnibbles = _mm_and_si128(vnibbles, _mm_set1_epi8(0xf));
  vnibbles = _mm_shuffle_epi8(vnibbles, shuff);
  __m128i shuff1 = _mm_setr_epi8(0,1,2,3,4,5,6,7,15,15,15,15,15,15,15,15);
  __m128i mask = _mm_shuffle_epi8(shuff1, vnibbles);
  boards7710[0] = cmn15_vnibblestou64(mask); // 0123456789abcdef -> 01234567ffffffff
  __m128i shuff2 = _mm_setr_epi8(0,15,15,15,15,15,15,15,15,1,2,3,4,5,6,7);
  mask = _mm_shuffle_epi8(shuff2, vnibbles);
  boards7710[1] = cmn15_vnibblestou64(mask); // 0123456789abcdef -> 0ffffffff1234567
  return true;
}
#else
_Bool cmn15_boards7710(uint64_t board, uint64_t *boards7710) {
  uint8_t nibbles[16];
  uint8_t nibbles7710[16];
  for (int32_t i=15; i>=0; i--) {
    nibbles[i] = board & 0xf;
    board >>= 4;    
  }
  for (uint32_t i=0; i<16; i++) {
    if (nibbles[i] <= 7) {
      nibbles7710[i] = nibbles[i];
    } else {
      nibbles7710[i] = 0xf;
    }
  }
  boards7710[0] = cmn15_boardset(nibbles7710);
  for (uint32_t i=0; i<16; i++) {
    if (nibbles[i] > 8) {
      nibbles7710[i] = nibbles[i] - 8;
    } else {
      nibbles7710[i] = 0xf;
    }
    if (nibbles[i] == 0) nibbles7710[i] = 0;
  }
  boards7710[1] = cmn15_boardset(nibbles7710);
  return true;
}
#endif

uint8_t cmn15_boardpos(uint8_t n, uint64_t board) {
  // Returns position of n in board.  
  uint8_t nibble;
  for (uint32_t i=0; i<16; i++) {
    nibble = board & 0xf;
    if (nibble == n) return i;
    board >>= 4;  
  }  
  exit(1);
  return 0xff;
}

uint32_t cmn15_heuristic7710(cmn15_db7710_t db7710, uint64_t boardA) {
  uint32_t boardindex7710[2];
  uint64_t boards7710[2];
  cmn15_boards7710(boardA, boards7710);
  uint8_t onepos = cmn15_boardpos(CMN15_ONEDBNUM, boardA);
  uint8_t oneposx = onepos % 4;
  uint8_t oneposy = onepos / 4;
  //uint32_t fMD = 0;
  uint32_t fMD = (oneposx < db7710.onedbx ? db7710.onedbx - oneposx : oneposx - db7710.onedbx);
  fMD += (oneposy < db7710.onedby ? db7710.onedby - oneposy : oneposy - db7710.onedby);
  //boardindex7710[1] = cmn15_boardindex7710(boards7710[1]);
  //boardindex7710[0] = cmn15_boardindex7710(boards7710[0]);
  cmn15_boardindex7710x2(boards7710, &boardindex7710[0], &boardindex7710[1]);
  __builtin_prefetch(&db7710.array[1][boardindex7710[1]], BUILTIN_PREFETCH_ARG2_FOR_READ, BUILTIN_PREFETCH_ARG3_TEMPORAL_LOCALITY_HIGH);
  return fMD + db7710.array[0][boardindex7710[0]] + db7710.array[1][boardindex7710[1]];
}

uint64_t cmn15_boardtranspose(uint64_t board) {
  /*
     1 2 3 4    1 5 9 d
     5 6 7 8    2 6 a e
     9 a b c    3 7 b f
     d e f g    4 8 c g
  */
  // Transpose 2x2 blocks first
  uint64_t res = (board & 0xff00ff0000ff00ffULL) | ((board & 0xff00ff00000000ULL) >> 24) | ((board & 0xff00ff00ULL) << 24); 
  // 2<->5 4<->7 10<->13 12<->15  
  res = (res & 0xf0f00f0ff0f00f0fULL) | ((res & 0x0f0f00000f0f0000) >> 12) | ((res & 0x0f0f00000f0f0) << 12);
  return res;
}

uint64_t cmn15_boardflipxaxis(uint64_t board) {
  /*
     1 2 3 4    d e f g
     5 6 7 8    9 a b c
     9 a b c    5 6 7 8
     d e f g    1 2 3 4
  */
  // Flip rows 1,2 and 3,4 first
  uint64_t res = ((board & 0xffff0000ffff0000ULL) >> 16) | ((board & 0x0000ffff0000ffffULL) << 16);
  res = ((res & 0xffffffff00000000ULL) >> 32) | ((res & 0x00000000ffffffffULL) << 32);
  return res;
}
 
uint64_t cmn15_boardflipyaxis(uint64_t board) {
  /*
     1 2 3 4    4 3 2 1
     5 6 7 8    8 7 6 5
     9 a b c    c b a 9
     d e f g    g f e d
  */
  uint64_t res = ((board & 0xff00ff00ff00ff00ULL) >> 8) | ((board & 0x00ff00ff00ff00ffULL) << 8);
  res = ((res & 0xf0f0f0f0f0f0f0f0ULL) >> 4) | ((res & 0x0f0f0f0f0f0f0f0fULL) << 4);
  return res;
}

uint64_t cmn15_boardtranslate(uint64_t board, uint32_t boardtranslation) {
  if (boardtranslation & 0x4) board = cmn15_boardflipyaxis(board);
  if (boardtranslation & 0x2) board = cmn15_boardflipxaxis(board);
  if (boardtranslation & 0x1) board = cmn15_boardtranspose(board);
  return board;
}

uint64_t cmn15_boarduntranslate(uint64_t board, uint32_t boardtranslation) {
  if (boardtranslation & 0x1) board = cmn15_boardtranspose(board);
  if (boardtranslation & 0x2) board = cmn15_boardflipxaxis(board);
  if (boardtranslation & 0x4) board = cmn15_boardflipyaxis(board);
  return board;
}

_Bool cmn15_boardmap(uint64_t board, uint64_t *mapboard, uint8_t *boardmap) {
  uint8_t nibble;
  *mapboard = 0;
  for (uint32_t i=0; i<16; i++) {
    nibble = board & 0xf;
    *mapboard |= ((uint64_t)boardmap[nibble] << 4*i);
    board >>= 4;
  }
  return true;
}

_Bool cmn15_boardmakemap(uint64_t board, uint8_t *boardmap, uint8_t *boardunmap) {
  uint8_t nibble;
  uint32_t nibbleix;
  boardmap[0] = 0;
  boardunmap[0] = 0;
  for (nibbleix = 15; nibbleix>0; ) {
    nibble = board & 0xf;
    if (nibble) {
      boardmap[nibble] = nibbleix;
      boardunmap[nibbleix] = nibble;
      nibbleix--;
    }  
    board >>= 4;
  }
  return true;
}

_Bool cmn15_boardgettranslation(uint64_t board, uint64_t *newboard, uint32_t *zeropos, uint32_t *boardtranslation) {
  // newboard has zero in positions 0,1, or 5.
  // returns 0 to 7, bits denoting which of operations transpose, flipxaxis, or flipyaxis are applied.
  *zeropos = cmn15_boardzeropos(board);
  if ((*zeropos == 0) || (*zeropos == 1) || (*zeropos == 5)) {
    *newboard = board;
    *boardtranslation = 0;
    return true;
  }
  if (*zeropos == 4) {
    *newboard = cmn15_boardtranspose(board);
    *zeropos = cmn15_boardzeropos(*newboard);
    *boardtranslation = 1;
    return true;
  }
  if ((*zeropos == 9) || (*zeropos == 12) || (*zeropos == 13)) {
    *newboard = cmn15_boardflipxaxis(board);
    *zeropos = cmn15_boardzeropos(*newboard);
    *boardtranslation = 2;
    return true;
  }
  if (*zeropos == 8) {
    *newboard = cmn15_boardtranspose(cmn15_boardflipxaxis(board));
    *zeropos = cmn15_boardzeropos(*newboard);
    *boardtranslation = 3;
    return true;
  }
  if (!cmn15_boardgettranslation(cmn15_boardflipyaxis(board), newboard, zeropos, boardtranslation)) return false;
  *boardtranslation += 4;
  return true;
}
 
#ifndef __SSSE3__
uint32_t cmn15_heuristicLC(uint64_t boardA, uint64_t boardB) {
  uint32_t V = cmn15_linearconflicts(boardA, boardB);
  V = (V / 3) + (V % 3); 
  uint32_t H = cmn15_linearconflicts(cmn15_boardtranspose(boardA), cmn15_boardtranspose(boardB));
  H = (H / 3) + (H % 3); 
  return H+V; 
}

_Bool cmn15_IDDLSrh(cmn15_db7710_t db7710, uint64_t boardB, _Bool pathleneven, uint64_t *solutionboards, uint32_t depth, uint32_t maxdepth, uint64_t *hevals) {
  uint64_t board = solutionboards[depth];
  uint64_t adjacentboards[4];
  uint32_t h, hLC, h7710;
  cmn15_nextboard(board, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
  for (uint32_t i=0; i<4; i++) {
    if (adjacentboards[i] != 0) {
      if (adjacentboards[i] == boardB) {
        solutionboards[depth+1] = boardB;
        return true;        
      }
      if (!cmn15_insolutionboards(adjacentboards[i], solutionboards, depth)) {
        //pathleneven = cmn15_pathleneven(adjacentboards[i], boardB);
        hLC = cmn15_heuristicLC(adjacentboards[i], boardB);
        h = hLC;
        h7710 = cmn15_heuristic7710(db7710, adjacentboards[i]);
        h = (h < h7710 ? h7710 : h);
        (*hevals)++;
        if ((depth + h) <= maxdepth) {
          depth++;
          solutionboards[depth] = adjacentboards[i];
          if (depth < maxdepth) {
            if (cmn15_IDDLSrh(db7710, boardB, !pathleneven, solutionboards, depth, maxdepth, hevals)) return true;
          } 
          depth--;
        }
      }
    }
  }
  return false;
}

_Bool cmn15_IDDLS(cmn15_db7710_t db7710F, uint64_t boardFA, cmn15_db7710_t db7710R, uint64_t boardRA, _Bool pathleneven, uint64_t *solutionboardsF, uint64_t *solutionboardsR, uint32_t *minsteps, uint64_t *hevalsF, uint64_t *hevalsR) {
  *hevalsF = 0;
  *hevalsR = 0;
  solutionboardsF[0] = boardFA;
  uint64_t boardFB = db7710F.goalboard;
  if (boardFA == boardFB) {
    *minsteps = 0;
    return true;
  }
  uint64_t up = 0, down = 0, left = 0, right = 0;
  if (!cmn15_nextboard(boardFA, &up, &down, &left, &right)) return false;
  if ((up == boardFB) || (down == boardFB) || (left == boardFB) || (right == boardFB)) {
    solutionboardsF[1] = boardFB;
    solutionboardsR[1] = 0;
    *minsteps = 1;
    return true;
  }
  uint32_t maxdepth = cmn15_heuristicLC(boardFA, boardFB);
  if ((pathleneven) && (maxdepth & 1)) maxdepth++;
  printf("Completed search at depth ");
  while (!cmn15_IDDLSrh(db7710F, boardFB, !pathleneven, solutionboardsF, 0, maxdepth, hevalsF)) {
    printf("%u ", maxdepth);
    fflush(stdout);
    maxdepth += 2;
  }
  printf("\n");
  *minsteps = maxdepth;
  return true;
}
#else
uint32_t cmn15_heuristicLC(uint64_t boardA, uint32_t *nibble1pos, uint32_t *nibble2pos) {
  uint32_t V, H;
  cmn15_linearconflicts(boardA, nibble1pos, cmn15_boardtranspose(boardA), nibble2pos, &V, &H);
  V = (V / 3) + (V % 3); 
  H = (H / 3) + (H % 3); 
  return H+V; 
}

typedef struct {
  cmn15_db7710_t db7710;
  uint64_t *boardB;
  uint64_t *solutionboards;
  uint32_t *depth;
  uint64_t *hevals;
  uint32_t *nibble1pos;
  uint32_t *nibble2pos;
  pthread_mutex_t *interruptmutex;
  _Bool *interrupt;
  _Bool *ret;
  uint32_t maxdepth;
  _Bool pathleneven;
} cmn15_IDDLSrhargs_t;

void *cmn15_IDDLSrh(void *args1) {
  cmn15_IDDLSrhargs_t *args = args1;
  uint64_t board = args->solutionboards[*(args->depth)];
  uint64_t adjacentboards[4];
  uint32_t h, hLC, h7710;
  cmn15_nextboard(board, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
  for (uint32_t i=0; i<4; i++) {
    if (adjacentboards[i] != 0) {
      if (adjacentboards[i] == *(args->boardB)) {
        args->solutionboards[*(args->depth)+1] = *(args->boardB);
        pthread_mutex_lock(args->interruptmutex);
        *(args->interrupt) = true;
        //printf("Target Found. Interrupting...\n");
        pthread_mutex_unlock(args->interruptmutex);
        *(args->ret) = true;
        return NULL;        
      }
      if (!cmn15_insolutionboards(adjacentboards[i], args->solutionboards, *(args->depth))) {
        hLC = cmn15_heuristicLC(adjacentboards[i], args->nibble1pos, args->nibble2pos);
        h = hLC;
        h7710 = cmn15_heuristic7710(args->db7710, adjacentboards[i]);
        h = (h < h7710 ? h7710 : h);
        (*(args->hevals))++;
        if ((*(args->hevals) & 0xffffULL) == 0) {
          pthread_mutex_lock(args->interruptmutex);
          if (*(args->interrupt)) {
            pthread_mutex_unlock(args->interruptmutex);
            //printf("Interrupted.\n");
            *(args->ret) = true;
            return NULL;        
          }
          pthread_mutex_unlock(args->interruptmutex);
        }
        if ((*(args->depth) + h) <= args->maxdepth) {
          (*(args->depth))++;
          args->solutionboards[*(args->depth)] = adjacentboards[i];
          if (*(args->depth) < args->maxdepth) {
            cmn15_IDDLSrh(args1);
            if (*(args->ret) == true) {
              return NULL;
            }
          } 
          (*(args->depth))--;
        }
      }
    }
  }
  *(args->ret) = false;
  if (*(args->depth) == 0) {
    pthread_mutex_lock(args->interruptmutex);
    if (!(*(args->interrupt))) {
      //printf("Search Completed. Interrupting...\n");      
      *(args->interrupt) = true;
    }
    pthread_mutex_unlock(args->interruptmutex);
  }
  return NULL;
}

_Bool cmn15_IDDLS(cmn15_db7710_t db7710F, uint64_t boardFA, cmn15_db7710_t db7710R, uint64_t boardRA, _Bool pathleneven, uint64_t *solutionboardsF, uint64_t *solutionboardsR, uint32_t *minsteps, uint64_t *hevalsF, uint64_t *hevalsR) {
  *hevalsF = 0;
  *hevalsR = 0;
  solutionboardsF[0] = boardFA;
  solutionboardsR[0] = boardRA;
  uint64_t boardFB = db7710F.goalboard;
  uint64_t boardRB = db7710R.goalboard;
  if (boardFA == boardFB) {
    *minsteps = 0;
    return true;
  }
  uint64_t up = 0, down = 0, left = 0, right = 0;
  if (!cmn15_nextboard(boardFA, &up, &down, &left, &right)) return false;
  if ((up == boardFB) || (down == boardFB) || (left == boardFB) || (right == boardFB)) {
    solutionboardsF[1] = boardFB;
    solutionboardsR[1] = 0;
    *minsteps = 1;
    return true;
  }
  uint32_t nibble1posF[16], nibble2posF[16];
  uint32_t nibble1posR[16], nibble2posR[16];
  uint64_t board1nozero, board2nozero;
  __m128i shuff = _mm_set_epi8(7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8);
  cmn15_removezeronibbles(boardFB, cmn15_boardtranspose(boardFB), &board1nozero, &board2nozero);
  __m128i v1 = _mm_set_epi64x(board1nozero & 0x0f0f0f0f0f0f0f0fULL, (board1nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  __m128i v2 = _mm_set_epi64x(board2nozero & 0x0f0f0f0f0f0f0f0fULL, (board2nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  v1 = _mm_shuffle_epi8(v1, shuff);
  v2 = _mm_shuffle_epi8(v2, shuff);
  __m128i vk;
  for(uint32_t k=1; k<16; k++) {
    vk = _mm_set1_epi8(k);
    nibble1posF[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v1)));
    nibble2posF[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v2)));
  }
  cmn15_removezeronibbles(boardRB, cmn15_boardtranspose(boardRB), &board1nozero, &board2nozero);
  v1 = _mm_set_epi64x(board1nozero & 0x0f0f0f0f0f0f0f0fULL, (board1nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  v2 = _mm_set_epi64x(board2nozero & 0x0f0f0f0f0f0f0f0fULL, (board2nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  v1 = _mm_shuffle_epi8(v1, shuff);
  v2 = _mm_shuffle_epi8(v2, shuff);
  for(uint32_t k=1; k<16; k++) {
    vk = _mm_set1_epi8(k);
    nibble1posR[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v1)));
    nibble2posR[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v2)));
  }
  uint32_t maxdepth = cmn15_heuristicLC(boardFA, nibble1posF, nibble2posF);
  _Bool interrupt = false;  
  _Bool retF = false;
  _Bool retR = false;
  uint32_t depthF = 0, depthR = 0;
  pthread_mutex_t interruptmutex = PTHREAD_MUTEX_INITIALIZER;
  if ((pathleneven) && (maxdepth & 1)) maxdepth++;
  printf("Completed search at depth ");
  cmn15_IDDLSrhargs_t IDDLSrhargsF, IDDLSrhargsR;
  IDDLSrhargsF.interruptmutex = &interruptmutex;
  IDDLSrhargsR.interruptmutex = &interruptmutex;
  IDDLSrhargsF.db7710 = db7710F;
  IDDLSrhargsR.db7710 = db7710R;
  IDDLSrhargsF.boardB = &boardFB;
  IDDLSrhargsR.boardB = &boardRB;
  IDDLSrhargsF.pathleneven = pathleneven;
  IDDLSrhargsR.pathleneven = pathleneven;
  IDDLSrhargsF.solutionboards = solutionboardsF;
  IDDLSrhargsR.solutionboards = solutionboardsR;
  IDDLSrhargsF.depth = &depthF;
  IDDLSrhargsR.depth = &depthR;
  IDDLSrhargsF.hevals = hevalsF;
  IDDLSrhargsR.hevals = hevalsR;
  IDDLSrhargsF.nibble1pos = nibble1posF;
  IDDLSrhargsR.nibble1pos = nibble1posR;
  IDDLSrhargsF.nibble2pos = nibble2posF;
  IDDLSrhargsR.nibble2pos = nibble2posR;
  IDDLSrhargsF.interrupt = &interrupt;
  IDDLSrhargsR.interrupt = &interrupt;
  IDDLSrhargsF.ret = &retF;
  IDDLSrhargsR.ret = &retR;
  pthread_t thread_idF; //, thread_idR;
  while (true) {
    IDDLSrhargsF.maxdepth = maxdepth;
    IDDLSrhargsR.maxdepth = maxdepth;
    pthread_create(&thread_idF, NULL, cmn15_IDDLSrh, (void*)&IDDLSrhargsF);
    //pthread_create(&thread_idR, NULL, cmn15_IDDLSrh, (void*)&IDDLSrhargsR);
    //cmn15_IDDLSrh(&IDDLSrhargsF); // Create thread
    cmn15_IDDLSrh(&IDDLSrhargsR); 
    pthread_join(thread_idF, NULL);
    //pthread_join(thread_idR, NULL);
    if ((solutionboardsF[maxdepth] == boardFB) || (solutionboardsR[maxdepth] == boardRB)) break;
    printf("%u ", maxdepth);
    fflush(stdout);
    maxdepth += 2;
    interrupt = false;
    retF = false;
    retR = false;
  }
  printf("\n");
  *minsteps = maxdepth;
  return true;
}
#endif

uint32_t cmn15_boardparity(uint64_t boardA) {
  uint8_t nibbles[16];
  uint8_t nibble;
  for (uint32_t i=0; i<16; i++) {
    nibbles[i] = boardA & 0xf;
    boardA >>= 4;    
  }
  uint32_t parity = 0;
  for (uint32_t i=0; i<16; i++) {
    for (uint32_t j=i+1; j<16; j++) {
      if (nibbles[j] == i) {
        nibble = nibbles[i];
        nibbles[i] = nibbles[j];
        nibbles[j] = nibble;
        parity++;
        break;
      }
    }    
  }
  return parity;
}

_Bool cmn15_printsoln(uint64_t *solutionboards, uint32_t minsteps, uint8_t *unmap, uint32_t boardtranslation, _Bool reverse) {
  uint64_t prevboard = 0;
  uint64_t board;
  uint64_t adjacentboards[4];
  uint32_t j;
  if (reverse) {
    for (int32_t i=minsteps; i>=0; i--) {
      if (!cmn15_boardmap(solutionboards[i], &board, unmap)) return false;
      board = cmn15_boarduntranslate(board, boardtranslation);
      if (i<minsteps) {
        cmn15_nextboard(prevboard, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
        for (j=0; j<4; j++) {
          if (adjacentboards[j] == board) break;
        }
      }
      if (j >= 4) return false;
      prevboard = board;
      printf("%016lx \n", board);
    }
  } else {
    for (uint32_t i=0; i<=minsteps; i++) {
      if (!cmn15_boardmap(solutionboards[i], &board, unmap)) return false;
      board = cmn15_boarduntranslate(board, boardtranslation);
      if (i>0) {
        cmn15_nextboard(prevboard, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
        for (j=0; j<4; j++) {
          if (adjacentboards[j] == board) break;
        }
      }
      if (j >= 4) return false;
      prevboard = board;
      printf("%016lx \n", board);
    }
  }
  return true;
}

_Bool cmn15_hextoboard(char* str, uint64_t *board) {
  *board = 0;
  char hexchars[16] = {0};
  uint8_t digit, digitcount = 0;
  while (*str) {
    (*board) <<= 4;
    if ((*str >= '0') && (*str <= '9')) {
      digit = *str - '0';
      if (hexchars[digit] != 0) return false;
      hexchars[digit] = 1;
      (*board) += digit;
    } else {
      if ((*str >= 'a') && (*str <= 'f')) {
        digit = *str + 10 - 'a';
        if (hexchars[digit] != 0) return false;
        hexchars[digit] = 1;
        (*board) += digit;
      } else {
        return false;
      }
    }
    digitcount++;
    if (digitcount > 16) return false;
    str++;
  }
  if (digitcount < 16) return false;
  return true;
}

_Bool cmn15_solvable(uint64_t boardA, uint64_t boardB, _Bool pathleneven) {
  //return true;
  uint32_t parityA, parityB;
  parityA = cmn15_boardparity(boardA);
  parityB = cmn15_boardparity(boardB);
  //printf("Parity of %016lx is %u\n", boardA, parityA);
  //printf("Parity of %016lx is %u\n", boardB, parityB);
  //return true;
  if (pathleneven) {
    return (1 & parityA) == (1 & parityB);
  } else {
    return (1 & parityA) != (1 & parityB);
  }
}


_Bool cmn15_is1to7move(uint64_t beforeboard, uint64_t afterboard) {
 // afterboard is 1 move after beforeboard.
 // Return true if the moved nibble has value 1 to 7 inclusive, false otherwise.
 uint64_t xorboards = beforeboard ^ afterboard;
 xorboards >>= __builtin_ctzll(xorboards);
 xorboards &= 0xf;
 return (xorboards >= 1) && (xorboards <= 7);
}

_Bool cmn15_builddb7710r(uint8_t *db7710, uint64_t *solutionboards, uint32_t depth, uint32_t moves7710, uint32_t maxdepth, _Bool *depthlimitreached) {
  uint64_t board7710 = solutionboards[depth];
  uint64_t adjacentboards[4];
  uint32_t thismoves7710;
  uint32_t index;
  _Bool is1to7move;
  cmn15_nextboard(board7710, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
  for (uint32_t i=0; i<4; i++) {
    if (adjacentboards[i] != 0) {
      if (!cmn15_insolutionboards(adjacentboards[i], solutionboards, depth)) {
        is1to7move = cmn15_is1to7move(board7710, adjacentboards[i]);
        thismoves7710 = moves7710 + is1to7move;
        index = cmn15_boardindex7710(adjacentboards[i]);
        if (((db7710[index] & 0x7f) > thismoves7710) || (db7710[index] == (thismoves7710 + 0x80))) {
          db7710[index] = thismoves7710;
          //printf("Node %016lx updated to %u\n", adjacentboards[i], thismoves555);
          depth++;
          solutionboards[depth] = adjacentboards[i];
          if (depth < maxdepth) {
            if (!cmn15_builddb7710r(db7710, solutionboards, depth, thismoves7710, maxdepth, depthlimitreached)) return false;
          } else {
            *depthlimitreached = true;
          }
          depth--;
        }
      }
    }
  }
  return true;
}

_Bool cmn15_builddb7710(cmn15_db7710_t *db7710, uint64_t board, char *argv0) {
  //uint64_t db7710size = 2413533012; 
  uint64_t db7710size = 518918400;
  db7710->arraysize = db7710size;
  if ((board != 0x123456789abcdef0ULL) && (board != 0x123456789abcde0fULL) && (board != 0x123456789a0bcdefULL)) return false;
  db7710->goalboard = board;
  uint8_t fpos = cmn15_boardpos(CMN15_ONEDBNUM, board);
  db7710->onedbx = fpos % 4;
  db7710->onedby = fpos / 4;
  char filename[1024];
  if (board == 0x123456789abcdef0ULL) sprintf(filename, "%s.db0", argv0); // md5* = c00ba9dfe05494918ba3a1f590fa752d
  if (board == 0x123456789abcde0fULL) sprintf(filename, "%s.db1", argv0); //   -  "  -
  if (board == 0x123456789a0bcdefULL) sprintf(filename, "%s.db2", argv0); // md5* = 34403647ffc4400c9589fe74e5ea7b8b
  // * If mapped 0123456789abcdef -> 01234567ffffffff and 0fffffff1234567f.
  FILE *fp;
  size_t freadsize;
  db7710->array[0] = malloc(db7710size);
  if (db7710->array[0] == NULL) return false;
  db7710->array[1] = malloc(db7710size);
  if (db7710->array[1] == NULL) {
    free(db7710->array[0]);      
    return false;
  }
  fp = fopen(filename, "rb");
  if (fp) {
    freadsize = fread(db7710->array[0], 1, db7710size, fp);
    if (freadsize == db7710size) {
      freadsize = fread(db7710->array[1], 1, db7710size, fp);
      if (freadsize == db7710size) {
        fclose(fp);
        printf("Loaded 7-7-1+0 pattern database.\n");
/*        
        uint8_t max0 = 0;
        uint8_t max1 = 0;
        for (uint32_t i=0; i<db7710size; i++) max0 = (max0 > db7710->array[0][i] ? max0 : db7710->array[0][i]);
        for (uint32_t i=0; i<db7710size; i++) max1 = (max1 > db7710->array[1][i] ? max1 : db7710->array[1][i]);
        printf("Maximum steps are %u %u.\n", max0, max1);
*/
        return true;
      } 
    }
    fclose(fp);
  }
  printf("Building 7-7-1+0 pattern database. Please wait...\n");
  uint64_t solutionboards[1024];  
  uint64_t boards7710[2];
  cmn15_boards7710(board, boards7710);
  uint32_t visited;
  uint32_t numpermutations7710 = 518918400;
  _Bool depthlimitreached;
  for (uint32_t i=0; i<2; i++) {
    //for (uint32_t j=0; j<db7710size; j++) db7710[i][j] = 0x7f;
    memset(db7710->array[i], 0x7f, db7710size);
    solutionboards[0] = boards7710[i];
    printf("%016lx \n", boards7710[i]);
    uint32_t maxdepth = 20;    
    visited = 0;
    depthlimitreached = true;
    while ((visited < numpermutations7710) || depthlimitreached) {
      printf("Limiting depth to %u - ", maxdepth);
      fflush(stdout);
      depthlimitreached = false;
      for (uint32_t j=0; j<db7710size; j++) db7710->array[i][j] |= 0x80;
      db7710->array[i][cmn15_boardindex7710(boards7710[i])] = 0;
      if (!cmn15_builddb7710r(db7710->array[i], solutionboards, 0, 0, maxdepth, &depthlimitreached)) {
        free(db7710->array[0]);
        free(db7710->array[1]);
        return false;
      }
      visited = 0;
      for (uint32_t j=0; j<db7710size; j++) {
        if((db7710->array[i][j] & 0x80) == 0) visited++;
      }
      printf("%u visited, %u remaining\n", visited, numpermutations7710 - visited);
      if (maxdepth < 60) {
        maxdepth += 10;
      } else {
        maxdepth = 1000;
      }
    }
  }
  printf("Pattern database built successfully.\n");
  if (strlen(argv0) <= 1000) {
    fp = fopen(filename, "wb");
    if (fp) {
      fwrite(db7710->array[0], 1, db7710size, fp);
      fwrite(db7710->array[1], 1, db7710size, fp);
      fclose(fp);
    }
  }
  return true;
}

uint64_t cmn15_randomboard(uint64_t goalboard) { 
  uint8_t j,k,temp, nibbles[16];
  uint64_t board = goalboard;
  for (uint32_t i=0; i<16; i++) {
    nibbles[15-i] = board & 0xf;
    board >>= 4;
  }
  while (true) {
    for (uint32_t i=0; i<1000; i++) {
      j = rand() % 16;
      k = rand() % 16;
      temp = nibbles[j];
      nibbles[j] = nibbles[k];
      nibbles[k] = temp;
    }
    board = cmn15_boardset(nibbles);
    if (cmn15_solvable(board, goalboard, cmn15_pathleneven(board, goalboard))) break;
  }
  return board;
}

int main(int argc, char*argv[]) {
  //printf("%u\n", cmn15_boardindex7710(0xf4f37ff1ff06f5f2ULL));
  //exit(0);
  uint8_t boardarrayA[16] = {15,14,1,6,9,11,4,12,0,10,7,3,13,8,5,2}; 
  uint64_t boardA = cmn15_boardset(boardarrayA);
  uint8_t boardarrayB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0}; // 123456789abcdef0
  uint64_t boardB = cmn15_boardset(boardarrayB);
  uint64_t boardFA, boardFB, boardRA, boardRB;
  //srand(4321);
  if (argc == 1) {
    printf("This program performs an exhaustive search and finds an optimally short path from two arbitrary board states of the famous 'fifteen' sliding square puzzle, if there is a solution, and outputs a 'not reachable' message otherwise. The boards are input as arguments with hex values for the squares and the empty square is zero. The normally solved board state (endboard) is 123456789abcdef0. It uses an Iterative Deepening Depth Limited Search method using the maximum of Linear Conflicts and an additive 7-7-1+0 pattern database heuristics.\nThe pattern database is generated by this program and typically takes up to an hour if it has not been saved previuosly, otherwise, it will be loaded from file. Random board pairs will probably be solved just seconds after the database is built/loaded. Difficult inputs will likely be solved in minutes, but extreme cases may take up to an hour or so. Exhaustive search is computationally expensive, but the results are guaranteed to be optimal, meaning there does not exist a shorter solution than the one returned. Relaxing the requirement for optimality can massively reduce the time to find solutions, but this program is for strictly optimal solutions only, and is optimised for difficult inputs (60+ steps). This program's run-time data memory requirement is approx. 2GB. It also requires approx. 3 GB of storage space.\n");
    printf("Copyright:- Simon Goater August 2024\n");
    printf("Usage:- %s startboard endboard\n", argv[0]);
    printf("e.g. %s 43218765cba90efd 5248a03ed6bc1f97\n", argv[0]);
    exit(0);
  }
  if (argc > 1) {
    if(!cmn15_hextoboard(argv[1], &boardA)) {
      printf("startboard invalid!\n");
      exit(1);
    }
  }
  if (argc > 2) {
    if(!cmn15_hextoboard(argv[2], &boardB)) {
      printf("endboard invalid!\n");
      exit(1);
    }
  }
  _Bool pathleneven = cmn15_pathleneven(boardA, boardB);
  if (!cmn15_solvable(boardA, boardB, pathleneven)) {
    printf("%016lx is not reachable from %016lx.\n", boardB, boardA);
    exit(0);
  }
  //uint32_t countdown = 25000;
  //boardA = cmn15_randomboard(boardB);
  //boardB = cmn15_randomboard(boardA);
  //while (countdown) {
  //printf("%u - Solving for %016lx %016lx.\n", countdown, boardA, boardB);  
  boardFA = boardA;
  boardFB = boardB;
  boardRA = boardB;
  boardRB = boardA;
  uint32_t zeropos;
  uint32_t boardtranslation, boardtranslationF, boardtranslationR;
  if (!cmn15_boardgettranslation(boardFB, &boardFB, &zeropos, &boardtranslationF)) {
      printf("cmn15_boardgettranslation() Failed!\n");
      exit(1);
  }
  boardFA = cmn15_boardtranslate(boardFA, boardtranslationF);
  if (!cmn15_boardgettranslation(boardRB, &boardRB, &zeropos, &boardtranslationR)) {
      printf("cmn15_boardgettranslation() Failed!\n");
      exit(1);
  }
  boardRA = cmn15_boardtranslate(boardB, boardtranslationR);
  uint8_t boardmapF[16];
  uint8_t boardunmapF[16];
  uint8_t boardmapR[16];
  uint8_t boardunmapR[16];
  uint8_t *boardunmap;
  if (!cmn15_boardmakemap(boardFB, boardmapF, boardunmapF)) {
      printf("cmn15_boardmakemap() Failed!\n");
      exit(1);
  }
  if (!cmn15_boardmap(boardFB, &boardFB, boardmapF)) {
      printf("cmn15_boardmap() Failed!\n");
      exit(1);
  }
  if (!cmn15_boardmap(boardFA, &boardFA, boardmapF)) {
      printf("cmn15_boardmap() Failed!\n");
      exit(1);
  }
  if (!cmn15_boardmakemap(boardRB, boardmapR, boardunmapR)) {
      printf("cmn15_boardmakemap() Failed!\n");
      exit(1);
  }
  if (!cmn15_boardmap(boardRB, &boardRB, boardmapR)) {
      printf("cmn15_boardmap() Failed!\n");
      exit(1);
  }
  if (!cmn15_boardmap(boardRA, &boardRA, boardmapR)) {
      printf("cmn15_boardmap() Failed!\n");
      exit(1);
  }
  //printf("Equivalent to %016lx %016lx.\n", boardFA, boardFB);
  //printf("Equivalent to %016lx %016lx.\n", boardRA, boardRB);
  uint64_t solutionboardsF[128] = {0};
  uint64_t solutionboardsR[128] = {0};
  uint64_t *solutionboards;
  uint32_t minsteps;
  uint64_t hevalsF, hevalsR;
  _Bool reverse = false;
  if (pathleneven) {
    printf("All solutions paths must have an even number of moves.\n");
  } else {
    printf("All solutions paths must have an odd number of moves.\n");
  }
  //if (boardFB != boardRB) {
  //    printf("Insufficient RAM!\n");
  //    exit(1);
  //}
  cmn15_db7710_t db7710F;
  if(!cmn15_builddb7710(&db7710F, boardFB, argv[0])) {
    printf("Create 7-7-1+0 Pattern Database Failed!\n");
    exit(1);
  }
  cmn15_db7710_t db7710R;
  if (boardFB == boardRB) {
    db7710R.array[0] = db7710F.array[0];
    db7710R.array[1] = db7710F.array[1];
    db7710R.arraysize = db7710F.arraysize;
    db7710R.onedbx = db7710F.onedbx;
    db7710R.onedby = db7710F.onedby;
    db7710R.goalboard = boardRB;
  } else {
    if(!cmn15_builddb7710(&db7710R, boardRB, argv[0])) {
      printf("Create 7-7-1+0 Pattern Database Failed!\n");
      exit(1);
    }
  }
  uint64_t starttime = time(0);
  if(!cmn15_IDDLS(db7710F, boardFA, db7710R, boardRA, pathleneven, solutionboardsF, solutionboardsR, &minsteps, &hevalsF, &hevalsR)) {
    printf("Search Failed!\n");
    exit(1);
  }
  if (solutionboardsR[minsteps] == boardRB) {
    solutionboards = solutionboardsR;
    boardunmap = boardunmapR;
    boardtranslation = boardtranslationR;
    reverse = true;
  } else {
    solutionboards = solutionboardsF;
    boardunmap = boardunmapF;
    boardtranslation = boardtranslationF;
  }
  uint64_t endtime = time(0);
  if(!cmn15_printsoln(solutionboards, minsteps, boardunmap, boardtranslation, reverse)) {
    printf("Print Of Solution Path Failed!\n");
    exit(1);
  }
  //printf("%u steps. (%lu+%lu=%lu heuristic evals.)\n", minsteps, hevalsF, hevalsR, hevalsF+hevalsR);
  printf("%u steps. (%lu heuristic evals.)\n", minsteps, hevalsF+hevalsR);
  if (endtime > starttime) printf("Approx. %f heuristic evals./s\n", (float)(hevalsF+hevalsR)/(endtime - starttime));
  //boardA = cmn15_randomboard(boardB);
  //boardB = cmn15_randomboard(boardA);
  //pathleneven = cmn15_pathleneven(boardA, boardB);
  //countdown--;
  //}
}
// See table https://kociemba.org/themen/fifteen/fifteensolver.html
/* 
From 123456789abcdef0
  19 moves 123056789abcdef4
  38 moves 5248a03ed6bc1f97
  45 moves 587b16c290dae34f
  53 moves 617a05cdb3e8f294
  59 moves b5cef209d7613a48
  63 moves 9cfedba867543102
  65 moves 43218765cba90efd 
  72 moves 159d26ae37bf48c0 
  80 moves 0c9dfbae37254861 

From 0123456789abcdef
  67 moves c9fedba867543102  
  75 moves fedcba0876543219
  78 moves fedcba9876543210
  80 moves fe8cab9d26513740 
*/
/*
time ./fifteen9b.bin 159d26ae37bf48c0 123456789abcdef0
72 steps. (2601335425 heuristic evals.)
Approx. 1758847.625000 heuristic evals./s
real	24m40.032s (A10 2.5GHz laptop)

time ./fifteen9b.bin fe8cab9d26513740 0123456789abcdef
80 steps. (194131032 heuristic evals.)
Approx. 1848867.000000 heuristic evals./s
real	1m45.964s  (A10 2.5GHz laptop)


*/
