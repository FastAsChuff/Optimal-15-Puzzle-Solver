#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <immintrin.h> 

// gcc fifteen3.c -o fifteen3.bin -O3 -march=native -Wall 


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
  uint64_t mask;
  mask = (board & 0x5555555555555555ULL) | ((board & 0xaaaaaaaaaaaaaaaaULL) >> 1);
  mask = (mask & 0x3333333333333333ULL) | ((mask & 0xccccccccccccccccULL) >> 2);
  mask |= (mask << 1);
  mask |= (mask << 2);
  return __builtin_ctzll(~mask) / 4;
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
 

#ifndef __SSSE3__
uint32_t cmn15_heuristicLC(uint64_t boardA, uint64_t boardB) {
  uint32_t V = cmn15_linearconflicts(boardA, boardB);
  V = (V / 3) + (V % 3); 
  uint32_t H = cmn15_linearconflicts(cmn15_boardtranspose(boardA), cmn15_boardtranspose(boardB));
  H = (H / 3) + (H % 3); 
  return H+V; 
}

_Bool cmn15_IDDLSrh(uint64_t boardB, _Bool pathleneven, uint64_t *solutionboards, uint32_t depth, uint32_t maxdepth, uint8_t *Bx, uint8_t *By, uint64_t *hevals) {
  uint64_t board = solutionboards[depth];
  uint64_t adjacentboards[4];
  uint32_t h, hMD, hLC;
  cmn15_nextboard(board, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
  for (uint32_t i=0; i<4; i++) {
    if (adjacentboards[i] != 0) {
      if (adjacentboards[i] == boardB) {
        solutionboards[depth+1] = boardB;
        return true;        
      }
      if (!cmn15_insolutionboards(adjacentboards[i], solutionboards, depth)) {
        //pathleneven = cmn15_pathleneven(adjacentboards[i], boardB);
        hMD = cmn15_heuristicMD(adjacentboards[i], Bx, By);
        hLC = cmn15_heuristicLC(adjacentboards[i], boardB);
        h = (hMD < hLC ? hLC : hMD);
        (*hevals)++;
        if ((depth + h) <= maxdepth) {
          depth++;
          solutionboards[depth] = adjacentboards[i];
          if (depth < maxdepth) {
            if (cmn15_IDDLSrh(boardB, !pathleneven, solutionboards, depth, maxdepth, Bx, By, hevals)) return true;
          } 
          depth--;
        }
      }
    }
  }
  return false;
}

_Bool cmn15_IDDLS(uint64_t boardA, uint64_t boardB, _Bool pathleneven, uint64_t *solutionboards, uint32_t *minsteps, uint64_t *hevals) {
  *hevals = 0;
  solutionboards[0] = boardA;
  if (boardA == boardB) {
    *minsteps = 0;
    return true;
  }
  uint64_t up = 0, down = 0, left = 0, right = 0;
  if (!cmn15_nextboard(boardA, &up, &down, &left, &right)) return false;
  if ((up == boardB) || (down == boardB) || (left == boardB) || (right == boardB)) {
    solutionboards[1] = boardB;
    *minsteps = 1;
    return true;
  }
  uint32_t nibble;
  uint64_t board = boardB;
  uint8_t Bx[16], By[16];
  for (uint32_t i=0; i<16; i++) {
    nibble = board & 0xf;
    Bx[nibble] = i & 0x3;
    By[nibble] = i >> 2;
    board >>= 4;
  }
  uint32_t maxdepth = 2;
  if (!pathleneven) maxdepth = 3;
  printf("Completed search at depth ");
  while (!cmn15_IDDLSrh(boardB, !pathleneven, solutionboards, 0, maxdepth, Bx, By, hevals)) {
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

_Bool cmn15_IDDLSrh(uint64_t boardB, _Bool pathleneven, uint64_t *solutionboards, uint32_t depth, uint32_t maxdepth, uint8_t *Bx, uint8_t *By, uint64_t *hevals, uint32_t *nibble1pos, uint32_t *nibble2pos) {
  uint64_t board = solutionboards[depth];
  uint64_t adjacentboards[4];
  uint32_t h, hMD, hLC;
  cmn15_nextboard(board, &adjacentboards[0], &adjacentboards[1], &adjacentboards[2], &adjacentboards[3]);
  for (uint32_t i=0; i<4; i++) {
    if (adjacentboards[i] != 0) {
      if (adjacentboards[i] == boardB) {
        solutionboards[depth+1] = boardB;
        return true;        
      }
      if (!cmn15_insolutionboards(adjacentboards[i], solutionboards, depth)) {
        //pathleneven = cmn15_pathleneven(adjacentboards[i], boardB);
        hMD = cmn15_heuristicMD(adjacentboards[i], Bx, By);
        hLC = cmn15_heuristicLC(adjacentboards[i], nibble1pos, nibble2pos);
        h = (hMD < hLC ? hLC : hMD);
        (*hevals)++;
        if ((depth + h) <= maxdepth) {
          depth++;
          solutionboards[depth] = adjacentboards[i];
          if (depth < maxdepth) {
            if (cmn15_IDDLSrh(boardB, !pathleneven, solutionboards, depth, maxdepth, Bx, By, hevals, nibble1pos, nibble2pos)) return true;
          } 
          depth--;
        }
      }
    }
  }
  return false;
}

_Bool cmn15_IDDLS(uint64_t boardA, uint64_t boardB, _Bool pathleneven, uint64_t *solutionboards, uint32_t *minsteps, uint64_t *hevals) {
  *hevals = 0;
  solutionboards[0] = boardA;
  if (boardA == boardB) {
    *minsteps = 0;
    return true;
  }
  uint64_t up = 0, down = 0, left = 0, right = 0;
  if (!cmn15_nextboard(boardA, &up, &down, &left, &right)) return false;
  if ((up == boardB) || (down == boardB) || (left == boardB) || (right == boardB)) {
    solutionboards[1] = boardB;
    *minsteps = 1;
    return true;
  }
  uint32_t nibble;
  uint64_t board = boardB;
  uint8_t Bx[16], By[16];
  for (uint32_t i=0; i<16; i++) {
    nibble = board & 0xf;
    Bx[nibble] = i & 0x3;
    By[nibble] = i >> 2;
    board >>= 4;
  }
  uint32_t nibble1pos[16], nibble2pos[16];
  uint64_t board1nozero, board2nozero;
  cmn15_removezeronibbles(boardB, cmn15_boardtranspose(boardB), &board1nozero, &board2nozero);
  __m128i shuff = _mm_set_epi8(7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8);
  __m128i v1 = _mm_set_epi64x(board1nozero & 0x0f0f0f0f0f0f0f0fULL, (board1nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  __m128i v2 = _mm_set_epi64x(board2nozero & 0x0f0f0f0f0f0f0f0fULL, (board2nozero & 0x00f0f0f0f0f0f0f0ULL) >> 4); 
  v1 = _mm_shuffle_epi8(v1, shuff);
  v2 = _mm_shuffle_epi8(v2, shuff);
  __m128i vk;
  for(uint32_t k=1; k<16; k++) {
    vk = _mm_set1_epi8(k);
    nibble1pos[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v1)));
    nibble2pos[k] = __builtin_ctz(_mm_movemask_epi8(_mm_cmpeq_epi8(vk, v2)));
  }
  uint32_t maxdepth = 2;
  if (!pathleneven) maxdepth = 3;
  printf("Completed search at depth ");
  while (!cmn15_IDDLSrh(boardB, !pathleneven, solutionboards, 0, maxdepth, Bx, By, hevals, nibble1pos, nibble2pos)) {
    printf("%u ", maxdepth);
    fflush(stdout);
    maxdepth += 2;
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

_Bool cmn15_printsoln(uint64_t *solutionboards, uint32_t minsteps) {
  for (uint32_t i=0; i<=minsteps; i++) printf("%016lx \n", solutionboards[i]);
  //for (uint32_t i=0; i<=minsteps; i++) printf("%016lx %u %u %u\n", solutionboards[i], cmn15_linearconflicts(solutionboards[i], solutionboards[0]), cmn15_linearconflicts(cmn15_boardtranspose(solutionboards[i]), cmn15_boardtranspose(solutionboards[0])), cmn15_boardparity(solutionboards[i]));
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

int main(int argc, char*argv[]) {
  uint8_t boardarrayA[16] = {15,14,1,6,9,11,4,12,0,10,7,3,13,8,5,2}; 
  uint64_t boardA = cmn15_boardset(boardarrayA);
  uint8_t boardarrayB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0}; // 123456789abcdef0
  uint64_t boardB = cmn15_boardset(boardarrayB);
  if (argc == 1) {
    printf("This program find an optimally short path from two arbitrary board states of the famous 'fifteen' sliding square puzzle, if there is a solution, and a 'not reachable' message otherwise. The boards are input as arguments with hex values for the squares and the empty square is zero. The normally solved board state (endboard) is 123456789abcdef0. It uses an Iterative Deepening Depth Limited Search method using the maximum of Linear Conflicts and Manhatten Distance heuristics. Random board pairs will probably be solved in seconds, but there are some problematic state pairs which can not be solved in reasonable time with this algorithm/heuristic. This program's run-time data memory requirement is much less than 1MB.\n");
    printf("Copyright:- Simon Goater July 2024\n");
    printf("Usage:- %s startboard endboard\n", argv[0]);
    printf("e.g. %s fe169b4c0a73d852 123456789abcdef0\n", argv[0]);
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
  uint64_t solutionboards[128];
  uint32_t minsteps;
  uint64_t hevals;
  if (pathleneven) {
    printf("All solutions paths must have an even number of moves.\n");
  } else {
    printf("All solutions paths must have an odd number of moves.\n");
  }
  if(!cmn15_IDDLS(boardA, boardB, pathleneven, solutionboards, &minsteps, &hevals)) {
    printf("Search Failed!\n");
    exit(1);
  }
  if(!cmn15_printsoln(solutionboards, minsteps)) {
    printf("Print Of Solution Path Failed!\n");
    exit(1);
  }
  printf("%u steps. (%lu heuristic evals.)\n", minsteps, hevals);
}
// See table https://kociemba.org/themen/fifteen/fifteensolver.html
// 
/* 
From 123456789abcdef0
  19 moves 123056789abcdef4
  38 moves 5248a03ed6bc1f97
  45 moves 587b16c290dae34f
  53 moves 617a05cdb3e8f294
  59 moves b5cef209d7613a48
  63 moves 9cfedba867543102
  72 moves 159d26ae37bf48c0 (unsolved)
  73 moves 43218765cba90efd (unsolved)

From 0123456789abcdef
  67 moves c9fedba867543102  
  78 moves fedcba9876543210
  80 moves? fe8cab9d26513740 (unsolved)
*/
