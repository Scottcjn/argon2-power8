/*
 * Argon2 reference source code package - POWER8/VSX optimized implementation
 * Ported by Scott (Scottcjn) - 2025
 *
 * Based on the reference C implementation
 * Copyright 2015 Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, Samuel Neves
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "argon2.h"
#include "core.h"

#include "blake2/blake2.h"

#if defined(__VSX__) || defined(__ALTIVEC__)

#include "blake2/blamka-round-vsx.h"
#include <altivec.h>

/* Use our typedef from the header */
typedef vector unsigned long long vsx_block_t;

#define ARGON2_VSX_OWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 16)

/*
 * Function fills a new memory block and optionally XORs the old block over the new one.
 */
static void fill_block(vsx_block_t *state, const block *ref_block,
                       block *next_block, int with_xor) {
    vsx_block_t block_XY[ARGON2_VSX_OWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_VSX_OWORDS_IN_BLOCK; i++) {
            state[i] = vec_xor(
                state[i], VSX_LOADU((const uint64_t *)ref_block->v + i*2));
            block_XY[i] = vec_xor(
                state[i], VSX_LOADU((const uint64_t *)next_block->v + i*2));
        }
    } else {
        for (i = 0; i < ARGON2_VSX_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = vec_xor(
                state[i], VSX_LOADU((const uint64_t *)ref_block->v + i*2));
        }
    }

    /* 8 row rounds */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_VSX(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
            state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    /* 8 column rounds */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_VSX(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
            state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    /* XOR and store */
    for (i = 0; i < ARGON2_VSX_OWORDS_IN_BLOCK; i++) {
        state[i] = vec_xor(state[i], block_XY[i]);
        VSX_STOREU((uint64_t *)next_block->v + i*2, state[i]);
    }
}

static void next_addresses(block *address_block, block *input_block) {
    vsx_block_t zero_block[ARGON2_VSX_OWORDS_IN_BLOCK];
    vsx_block_t zero2_block[ARGON2_VSX_OWORDS_IN_BLOCK];

    memset(zero_block, 0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));

    input_block->v[6]++;

    fill_block(zero_block, input_block, address_block, 0);
    fill_block(zero2_block, address_block, address_block, 0);
}

void fill_segment(const argon2_instance_t *instance,
                  argon2_position_t position) {
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    vsx_block_t state[ARGON2_VSX_OWORDS_IN_BLOCK];
    int data_independent_addressing;

    if (instance == NULL) {
        return;
    }

    data_independent_addressing =
        (instance->type == Argon2_i) ||
        (instance->type == Argon2_id && (position.pass == 0) &&
         (position.slice < ARGON2_SYNC_POINTS / 2));

    if (data_independent_addressing) {
        init_block_value(&input_block, 0);

        input_block.v[0] = position.pass;
        input_block.v[1] = position.lane;
        input_block.v[2] = position.slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = instance->type;
    }

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2;

        if (data_independent_addressing) {
            next_addresses(&address_block, &input_block);
        }
    }

    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        prev_offset = curr_offset - 1;
    }

    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {
        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        if (data_independent_addressing) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses(&address_block, &input_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
        } else {
            pseudo_rand = instance->memory[prev_offset].v[0];
        }

        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

        if ((position.pass == 0) && (position.slice == 0)) {
            ref_lane = position.lane;
        }

        position.index = i;
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

        ref_block =
            instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;
        if (ARGON2_VERSION_10 == instance->version) {
            fill_block(state, ref_block, curr_block, 0);
        } else {
            if(0 == position.pass) {
                fill_block(state, ref_block, curr_block, 0);
            } else {
                fill_block(state, ref_block, curr_block, 1);
            }
        }
    }
}

#else
#error "This file requires VSX or AltiVec support. Use opt.c for x86 or ref.c for scalar."
#endif /* __VSX__ || __ALTIVEC__ */
