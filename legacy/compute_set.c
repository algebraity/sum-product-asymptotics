// sumprod_mask.c
// Build as a shared library and call from Python via ctypes.
//
// Computes |A+A| and |A*A| for A ⊆ [n] represented by a bitmask.
// Assumes n <= 63 (so mask fits in uint64_t). For powersets, that is already the practical regime.

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
  #define POPCNT64(x) __builtin_popcountll((unsigned long long)(x))
#else
static inline int POPCNT64(uint64_t x) {
    // portable popcount
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
}
#endif

// Export-friendly signature for ctypes.
// Returns 0 on success, nonzero on invalid input.
int sp_compute_from_mask(uint64_t mask, int n, int *out_add_card, int *out_mult_card) {
    if (!out_add_card || !out_mult_card) return 2;
    if (n < 1 || n > 63) return 3;
    if (mask == 0) { *out_add_card = 0; *out_mult_card = 0; return 0; }

    // Collect elements of A into a small array a[0..m-1].
    // Elements are in [1..n].
    uint32_t a[63];
    int m = 0;
    for (int i = 0; i < n; i++) {
        if ((mask >> i) & 1ULL) a[m++] = (uint32_t)(i + 1);
    }

    // Bitset for sums in [2..2n]. We'll index directly by value.
    // Need bits 0..2n inclusive => (2n+1) bits.
    const int max_sum = 2 * n;
    const int sum_words = (max_sum + 64) / 64; // enough for bit max_sum
    uint64_t sum_bits[3]; // for n<=63, max_sum<=126 => sum_words<=2, but keep 3 for safety
    for (int i = 0; i < sum_words; i++) sum_bits[i] = 0ULL;

    // Bitset for products in [1..n^2].
    const int max_prod = n * n;
    const int prod_words = (max_prod + 64) / 64; // enough for bit max_prod
    // For n<=63, max_prod<=3969 => prod_words<=63
    uint64_t prod_bits[63];
    for (int i = 0; i < prod_words; i++) prod_bits[i] = 0ULL;

    // Mark all pairwise sums/products (ordered pairs, but sets remove duplicates).
    for (int i = 0; i < m; i++) {
        const uint32_t ai = a[i];
        for (int j = 0; j < m; j++) {
            const uint32_t aj = a[j];

            const uint32_t s = ai + aj;      // in [2..2n]
            const uint32_t p = ai * aj;      // in [1..n^2]

            sum_bits[s >> 6]  |= 1ULL << (s & 63);
            prod_bits[p >> 6] |= 1ULL << (p & 63);
        }
    }

    // Count distinct sums/products via popcount of the bitsets.
    int add_card = 0;
    for (int i = 0; i < sum_words; i++) add_card += POPCNT64(sum_bits[i]);

    int mult_card = 0;
    for (int i = 0; i < prod_words; i++) mult_card += POPCNT64(prod_bits[i]);

    *out_add_card = add_card;
    *out_mult_card = mult_card;
    return 0;
}

