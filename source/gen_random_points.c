// gen_random_points_counts.c
//
// Generate random sets A of fixed size n with elements sampled uniformly
// without replacement from [1, 2^b], compute (|A+A|, |A*A|), and record
// points with COUNTS.
//
// Output CSV does NOT store b (you encode b in filename).
// For b=64, sample uniformly from [1, 2^64-1] (no overflow/zero).
//
// Usage:
//   ./gen_random_points_counts <n> <num_sets> <b> <jobs> <out_csv>
//
// Example:
//   ./gen_random_points_counts 7 1000000000 64 40 data/random/random_7_64.csv
//
// Compile:
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG \
//     -o gen_random_points_counts gen_random_points_counts.c

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ============================================================
   PRNG: xoshiro256**
   ============================================================ */

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t x = seed;
    r->s[0] = splitmix64_next(&x);
    r->s[1] = splitmix64_next(&x);
    r->s[2] = splitmix64_next(&x);
    r->s[3] = splitmix64_next(&x);
}

static inline uint64_t rng_next_u64(rng_t *r) {
    const uint64_t result = rotl64(r->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = r->s[1] << 17;

    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];

    r->s[2] ^= t;
    r->s[3] = rotl64(r->s[3], 45);
    return result;
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint64_t rng_u64_bounded(rng_t *r, uint64_t bound) {
    uint64_t x, m = (uint64_t)(-bound) % bound;
    do { x = rng_next_u64(r); } while (x < m);
    return x % bound;
}

/* ============================================================
   Hash sets for uint128 keys (open addressing)
   ============================================================ */

static inline uint64_t hash_u128(__uint128_t v) {
    uint64_t lo = (uint64_t)v;
    uint64_t hi = (uint64_t)(v >> 64);
    return mix64(lo ^ mix64(hi + 0x9e3779b97f4a7c15ULL));
}

typedef struct {
    __uint128_t *keys;
    uint8_t     *used;
    size_t cap;     // power of two
    size_t size;
} set128_t;

static int set128_init(set128_t *S, size_t cap_pow2) {
    S->cap = cap_pow2;
    S->size = 0;
    S->keys = (__uint128_t *)malloc(S->cap * sizeof(__uint128_t));
    S->used = (uint8_t *)calloc(S->cap, 1);
    return (S->keys && S->used) ? 0 : -1;
}

static void set128_free(set128_t *S) {
    free(S->keys); free(S->used);
    S->keys = NULL; S->used = NULL; S->cap = S->size = 0;
}

static int set128_rehash(set128_t *S, size_t newcap) {
    set128_t T;
    if (set128_init(&T, newcap) != 0) return -1;

    for (size_t i = 0; i < S->cap; i++) {
        if (!S->used[i]) continue;
        __uint128_t key = S->keys[i];
        uint64_t h = hash_u128(key);
        size_t mask = T.cap - 1;
        size_t j = (size_t)h & mask;
        while (T.used[j]) j = (j + 1) & mask;
        T.used[j] = 1;
        T.keys[j] = key;
        T.size++;
    }

    set128_free(S);
    *S = T;
    return 0;
}

static int set128_insert(set128_t *S, __uint128_t key) {
    if ((S->size * 10) >= (S->cap * 7)) {
        if (set128_rehash(S, S->cap * 2) != 0) return -1;
    }
    uint64_t h = hash_u128(key);
    size_t mask = S->cap - 1;
    size_t i = (size_t)h & mask;

    while (S->used[i]) {
        if (S->keys[i] == key) return 0;
        i = (i + 1) & mask;
    }
    S->used[i] = 1;
    S->keys[i] = key;
    S->size++;
    return 1;
}

/* ============================================================
   Sample n distinct values from [1, 2^b] (b=64 => [1,2^64-1])
   ============================================================ */

static int sample_distinct(uint64_t *A, int n, int b, rng_t *rng) {
    set128_t chosen;
    size_t cap = 1;
    while (cap < (size_t)(n * 4)) cap <<= 1;
    if (cap < 64) cap = 64;
    if (set128_init(&chosen, cap) != 0) return -1;

    int m = 0;
    if (b == 64) {
        while (m < n) {
            uint64_t x = rng_next_u64(rng);
            if (x == 0) continue; // ensure in [1,2^64-1]
            int ins = set128_insert(&chosen, (__uint128_t)x);
            if (ins < 0) { set128_free(&chosen); return -1; }
            if (ins == 1) A[m++] = x;
        }
    } else {
        uint64_t bound = (1ULL << b);
        while (m < n) {
            uint64_t x = rng_u64_bounded(rng, bound) + 1ULL; // 1..2^b
            int ins = set128_insert(&chosen, (__uint128_t)x);
            if (ins < 0) { set128_free(&chosen); return -1; }
            if (ins == 1) A[m++] = x;
        }
    }

    set128_free(&chosen);
    return 0;
}

/* ============================================================
   Compute |A+A| and |A*A| (unordered pairs i<=j)
   ============================================================ */

static int compute_add_mult_cards_u64(const uint64_t *A, int n, uint16_t *out_add, uint16_t *out_mult) {
    size_t pairs = (size_t)n * (size_t)(n + 1) / 2;

    set128_t sums, prods;
    size_t cap = 1;
    while (cap < pairs * 2) cap <<= 1;
    if (cap < 1024) cap = 1024;

    if (set128_init(&sums, cap) != 0) return -1;
    if (set128_init(&prods, cap) != 0) { set128_free(&sums); return -1; }

    for (int i = 0; i < n; i++) {
        __uint128_t ai = (__uint128_t)A[i];
        for (int j = i; j < n; j++) {
            __uint128_t aj = (__uint128_t)A[j];
            __uint128_t s = ai + aj;
            __uint128_t p = ai * aj;
            if (set128_insert(&sums, s) < 0) { set128_free(&sums); set128_free(&prods); return -1; }
            if (set128_insert(&prods, p) < 0) { set128_free(&sums); set128_free(&prods); return -1; }
        }
    }

    *out_add  = (uint16_t)sums.size;
    *out_mult = (uint16_t)prods.size;

    set128_free(&sums);
    set128_free(&prods);
    return 0;
}

/* ============================================================
   Point->count hash map (open addressing)
   ============================================================ */

typedef struct {
    uint32_t *keys;
    uint64_t *vals;
    uint8_t  *used;
    size_t cap;
    size_t size;
} map32_u64_t;

static inline uint32_t hash32(uint32_t x) { return (uint32_t)mix64((uint64_t)x); }

static int map_init(map32_u64_t *M, size_t cap_pow2) {
    M->cap = cap_pow2;
    M->size = 0;
    M->keys = (uint32_t *)malloc(M->cap * sizeof(uint32_t));
    M->vals = (uint64_t *)malloc(M->cap * sizeof(uint64_t));
    M->used = (uint8_t  *)calloc(M->cap, 1);
    return (M->keys && M->vals && M->used) ? 0 : -1;
}

static void map_free(map32_u64_t *M) {
    free(M->keys); free(M->vals); free(M->used);
    M->keys = NULL; M->vals = NULL; M->used = NULL;
    M->cap = M->size = 0;
}

static int map_rehash(map32_u64_t *M, size_t newcap) {
    map32_u64_t T;
    if (map_init(&T, newcap) != 0) return -1;

    for (size_t i = 0; i < M->cap; i++) {
        if (!M->used[i]) continue;
        uint32_t key = M->keys[i];
        uint64_t val = M->vals[i];

        size_t mask = T.cap - 1;
        size_t j = (size_t)hash32(key) & mask;
        while (T.used[j]) j = (j + 1) & mask;
        T.used[j] = 1;
        T.keys[j] = key;
        T.vals[j] = val;
        T.size++;
    }

    map_free(M);
    *M = T;
    return 0;
}

// increment count for key by 1
static int map_inc(map32_u64_t *M, uint32_t key) {
    if ((M->size * 10) >= (M->cap * 7)) {
        if (map_rehash(M, M->cap * 2) != 0) return -1;
    }
    size_t mask = M->cap - 1;
    size_t i = (size_t)hash32(key) & mask;
    while (M->used[i]) {
        if (M->keys[i] == key) { M->vals[i] += 1ULL; return 0; }
        i = (i + 1) & mask;
    }
    M->used[i] = 1;
    M->keys[i] = key;
    M->vals[i] = 1ULL;
    M->size++;
    return 0;
}

static inline uint32_t pack_point(uint16_t add, uint16_t mult) {
    return ((uint32_t)add << 16) | (uint32_t)mult;
}
static inline void unpack_point(uint32_t key, uint16_t *add, uint16_t *mult) {
    *add = (uint16_t)(key >> 16);
    *mult = (uint16_t)(key & 0xFFFFu);
}

/* ============================================================
   Worker / merge IO
   ============================================================ */

typedef struct __attribute__((packed)) {
    char magic[8];      // "RSPC1\0\0"
    uint32_t count;     // number of distinct points
} MapHdr;

static int worker_run(int wid, int n, uint64_t num_sets, int b, int jobs, const char *tmp_prefix) {
    uint64_t start = (num_sets * (uint64_t)wid) / (uint64_t)jobs;
    uint64_t end   = (num_sets * (uint64_t)(wid + 1)) / (uint64_t)jobs;
    uint64_t m = end - start;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t seed = (uint64_t)getpid()
                  ^ mix64((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * 1000000007ULL)
                  ^ (uint64_t)wid;
    rng_t rng;
    rng_seed(&rng, seed);

    map32_u64_t mp;
    if (map_init(&mp, 1u << 18) != 0) {
        fprintf(stderr, "Worker %d: failed to alloc map\n", wid);
        return 2;
    }

    uint64_t *A = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    if (!A) { map_free(&mp); return 2; }

    const double t0 = now_seconds();
    uint64_t next_report = 1000000ULL;

    for (uint64_t t = 0; t < m; t++) {
        if (sample_distinct(A, n, b, &rng) != 0) {
            fprintf(stderr, "Worker %d: sample_distinct failed\n", wid);
            free(A); map_free(&mp);
            return 2;
        }

        uint16_t addc=0, multc=0;
        if (compute_add_mult_cards_u64(A, n, &addc, &multc) != 0) {
            fprintf(stderr, "Worker %d: compute failed\n", wid);
            free(A); map_free(&mp);
            return 2;
        }

        if (map_inc(&mp, pack_point(addc, multc)) != 0) {
            fprintf(stderr, "Worker %d: map_inc failed\n", wid);
            free(A); map_free(&mp);
            return 2;
        }

        if (t + 1 == next_report) {
            double dt = now_seconds() - t0;
            fprintf(stderr, "Worker %d: %" PRIu64 "/%" PRIu64 " sets (%.1f s), distinct pts=%zu\n",
                    wid, t + 1, m, dt, mp.size);
            next_report *= 2;
        }
    }

    free(A);

    char path[4096];
    snprintf(path, sizeof(path), "%s_%04d.bin", tmp_prefix, wid);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Worker %d: cannot open %s: %s\n", wid, path, strerror(errno));
        map_free(&mp);
        return 3;
    }

    MapHdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "RSPC1\0\0", 8);
    hdr.count = (uint32_t)mp.size;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Worker %d: header write failed\n", wid);
        fclose(f);
        map_free(&mp);
        return 3;
    }

    for (size_t i = 0; i < mp.cap; i++) {
        if (!mp.used[i]) continue;
        uint32_t key = mp.keys[i];
        uint64_t val = mp.vals[i];
        if (fwrite(&key, sizeof(uint32_t), 1, f) != 1 ||
            fwrite(&val, sizeof(uint64_t), 1, f) != 1) {
            fprintf(stderr, "Worker %d: map entry write failed\n", wid);
            fclose(f);
            map_free(&mp);
            return 3;
        }
    }

    fclose(f);
    map_free(&mp);
    return 0;
}

static int merge_and_write_csv(int n, int jobs, const char *tmp_prefix, const char *out_csv) {
    map32_u64_t all;
    if (map_init(&all, 1u << 20) != 0) {
        fprintf(stderr, "Merge: failed to alloc global map\n");
        return 2;
    }

    for (int wid = 0; wid < jobs; wid++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s_%04d.bin", tmp_prefix, wid);

        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "Merge: missing %s\n", path);
            map_free(&all);
            return 3;
        }

        MapHdr hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1 || memcmp(hdr.magic, "RSPC1\0\0", 8) != 0) {
            fprintf(stderr, "Merge: bad header %s\n", path);
            fclose(f);
            map_free(&all);
            return 3;
        }

        for (uint32_t i = 0; i < hdr.count; i++) {
            uint32_t key;
            uint64_t val;
            if (fread(&key, sizeof(uint32_t), 1, f) != 1 ||
                fread(&val, sizeof(uint64_t), 1, f) != 1) {
                fprintf(stderr, "Merge: short read %s\n", path);
                fclose(f);
                map_free(&all);
                return 3;
            }

            // add val into global map:
            // do a find/insert loop (inlined-ish)
            if ((all.size * 10) >= (all.cap * 7)) {
                if (map_rehash(&all, all.cap * 2) != 0) {
                    fclose(f);
                    map_free(&all);
                    return 2;
                }
            }
            size_t mask = all.cap - 1;
            size_t j = (size_t)hash32(key) & mask;
            while (all.used[j]) {
                if (all.keys[j] == key) { all.vals[j] += val; goto merged; }
                j = (j + 1) & mask;
            }
            all.used[j] = 1;
            all.keys[j] = key;
            all.vals[j] = val;
            all.size++;
        merged:
            (void)0;
        }

        fclose(f);
        unlink(path);
    }

    FILE *csv = fopen(out_csv, "w");
    if (!csv) {
        fprintf(stderr, "Cannot open %s: %s\n", out_csv, strerror(errno));
        map_free(&all);
        return 3;
    }

    fprintf(csv, "set_cardinality,add_ds_card,mult_ds_card,count\n");
    for (size_t i = 0; i < all.cap; i++) {
        if (!all.used[i]) continue;
        uint16_t add, mult;
        unpack_point(all.keys[i], &add, &mult);
        fprintf(csv, "%d,%u,%u,%" PRIu64 "\n", n, (unsigned)add, (unsigned)mult, all.vals[i]);
    }

    fclose(csv);
    map_free(&all);
    return 0;
}

/* ============================================================
   Main
   ============================================================ */

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <n> <num_sets> <b> <jobs> <out_csv>\n", argv[0]);
        return 1;
    }

    const int n = atoi(argv[1]);
    const uint64_t num_sets = (uint64_t)strtoull(argv[2], NULL, 10);
    const int b = atoi(argv[3]);
    const int jobs = atoi(argv[4]);
    const char *out_csv = argv[5];

    if (n < 1 || n > 300) {
        fprintf(stderr, "Error: n must be in [1,300]\n");
        return 1;
    }
    if (b < 1 || b > 64) {
        fprintf(stderr, "Error: b must be in [1,64]\n");
        return 1;
    }
    if (jobs < 1) {
        fprintf(stderr, "Error: jobs must be >= 1\n");
        return 1;
    }

    char tmp_prefix[4096];
    snprintf(tmp_prefix, sizeof(tmp_prefix), "tmp_rspc_p%d_n%d_b%d", (int)getpid(), n, b);

    pid_t *pids = (pid_t *)calloc((size_t)jobs, sizeof(pid_t));
    if (!pids) { fprintf(stderr, "alloc failed\n"); return 1; }

    double t0 = now_seconds();

    for (int wid = 0; wid < jobs; wid++) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed: %s\n", strerror(errno));
            free(pids);
            return 1;
        }
        if (pid == 0) {
            int rc = worker_run(wid, n, num_sets, b, jobs, tmp_prefix);
            _exit(rc);
        }
        pids[wid] = pid;
    }

    int bad = 0;
    for (int i = 0; i < jobs; i++) {
        int st = 0;
        if (waitpid(pids[i], &st, 0) < 0) { bad = 1; continue; }
        if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) bad = 1;
    }
    free(pids);

    if (bad) {
        fprintf(stderr, "Some workers failed; not merging.\n");
        return 2;
    }

    int mrc = merge_and_write_csv(n, jobs, tmp_prefix, out_csv);
    if (mrc != 0) return mrc;

    fprintf(stderr, "Wrote %s (elapsed %.1fs)\n", out_csv, now_seconds() - t0);
    return 0;
}

