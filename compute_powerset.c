// compute_powerset.c
//
// Computes distributions of (|A|, |A+A|, |A*A|) over all nonempty A ⊆ [n].
//
// Key speed/memory changes implemented:
//  (1) Counts table shrunk using the unconditional bound |A*A| ≤ C(|A|+1,2).
//      So for each k=|A| we only allocate mult ∈ [0..k(k+1)/2] instead of [0..n^2].
//  (2) Output is written in a compact binary format (native endianness) instead of CSV;
//      convert to CSV later with a separate pass.
//
// Other performance features kept:
//  - Gray-code walk with O(1) toggled-bit update: toggled bit at step t is CTZ(t).
//  - Incremental maintenance of add/mult cardinalities via ordered-pair reps.
//  - Concurrency capped to <= jobs processes (even if total_tasks = jobs*k is larger).
//  - Sparse “touched bins” list to avoid scanning the full table on output.
//
// Usage:
//   ./compute_powerset <n> <out_dir> <jobs> <k>
//
// Output files:
//   <out_dir>/pairs_<n>_<file_id:04d>.bin   where file_id = chunk_id + 1
//
// Binary format (native endianness):
//   Header:
//     char     magic[8]   = "SPP1BIN\0"
//     uint8_t  version    = 1
//     uint8_t  n
//     uint16_t max_sum    = 2n
//     uint32_t record_cnt = number of records
//   Records (record_cnt times):
//     uint8_t  set_cardinality  (k)
//     uint8_t  add_ds_card      (|A+A|)
//     uint16_t mult_ds_card     (|A*A|)
//     uint64_t count
//
// Notes:
// - Subsets represented as uint64 masks; requires 1 <= n <= 63.
// - Enumerates all nonempty subsets of [n].
// - For n up to ~51 this is still computationally enormous; your aim (“a few days”) is realistic
//   only with substantial parallel hardware and/or multiple machines.
//
// Compile (recommended):
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG -o compute_powerset compute_powerset.c

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
  #define POPCNT64(x) __builtin_popcountll((unsigned long long)(x))
  #define CTZ64(x)    __builtin_ctzll((unsigned long long)(x))
#else
static inline int POPCNT64(uint64_t x) {
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
}
static inline unsigned CTZ64(uint64_t x) {
    unsigned c = 0;
    while ((x & 1ULL) == 0ULL) { x >>= 1; c++; }
    return c;
}
#endif

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// mkdir -p style
static int mkdir_p(const char *path) {
    if (!path || !*path) return -1;

    char *tmp = strdup(path);
    if (!tmp) return -1;

    size_t len = strlen(tmp);
    if (len == 0) { free(tmp); return -1; }

    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0) {
                if (errno != EEXIST) { free(tmp); return -1; }
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0) {
        if (errno != EEXIST) { free(tmp); return -1; }
    }
    free(tmp);
    return 0;
}

/* ============================================================
   Gray code helpers
   ============================================================ */

static inline uint64_t gray64(uint64_t t) { return t ^ (t >> 1); }

/* ============================================================
   Incremental rep tracking
   ============================================================ */

typedef struct {
    uint8_t elems[63]; // values 1..63
    uint8_t pos[64];   // pos[x] valid when x in set
    int m;             // |A|
} SetList;

static inline void setlist_init(SetList *S) {
    S->m = 0;
    for (int i = 0; i < 64; i++) S->pos[i] = 0;
}

static inline void setlist_add(SetList *S, uint8_t x) {
    S->pos[x] = (uint8_t)S->m;
    S->elems[S->m++] = x;
}

static inline void setlist_remove(SetList *S, uint8_t x) {
    uint8_t idx = S->pos[x];
    int last_i = S->m - 1;
    uint8_t last = S->elems[last_i];
    S->elems[idx] = last;
    S->pos[last] = idx;
    S->m = last_i;
}

// rep updates; deltas are always 2 or 1
static inline void rep_inc2_u16(uint16_t *rep, int idx, int *card) {
    uint16_t before = rep[idx];
    rep[idx] = (uint16_t)(before + 2u);
    if (before == 0) (*card)++;
}
static inline void rep_dec2_u16(uint16_t *rep, int idx, int *card) {
    uint16_t after = (uint16_t)(rep[idx] - 2u);
    rep[idx] = after;
    if (after == 0) (*card)--;
}
static inline void rep_inc1_u16(uint16_t *rep, int idx, int *card) {
    uint16_t before = rep[idx];
    rep[idx] = (uint16_t)(before + 1u);
    if (before == 0) (*card)++;
}
static inline void rep_dec1_u16(uint16_t *rep, int idx, int *card) {
    uint16_t after = (uint16_t)(rep[idx] - 1u);
    rep[idx] = after;
    if (after == 0) (*card)--;
}

static inline void add_element(SetList *S, uint8_t x,
                               uint16_t * __restrict rep_sum,
                               uint16_t * __restrict rep_prod,
                               int *add_card, int *mult_card) {
    const int m = S->m;
    for (int i = 0; i < m; i++) {
        uint8_t y = S->elems[i];
        rep_inc2_u16(rep_sum,  (int)x + (int)y, add_card);
        rep_inc2_u16(rep_prod, (int)x * (int)y, mult_card);
    }
    rep_inc1_u16(rep_sum,  2 * (int)x, add_card);
    rep_inc1_u16(rep_prod, (int)x * (int)x, mult_card);
    setlist_add(S, x);
}

// remove (x,x), then remove x from set, then loop without a branch
static inline void remove_element(SetList *S, uint8_t x,
                                  uint16_t * __restrict rep_sum,
                                  uint16_t * __restrict rep_prod,
                                  int *add_card, int *mult_card) {
    rep_dec1_u16(rep_sum,  2 * (int)x, add_card);
    rep_dec1_u16(rep_prod, (int)x * (int)x, mult_card);

    setlist_remove(S, x);

    const int m = S->m;
    for (int i = 0; i < m; i++) {
        uint8_t y = S->elems[i];
        rep_dec2_u16(rep_sum,  (int)x + (int)y, add_card);
        rep_dec2_u16(rep_prod, (int)x * (int)y, mult_card);
    }
}

/* ============================================================
   Sparse touched-bins vector
   ============================================================ */

typedef struct {
    uint32_t *data;
    size_t len;
    size_t cap;
} U32Vec;

static inline int u32vec_init(U32Vec *v, size_t cap0) {
    v->len = 0;
    v->cap = (cap0 ? cap0 : (size_t)1);
    v->data = (uint32_t *)malloc(v->cap * sizeof(uint32_t));
    return (v->data != NULL) ? 0 : -1;
}

static inline int u32vec_push(U32Vec *v, uint32_t x) {
    if (v->len == v->cap) {
        size_t newcap = v->cap * 2;
        uint32_t *p = (uint32_t *)realloc(v->data, newcap * sizeof(uint32_t));
        if (!p) return -1;
        v->data = p;
        v->cap = newcap;
    }
    v->data[v->len++] = x;
    return 0;
}

static inline void u32vec_free(U32Vec *v) {
    free(v->data);
    v->data = NULL;
    v->len = v->cap = 0;
}

// Pack (k, add, mult) into 32 bits.
// k <= 63, add <= 126, mult <= 3969 (< 4096).
static inline uint32_t pack_key(uint8_t k, uint8_t add, uint16_t mult) {
    return ((uint32_t)k << 19) | ((uint32_t)add << 12) | (uint32_t)mult;
}
static inline void unpack_key(uint32_t key, uint8_t *k, uint8_t *add, uint16_t *mult) {
    *k    = (uint8_t)(key >> 19);
    *add  = (uint8_t)((key >> 12) & 0x7Fu);
    *mult = (uint16_t)(key & 0x0FFFu);
}

/* ============================================================
   Binary output structs (packed)
   ============================================================ */

#if defined(__GNUC__) || defined(__clang__)
  #define PACKED __attribute__((packed))
#else
  #define PACKED
#endif

typedef struct PACKED {
    char     magic[8];    // "SPP1BIN\0"
    uint8_t  version;     // 1
    uint8_t  n;           // n
    uint16_t max_sum;     // 2n
    uint32_t record_cnt;  // number of records
} BinHeader;

typedef struct PACKED {
    uint8_t  k;
    uint8_t  add;
    uint16_t mult;
    uint64_t count;
} BinRec;

/* ============================================================
   Per-chunk worker
   ============================================================ */

static int run_task(int chunk_id, int n, int total_tasks, const char *out_dir) {
    // Partition [0,2^n) into contiguous blocks.
    const __uint128_t TOTAL = (((__uint128_t)1) << n);
    const uint64_t start = (uint64_t)((TOTAL * (unsigned)chunk_id) / (unsigned)total_tasks);
    const uint64_t end   = (uint64_t)((TOTAL * (unsigned)(chunk_id + 1)) / (unsigned)total_tasks);

    const int max_sum  = 2 * n;
    const int max_prod = n * n;

    // Per-k mult caps: mult_cap[k] = k(k+1)/2.
    // Strides and offsets for the shrunk counts array:
    // counts stores, for each k, a (max_sum+1) x (mult_cap[k]+1) slab.
    uint16_t *stride_k = (uint16_t *)malloc((size_t)(n + 1) * sizeof(uint16_t));
    size_t   *offset_k = (size_t   *)malloc((size_t)(n + 1) * sizeof(size_t));
    if (!stride_k || !offset_k) {
        fprintf(stderr, "Task %d: failed to allocate stride/offset arrays\n", chunk_id);
        free(stride_k); free(offset_k);
        return 2;
    }

    size_t sum_strides = 0;
    for (int k = 0; k <= n; k++) {
        const int cap = (k * (k + 1)) / 2;  // <= max_prod always for k<=n
        stride_k[k] = (uint16_t)(cap + 1);
        offset_k[k] = (size_t)(max_sum + 1) * sum_strides;
        sum_strides += (size_t)stride_k[k];
    }

    const size_t total_bins = (size_t)(max_sum + 1) * sum_strides;

    uint64_t *counts = (uint64_t *)calloc(total_bins, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to allocate counts table (%zu bins)\n", chunk_id, total_bins);
        free(stride_k); free(offset_k);
        return 2;
    }

    U32Vec touched;
    // Preallocate moderately; will grow if needed.
    if (u32vec_init(&touched, (size_t)1 << 20) != 0) {
        fprintf(stderr, "Task %d: failed to allocate touched vector\n", chunk_id);
        free(counts);
        free(stride_k); free(offset_k);
        return 2;
    }

    // Representation counts for ordered pairs.
    uint16_t rep_sum[127]; // up to 2*63
    for (int i = 0; i <= max_sum; i++) rep_sum[i] = 0;

    // Products in [0..n^2]
    uint16_t rep_prod[max_prod + 1];
    memset(rep_prod, 0, (size_t)(max_prod + 1) * sizeof(uint16_t));

    SetList S;
    setlist_init(&S);
    int add_card = 0;
    int mult_card = 0;

    // Initialize A = Gray(start) from empty by adding its elements.
    uint64_t g = gray64(start);
    uint64_t mm = g;
    while (mm) {
        unsigned bit = (unsigned)CTZ64(mm);
        uint8_t x = (uint8_t)(bit + 1);
        mm &= (mm - 1);
        add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
    }

    // Record start (skip empty set)
    if (g != 0) {
        const uint8_t k = (uint8_t)S.m;

        // Safety: mult_card must be <= k(k+1)/2
        const int cap = ((int)k * ((int)k + 1)) / 2;
        if (mult_card > cap) {
            fprintf(stderr, "Task %d: invariant violated at init: mult_card=%d > C(k+1,2)=%d\n",
                    chunk_id, mult_card, cap);
            u32vec_free(&touched);
            free(counts);
            free(stride_k); free(offset_k);
            return 2;
        }

        const size_t idx = offset_k[k] + (size_t)add_card * (size_t)stride_k[k] + (size_t)mult_card;
        uint64_t *c = &counts[idx];
        if ((*c)++ == 0) {
            const uint32_t key = pack_key(k, (uint8_t)add_card, (uint16_t)mult_card);
            if (u32vec_push(&touched, key) != 0) {
                fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                u32vec_free(&touched);
                free(counts);
                free(stride_k); free(offset_k);
                return 2;
            }
        }
    }

    // Walk forward in Gray order over [start+1, end).
    // For reflected Gray code g(t)=t^(t>>1), the toggled bit from t-1->t is CTZ(t).
    for (uint64_t t = start + 1; t < end; t++) {
        unsigned bit = (unsigned)CTZ64(t);
        uint64_t bmask = 1ULL << bit;

        g ^= bmask;
        uint8_t x = (uint8_t)(bit + 1);

        if (g & bmask) {
            add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        } else {
            remove_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        }

        if (g != 0) {
            const uint8_t k = (uint8_t)S.m;
            const int cap = ((int)k * ((int)k + 1)) / 2;
            if (mult_card > cap) {
                fprintf(stderr, "Task %d: invariant violated: mult_card=%d > C(k+1,2)=%d\n",
                        chunk_id, mult_card, cap);
                u32vec_free(&touched);
                free(counts);
                free(stride_k); free(offset_k);
                return 2;
            }

            const size_t idx = offset_k[k] + (size_t)add_card * (size_t)stride_k[k] + (size_t)mult_card;
            uint64_t *c = &counts[idx];
            if ((*c)++ == 0) {
                const uint32_t key = pack_key(k, (uint8_t)add_card, (uint16_t)mult_card);
                if (u32vec_push(&touched, key) != 0) {
                    fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                    u32vec_free(&touched);
                    free(counts);
                    free(stride_k); free(offset_k);
                    return 2;
                }
            }
        }
    }

    // Write binary output
    const int file_id = chunk_id + 1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/pairs_%d_%04d.bin", out_dir, n, file_id);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Task %d: failed to open output file %s: %s\n", chunk_id, path, strerror(errno));
        u32vec_free(&touched);
        free(counts);
        free(stride_k); free(offset_k);
        return 3;
    }

    BinHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "SPP1BIN\0", 8);
    hdr.version = 1;
    hdr.n = (uint8_t)n;
    hdr.max_sum = (uint16_t)max_sum;
    hdr.record_cnt = (uint32_t)touched.len;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Task %d: failed to write header to %s\n", chunk_id, path);
        fclose(f);
        u32vec_free(&touched);
        free(counts);
        free(stride_k); free(offset_k);
        return 3;
    }

    for (size_t i = 0; i < touched.len; i++) {
        uint8_t k, a;
        uint16_t m;
        unpack_key(touched.data[i], &k, &a, &m);

        // Lookup count
        const size_t idx = offset_k[k] + (size_t)a * (size_t)stride_k[k] + (size_t)m;
        const uint64_t c = counts[idx];

        BinRec rec;
        rec.k = k;
        rec.add = a;
        rec.mult = m;
        rec.count = c;

        if (fwrite(&rec, sizeof(rec), 1, f) != 1) {
            fprintf(stderr, "Task %d: failed to write record to %s\n", chunk_id, path);
            fclose(f);
            u32vec_free(&touched);
            free(counts);
            free(stride_k); free(offset_k);
            return 3;
        }
    }

    fclose(f);

    u32vec_free(&touched);
    free(counts);
    free(stride_k);
    free(offset_k);
    return 0;
}

/* ============================================================
   Main: spawn at most "jobs" children concurrently
   ============================================================ */

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <n> <out_dir> <jobs> <k>\n", argv[0]);
        return 1;
    }

    const int n = atoi(argv[1]);
    const char *out_dir = argv[2];
    const int jobs = atoi(argv[3]);
    const int k = atoi(argv[4]);

    if (n < 1 || n > 63) {
        fprintf(stderr, "Error: n must be in [1,63]\n");
        return 1;
    }
    if (jobs < 1) {
        fprintf(stderr, "Error: jobs must be >= 1\n");
        return 1;
    }
    if (k < 1) {
        fprintf(stderr, "Error: k must be >= 1\n");
        return 1;
    }

    if (mkdir_p(out_dir) != 0) {
        fprintf(stderr, "Error: could not create out_dir '%s'\n", out_dir);
        return 1;
    }

    const int total_tasks = jobs * k;
    const double t0 = now_seconds();

    pid_t *pids = (pid_t *)calloc((size_t)total_tasks, sizeof(pid_t));
    if (!pids) {
        fprintf(stderr, "Error: allocation failure\n");
        return 1;
    }

    int next_chunk = 0;
    int active = 0;
    int done = 0;

    while (done < total_tasks) {
        while (active < jobs && next_chunk < total_tasks) {
            const int chunk_id = next_chunk;

            pid_t pid = fork();
            if (pid < 0) {
                fprintf(stderr, "Error: fork failed at chunk %d: %s\n", chunk_id, strerror(errno));
                next_chunk = total_tasks; // stop launching; wait for the active ones
                break;
            }
            if (pid == 0) {
                int rc = run_task(chunk_id, n, total_tasks, out_dir);
                _exit(rc);
            }

            pids[chunk_id] = pid;
            next_chunk++;
            active++;
        }

        int status = 0;
        pid_t pid = wait(&status);
        if (pid < 0) {
            if (errno == EINTR) continue;
            break;
        }

        active--;
        done++;

        int chunk_id = -1;
        for (int i = 0; i < total_tasks; i++) {
            if (pids[i] == pid) { chunk_id = i; break; }
        }

        int exit_code = 0;
        if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
        else exit_code = 128;

        const int file_id = (chunk_id >= 0) ? (chunk_id + 1) : 0;
        char path[4096];
        snprintf(path, sizeof(path), "%s/pairs_%d_%04d.bin", out_dir, n, file_id);

        double elapsed = now_seconds() - t0;
        int pct = (100 * done) / total_tasks;

        if (exit_code == 0) {
            printf("%d%% done, wrote %s, %.1fs since start\n", pct, path, elapsed);
        } else {
            printf("%d%% done, task %d failed (exit=%d), intended %s, %.1fs since start\n",
                   pct, chunk_id, exit_code, path, elapsed);
        }
        fflush(stdout);
    }

    free(pids);
    return 0;
}

