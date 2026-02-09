// compute_powerset.c
//
// C port of the Python multiprocess structure, but writes compact distributions.
//
// MODS:
//  1) Store counts keyed by (set_cardinality=|A|, add_ds_card=|A+A|, mult_ds_card=|A*A|).
//  2) Speed-ups without compromising accuracy:
//     - Limit concurrency to <= jobs processes at a time (prevents oversubscription when jobs*k >> jobs).
//     - Faster Gray walk update: use the known property that the toggled Gray bit at step t is CTZ(t).
//       (avoids recomputing gray(t) and diff each iteration).
//     - Faster removal update: remove x from the list first to avoid "if (y==x)" branch in the loop.
//     - Avoid unnecessary membership bookkeeping; keep only elems[] and pos[].
//     - Use 64-bit counters (counts can exceed 2^32).
//     - Keep a sparse "touched bins" list to avoid scanning the entire 3D table on output.
//
// Usage:
//   ./compute_powerset <n> <out_dir> <jobs> <k>
//
// Output files:
//   <out_dir>/pairs_<n>_<file_id:04d>.csv   where file_id = chunk_id + 1
//
// Output CSV columns:
//   set_cardinality,add_ds_card,mult_ds_card,count
//
// Notes:
// - Subsets are represented as uint64 bitmasks, so requires 1 <= n <= 63.
// - Enumerates all nonempty subsets of [n] (skips A=∅).
// - total_tasks = jobs*k chunks, but runs at most "jobs" chunks concurrently.
//
// Compile (recommended):
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -o compute_powerset compute_powerset.c

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

// Maintain a dynamic list of elements in A for O(|A|) iteration and O(1) delete.
typedef struct {
    uint8_t elems[63]; // values 1..63
    uint8_t pos[64];   // pos[x] valid when x in set
    int m;             // |A|
} SetList;

static inline void setlist_init(SetList *S) {
    S->m = 0;
    // pos[] only needs to be correct for present elements; still initialize for safety.
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

// Specialized rep updates (delta is always 2 or 1).
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

// Ordered-pair reps: when adding x, for each y in A(before):
// (x,y) and (y,x): +2; plus (x,x): +1.
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

// Faster remove: remove x from S first so we can loop without a branch.
static inline void remove_element(SetList *S, uint8_t x,
                                  uint16_t * __restrict rep_sum,
                                  uint16_t * __restrict rep_prod,
                                  int *add_card, int *mult_card) {
    // Remove (x,x)
    rep_dec1_u16(rep_sum,  2 * (int)x, add_card);
    rep_dec1_u16(rep_prod, (int)x * (int)x, mult_card);

    // Remove x from the list (now S contains A \ {x})
    setlist_remove(S, x);

    // Remove (x,y) and (y,x) for all remaining y
    const int m = S->m;
    for (int i = 0; i < m; i++) {
        uint8_t y = S->elems[i];
        rep_dec2_u16(rep_sum,  (int)x + (int)y, add_card);
        rep_dec2_u16(rep_prod, (int)x * (int)y, mult_card);
    }
}

/* ============================================================
   Sparse "touched bins" vector to speed output
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

/* ============================================================
   Per-chunk worker
   ============================================================ */

static int run_task(int chunk_id, int n, int total_tasks, const char *out_dir) {
    // total = 2^n (n<=63). Use 128-bit for partition arithmetic to avoid overflow.
    const __uint128_t TOTAL = (((__uint128_t)1) << n);

    const int max_sum  = 2 * n;     // indices 0..2n (we'll only hit >=2)
    const int max_prod = n * n;     // indices 0..n^2 (we'll only hit >=1)

    // 3D dense distribution:
    // idx = k * plane + add * stride + mult
    const size_t stride   = (size_t)(max_prod + 1);        // mult dimension
    const size_t sum_span = (size_t)(max_sum + 1);         // add dimension
    const size_t plane    = sum_span * stride;             // per k plane
    const size_t table_sz = (size_t)(n + 1) * plane;       // all k

    uint64_t *counts = (uint64_t *)calloc(table_sz, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to allocate counts table (%zu entries)\n", chunk_id, table_sz);
        return 2;
    }

    // Track which bins become nonzero (to avoid scanning all of counts on output).
    U32Vec touched;
    if (u32vec_init(&touched, (size_t)1 << 20) != 0) { // start at ~1M bins, grow as needed
        fprintf(stderr, "Task %d: failed to allocate touched vector\n", chunk_id);
        free(counts);
        return 2;
    }

    const uint64_t start = (uint64_t)((TOTAL * (unsigned)chunk_id) / (unsigned)total_tasks);
    const uint64_t end   = (uint64_t)((TOTAL * (unsigned)(chunk_id + 1)) / (unsigned)total_tasks);

    // Rep counts for ordered pairs.
    uint16_t rep_sum[127]; // max is 2*63 = 126
    for (int i = 0; i <= max_sum; i++) rep_sum[i] = 0;

    // rep_prod size max_prod+1; VLA is fine (<= 3970 for n<=63)
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
        unsigned bit = (unsigned)CTZ64(mm);     // 0..n-1
        uint8_t x = (uint8_t)(bit + 1);         // 1..n
        mm &= (mm - 1);
        add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
    }

    // Record start (skip empty set)
    if (g != 0) {
        const int k = S.m;
        const size_t idx = (size_t)k * plane + (size_t)add_card * stride + (size_t)mult_card;
        uint64_t *c = &counts[idx];
        if ((*c)++ == 0) {
            if (u32vec_push(&touched, (uint32_t)idx) != 0) {
                fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                u32vec_free(&touched);
                free(counts);
                return 2;
            }
        }
    }

    // Walk forward in Gray order over the contiguous t-range [start, end).
    // Speed-up: For reflected Gray code g(t)=t^(t>>1), the toggled bit from t-1 -> t is CTZ(t).
    for (uint64_t t = start + 1; t < end; t++) {
        unsigned bit = (unsigned)CTZ64(t);      // toggled bit index
        uint64_t bmask = 1ULL << bit;

        g ^= bmask;                             // update Gray code in O(1)
        uint8_t x = (uint8_t)(bit + 1);

        if (g & bmask) {
            add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        } else {
            remove_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        }

        if (g != 0) {
            const int k = S.m;
            const size_t idx = (size_t)k * plane + (size_t)add_card * stride + (size_t)mult_card;
            uint64_t *c = &counts[idx];
            if ((*c)++ == 0) {
                if (u32vec_push(&touched, (uint32_t)idx) != 0) {
                    fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                    u32vec_free(&touched);
                    free(counts);
                    return 2;
                }
            }
        }
    }

    // Write CSV (sparse by touched bins)
    const int file_id = chunk_id + 1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/pairs_%d_%04d.csv", out_dir, n, file_id);

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Task %d: failed to open output file %s: %s\n", chunk_id, path, strerror(errno));
        u32vec_free(&touched);
        free(counts);
        return 3;
    }

    fprintf(f, "set_cardinality,add_ds_card,mult_ds_card,count\n");

    // Note: output order is traversal order of first-hit bins (not sorted).
    for (size_t i = 0; i < touched.len; i++) {
        const size_t idx = (size_t)touched.data[i];
        const uint64_t c = counts[idx];

        // Decode idx -> (k, add, mult)
        const size_t k   = idx / plane;
        const size_t rem = idx - k * plane;
        const size_t a   = rem / stride;
        const size_t m   = rem - a * stride;

        fprintf(f, "%zu,%zu,%zu,%" PRIu64 "\n", k, a, m, c);
    }

    fclose(f);
    u32vec_free(&touched);
    free(counts);
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

    // Launch up to "jobs" processes at a time.
    while (done < total_tasks) {
        while (active < jobs && next_chunk < total_tasks) {
            const int chunk_id = next_chunk;

            pid_t pid = fork();
            if (pid < 0) {
                fprintf(stderr, "Error: fork failed at chunk %d: %s\n", chunk_id, strerror(errno));
                // stop launching new children; we'll just wait for the active ones
                next_chunk = total_tasks;
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

        // Find chunk_id for this pid (linear scan is fine; total_tasks is small).
        int chunk_id = -1;
        for (int i = 0; i < total_tasks; i++) {
            if (pids[i] == pid) { chunk_id = i; break; }
        }

        int exit_code = 0;
        if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
        else exit_code = 128;

        const int file_id = (chunk_id >= 0) ? (chunk_id + 1) : 0;
        char path[4096];
        snprintf(path, sizeof(path), "%s/pairs_%d_%04d.csv", out_dir, n, file_id);

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

