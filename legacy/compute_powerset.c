// compute_powerset.c
//
// C port of the Python multiprocess structure, but writes compact distributions:
// for each task chunk_id, writes a CSV containing rows (add_ds_card, mult_ds_card, count).
//
// Usage:
//   ./compute_powerset <n> <out_dir> <jobs> <k>
//
// Output files:
//   <out_dir>/pairs_<n>_<file_id:04d>.csv   where file_id = chunk_id + 1
//
// Notes:
// - Subsets are represented as uint64 bitmasks, so requires 1 <= n <= 63.
// - This enumerates all nonempty subsets of [n].
// - Uses fork() to spawn total_tasks = jobs*k processes (mirrors your Python tasks).
//
// Compile:
//   gcc -O3 -march=native -std=c11 -Wall -Wextra -o compute_powerset compute_powerset.c

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

    // Remove trailing slash
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

// Compute |A+A| and |A*A| for A ⊆ [n] given by mask. (Kept as-is; no longer used in run_task.)
static inline void compute_add_mult_cards(uint64_t mask, int n, int *out_add, int *out_mult) {
    uint64_t A_bits = (mask << 1);

    uint64_t sums0 = 0ULL;
    uint64_t sums1 = 0ULL;

    uint64_t mm = mask;
    while (mm) {
        unsigned i = (unsigned)CTZ64(mm);
        unsigned a = i + 1;
        mm &= (mm - 1);

        if (a < 64) {
            uint64_t lo = A_bits << a;
            uint64_t hi = (a == 0) ? 0ULL : (A_bits >> (64 - a));
            sums0 |= lo;
            sums1 |= hi;
        } else {
            uint64_t hi = A_bits << (a - 64);
            sums1 |= hi;
        }
    }

    const int max_sum = 2 * n;
    if (max_sum < 64) {
        uint64_t keep = (max_sum == 63) ? ~0ULL : ((1ULL << (max_sum + 1)) - 1ULL);
        sums0 &= keep;
        sums1 = 0ULL;
    } else {
        int hi_max = max_sum - 64;
        uint64_t keep1 = (hi_max == 63) ? ~0ULL : ((1ULL << (hi_max + 1)) - 1ULL);
        sums1 &= keep1;
    }
    sums0 &= ~((1ULL << 2) - 1ULL);

    int add_card = POPCNT64(sums0) + POPCNT64(sums1);

    const int max_prod  = n * n;
    const int prod_words = (max_prod + 64) / 64;

    uint64_t prod_bits[63];
    for (int w = 0; w < prod_words; w++) prod_bits[w] = 0ULL;

    uint32_t a_arr[63];
    int m2 = 0;
    mm = mask;
    while (mm) {
        unsigned i = (unsigned)CTZ64(mm);
        a_arr[m2++] = (uint32_t)(i + 1);
        mm &= (mm - 1);
    }

    for (int i = 0; i < m2; i++) {
        uint32_t ai = a_arr[i];
        for (int j = i; j < m2; j++) {
            uint32_t p = ai * a_arr[j];
            prod_bits[p >> 6] |= 1ULL << (p & 63);
        }
    }

    int mult_card = 0;
    for (int w = 0; w < prod_words; w++) mult_card += POPCNT64(prod_bits[w]);

    *out_add  = add_card;
    *out_mult = mult_card;
}

/* ============================================================
   ADDITION #1: Gray-code + incremental update helpers
   ============================================================ */

static inline uint64_t gray64(uint64_t t) { return t ^ (t >> 1); }

// rep arrays store ordered-pair representation counts.
// Track cardinals by counting how many indices have rep[idx] > 0.
// These helpers update card when crossing zero.
static inline void rep_inc_u16(uint16_t *rep, int idx, int delta, int *card) {
    uint16_t before = rep[idx];
    uint16_t after  = (uint16_t)(before + (uint16_t)delta);
    rep[idx] = after;
    if (before == 0 && after > 0) (*card)++;
}

static inline void rep_dec_u16(uint16_t *rep, int idx, int delta, int *card) {
    uint16_t before = rep[idx];
    uint16_t after  = (uint16_t)(before - (uint16_t)delta);
    rep[idx] = after;
    if (before > 0 && after == 0) (*card)--;
}

// Maintain a dynamic list of elements in A for O(|A|) iteration and O(1) delete.
typedef struct {
    uint32_t elems[63];
    uint8_t  pos[64];
    uint8_t  in_set[64];
    int m;
} SetList;

static inline void setlist_init(SetList *S) {
    S->m = 0;
    for (int i = 0; i < 64; i++) { S->pos[i] = 0; S->in_set[i] = 0; }
}

static inline void setlist_add(SetList *S, uint32_t x) {
    S->in_set[x] = 1;
    S->pos[x] = (uint8_t)S->m;
    S->elems[S->m++] = x;
}

static inline void setlist_remove(SetList *S, uint32_t x) {
    uint8_t idx = S->pos[x];
    int last_i = S->m - 1;
    uint32_t last = S->elems[last_i];
    S->elems[idx] = last;
    S->pos[last] = idx;
    S->m = last_i;
    S->in_set[x] = 0;
}

// Ordered-pair reps: when adding x, for each y in A(before):
// (x,y) and (y,x): +2; plus (x,x): +1.
static inline void add_element(SetList *S, uint32_t x,
                               uint16_t *rep_sum, uint16_t *rep_prod,
                               int *add_card, int *mult_card) {
    for (int i = 0; i < S->m; i++) {
        uint32_t y = S->elems[i];
        rep_inc_u16(rep_sum,  (int)(x + y), 2, add_card);
        rep_inc_u16(rep_prod, (int)(x * y), 2, mult_card);
    }
    rep_inc_u16(rep_sum,  (int)(x + x), 1, add_card);
    rep_inc_u16(rep_prod, (int)(x * x), 1, mult_card);
    setlist_add(S, x);
}

static inline void remove_element(SetList *S, uint32_t x,
                                  uint16_t *rep_sum, uint16_t *rep_prod,
                                  int *add_card, int *mult_card) {
    for (int i = 0; i < S->m; i++) {
        uint32_t y = S->elems[i];
        if (y == x) continue;
        rep_dec_u16(rep_sum,  (int)(x + y), 2, add_card);
        rep_dec_u16(rep_prod, (int)(x * y), 2, mult_card);
    }
    rep_dec_u16(rep_sum,  (int)(x + x), 1, add_card);
    rep_dec_u16(rep_prod, (int)(x * x), 1, mult_card);
    setlist_remove(S, x);
}

/* ============================================================
   ADDITION #2: Modify run_task() to use contiguous Gray blocks
   ============================================================ */

static int run_task(int chunk_id, int n, int total_tasks, const char *out_dir) {
    // total = 1<<n (n<=63)
    const uint64_t total = (1ULL << n);

    const int max_sum  = 2 * n;
    const int max_prod = n * n;

    // Dense distribution table:
    // counts[(add)*(max_prod+1) + mult]
    const size_t stride = (size_t)(max_prod + 1);
    const size_t table_size = (size_t)(max_sum + 1) * stride;

    uint32_t *counts = (uint32_t *)calloc(table_size, sizeof(uint32_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to allocate counts table (%zu entries)\n", chunk_id, table_size);
        return 2;
    }

    // Partition [0, 2^n) into contiguous Gray-index blocks.
    // This is required for incremental Gray updates.
    const uint64_t start = (total * (uint64_t)chunk_id) / (uint64_t)total_tasks;
    const uint64_t end   = (total * (uint64_t)(chunk_id + 1)) / (uint64_t)total_tasks;

    // Representation counts for ordered pairs.
    // sums in [2..2n] so rep_sum needs indices up to 2n.
    uint16_t rep_sum[127];
    for (int i = 0; i <= max_sum; i++) rep_sum[i] = 0;

    // products in [1..n^2]; allocate exactly max_prod+1
    uint16_t *rep_prod = (uint16_t *)calloc((size_t)(max_prod + 1), sizeof(uint16_t));
    if (!rep_prod) {
        fprintf(stderr, "Task %d: failed to allocate rep_prod\n", chunk_id);
        free(counts);
        return 2;
    }

    SetList S;
    setlist_init(&S);

    int add_card = 0;
    int mult_card = 0;

    // Initialize A = Gray(start) from empty by adding its elements.
    uint64_t g = gray64(start);
    uint64_t mm = g;
    while (mm) {
        unsigned bit = (unsigned)CTZ64(mm);     // 0..n-1
        uint32_t x = (uint32_t)(bit + 1);       // 1..n
        mm &= (mm - 1);
        add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
    }

    // Record start (skip empty set, matching your original Python behavior)
    if (g != 0) {
        counts[(size_t)add_card * stride + (size_t)mult_card] += 1U;
    }

    // Walk forward in Gray order: t = start+1 ... end-1
    uint64_t prev_g = g;
    for (uint64_t t = start + 1; t < end; t++) {
        g = gray64(t);
        uint64_t diff = g ^ prev_g;            // exactly one bit set
        unsigned bit = (unsigned)CTZ64(diff);
        uint32_t x = (uint32_t)(bit + 1);

        if ((g >> bit) & 1ULL) {
            add_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        } else {
            remove_element(&S, x, rep_sum, rep_prod, &add_card, &mult_card);
        }

        if (g != 0) {
            counts[(size_t)add_card * stride + (size_t)mult_card] += 1U;
        }

        prev_g = g;
    }

    free(rep_prod);

    // Write CSV (unchanged)
    const int file_id = chunk_id + 1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/pairs_%d_%04d.csv", out_dir, n, file_id);

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Task %d: failed to open output file %s: %s\n", chunk_id, path, strerror(errno));
        free(counts);
        return 3;
    }

    fprintf(f, "add_ds_card,mult_ds_card,count\n");
    for (int a = 0; a <= max_sum; a++) {
        const size_t base = (size_t)a * stride;
        for (int m = 0; m <= max_prod; m++) {
            const uint32_t c = counts[base + (size_t)m];
            if (c) {
                fprintf(f, "%d,%d,%" PRIu32 "\n", a, m, c);
            }
        }
    }

    fclose(f);
    free(counts);
    return 0;
}

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
    int   *chunk_of_pid = (int *)calloc((size_t)total_tasks, sizeof(int));
    if (!pids || !chunk_of_pid) {
        fprintf(stderr, "Error: allocation failure\n");
        free(pids);
        free(chunk_of_pid);
        return 1;
    }

    for (int chunk_id = 0; chunk_id < total_tasks; chunk_id++) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "Error: fork failed at chunk %d: %s\n", chunk_id, strerror(errno));
            break;
        }
        if (pid == 0) {
            int rc = run_task(chunk_id, n, total_tasks, out_dir);
            _exit(rc);
        }
        pids[chunk_id] = pid;
        chunk_of_pid[chunk_id] = chunk_id;
    }

    int done = 0;
    for (;;) {
        int status = 0;
        pid_t pid = wait(&status);
        if (pid < 0) {
            if (errno == EINTR) continue;
            break;
        }

        int chunk_id = -1;
        for (int i = 0; i < total_tasks; i++) {
            if (pids[i] == pid) { chunk_id = chunk_of_pid[i]; break; }
        }

        done += 1;

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
    free(chunk_of_pid);
    return 0;
}
