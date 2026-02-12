// compute_divisor_prefix_subsets_spp.c
//
// For each n in [1..n_max], let D_n be the n smallest divisors of
//   N0 = 2^11 * 3^6 * 5^4 * 7^3 * 11^3 * 13^2 * 17.
// For each subset A ⊆ D_n (including empty), compute:
//   k = |A|
//   add = |A + A|  (distinct sums a_i + a_j with i<=j)
//   mult = |A * A| (distinct products a_i * a_j with i<=j; via GMP)
// Aggregate counts of (k, add, mult) and write CSV shards.
//
// Usage:
//   ./compute_divisor_prefix_subsets_spp <n_max> <out_dir> <jobs> <k>
//
// Output files:
//   <out_dir>/divprefix_N0_n<n>_<chunk:04d>.csv
//
// Compile:
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG \
//       -pthread -lgmp -o compute_divisor_prefix_subsets_spp compute_divisor_prefix_subsets_spp.c

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <gmp.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---------------- timing / mkdir ---------------- */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

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
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) { free(tmp); return -1; }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) { free(tmp); return -1; }
    free(tmp);
    return 0;
}

/* ---------------- sorting / uniques for u64 ---------------- */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x < y) ? -1 : (x > y);
}

static uint32_t count_unique_sorted_u64(const uint64_t *arr, uint32_t n) {
    if (n == 0) return 0;
    uint32_t u = 1;
    for (uint32_t i = 1; i < n; i++) {
        if (arr[i] != arr[i - 1]) u++;
    }
    return u;
}

/* ---------------- mpz helpers / sorting / uniques for mpz ---------------- */

static inline void mpz_set_u64(mpz_t z, uint64_t x) {
#if ULONG_MAX >= 0xFFFFFFFFFFFFFFFFULL
    mpz_set_ui(z, (unsigned long)x);
#else
    // portable fallback (shouldn't trigger on x86_64 Linux, but safe)
    mpz_import(z, 1, -1, sizeof(x), 0, 0, &x);
#endif
}

typedef struct { mpz_t z; } mpz_wrap;

static int cmp_mpz_wrap(const void *a, const void *b) {
    const mpz_wrap *x = (const mpz_wrap *)a;
    const mpz_wrap *y = (const mpz_wrap *)b;
    return mpz_cmp(x->z, y->z);
}

static uint32_t count_unique_sorted_mpz(mpz_wrap *arr, uint32_t n) {
    if (n == 0) return 0;
    uint32_t u = 1;
    for (uint32_t i = 1; i < n; i++) {
        if (mpz_cmp(arr[i].z, arr[i - 1].z) != 0) u++;
    }
    return u;
}

/* ============================================================
   Hash map for (k, add, mult) -> count (open addressing)
   ============================================================ */

typedef struct {
    uint32_t k;
    uint32_t add;
    uint32_t mult;
    uint64_t count;
} Entry;

typedef struct {
    Entry *tab;
    uint8_t *used;
    size_t cap;
    size_t fill;
} Map;

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t key_hash(uint32_t k, uint32_t add, uint32_t mult) {
    uint64_t x = ((uint64_t)k << 42) ^ ((uint64_t)add << 21) ^ (uint64_t)mult;
    return splitmix64(x);
}

static int map_init(Map *M, size_t cap0) {
    size_t cap = 1;
    while (cap < cap0) cap <<= 1;
    M->tab = (Entry *)calloc(cap, sizeof(Entry));
    M->used = (uint8_t *)calloc(cap, 1);
    if (!M->tab || !M->used) {
        free(M->tab); free(M->used);
        M->tab = NULL; M->used = NULL; M->cap = M->fill = 0;
        return -1;
    }
    M->cap = cap;
    M->fill = 0;
    return 0;
}

static void map_free(Map *M) {
    free(M->tab);
    free(M->used);
    M->tab = NULL;
    M->used = NULL;
    M->cap = M->fill = 0;
}

static int map_rehash(Map *M, size_t newcap) {
    Map N;
    if (map_init(&N, newcap) != 0) return -1;

    for (size_t i = 0; i < M->cap; i++) {
        if (!M->used[i]) continue;
        Entry e = M->tab[i];

        uint64_t h = key_hash(e.k, e.add, e.mult);
        size_t mask = N.cap - 1;
        size_t j = (size_t)h & mask;
        while (N.used[j]) j = (j + 1) & mask;

        N.used[j] = 1;
        N.tab[j] = e;
        N.fill++;
    }

    map_free(M);
    *M = N;
    return 0;
}

static int map_inc(Map *M, uint32_t k, uint32_t add, uint32_t mult, uint64_t delta) {
    if ((M->fill + 1) * 10 >= M->cap * 7) {
        if (map_rehash(M, M->cap ? (M->cap * 2) : 1024) != 0) return -1;
    }

    uint64_t h = key_hash(k, add, mult);
    size_t mask = M->cap - 1;
    size_t i = (size_t)h & mask;

    while (M->used[i]) {
        Entry *e = &M->tab[i];
        if (e->k == k && e->add == add && e->mult == mult) {
            e->count += delta;
            return 0;
        }
        i = (i + 1) & mask;
    }

    M->used[i] = 1;
    M->tab[i] = (Entry){k, add, mult, delta};
    M->fill++;
    return 0;
}

/* ============================================================
   Generate all divisors of N0 (fits in uint64), sort ascending
   ============================================================ */

typedef struct { uint32_t p; uint32_t e; } Factor;

static int build_divisors_N0(uint64_t **out_divs, uint32_t *out_len) {
    // N0 = 2^11 * 3^6 * 5^4 * 7^3 * 11^3 * 13^2 * 17^1
    const Factor F[] = {
        {2,11},{3,6},{5,4},{7,3},{11,3},{13,2},{17,1}
    };
    const uint32_t nf = (uint32_t)(sizeof(F)/sizeof(F[0]));

    uint64_t tau = 1;
    for (uint32_t i = 0; i < nf; i++) tau *= (uint64_t)(F[i].e + 1);
    if (tau == 0 || tau > UINT32_MAX) return -1;

    uint32_t m = (uint32_t)tau;
    uint64_t *divs = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
    if (!divs) return -1;

    divs[0] = 1;
    uint32_t cur = 1;

    for (uint32_t i = 0; i < nf; i++) {
        uint64_t p = (uint64_t)F[i].p;
        uint32_t e = F[i].e;

        uint32_t prev = cur;
        uint64_t pow = 1;
        for (uint32_t j = 1; j <= e; j++) {
            pow *= p;
            for (uint32_t t = 0; t < prev; t++) {
                divs[cur++] = divs[t] * pow;
            }
        }
    }

    if (cur != m) { free(divs); return -1; }

    qsort(divs, (size_t)m, sizeof(uint64_t), cmp_u64);

    *out_divs = divs;
    *out_len = m;
    return 0;
}

/* ============================================================
   Worker over subset masks of D_n (n smallest divisors)
   ============================================================ */

typedef struct {
    int n;                  // current prefix size
    int chunk_id;
    int total_tasks;
    const uint64_t *prefix_divs;  // length >= n
    const char *out_dir;
} Task;

typedef struct {
    // per-thread scratch to avoid realloc/clear overhead
    uint64_t *sums;     // u64 sums, length max_pairs
    mpz_wrap *prods;    // mpz products, length max_pairs
    uint32_t max_pairs;
} Scratch;

static int scratch_init(Scratch *S, uint32_t max_n) {
    uint64_t m64 = ((uint64_t)max_n * (uint64_t)(max_n + 1)) / 2;
    if (m64 > UINT32_MAX) return -1;
    S->max_pairs = (uint32_t)m64;

    S->sums = (uint64_t *)malloc((size_t)S->max_pairs * sizeof(uint64_t));
    S->prods = (mpz_wrap *)malloc((size_t)S->max_pairs * sizeof(mpz_wrap));
    if (!S->sums || !S->prods) {
        free(S->sums); free(S->prods);
        S->sums = NULL; S->prods = NULL; S->max_pairs = 0;
        return -1;
    }
    for (uint32_t i = 0; i < S->max_pairs; i++) mpz_init(S->prods[i].z);
    return 0;
}

static void scratch_free(Scratch *S) {
    if (S->prods) {
        for (uint32_t i = 0; i < S->max_pairs; i++) mpz_clear(S->prods[i].z);
    }
    free(S->sums);
    free(S->prods);
    S->sums = NULL;
    S->prods = NULL;
    S->max_pairs = 0;
}

static inline __uint128_t pow2_u128(int n) {
    // requires 0 <= n <= 126 realistically; we will restrict n <= 62 for feasibility.
    return ((__uint128_t)1) << (unsigned)n;
}

static int process_chunk_and_write(const Task *T, Scratch *S) {
    const int n = T->n;
    if (n < 0 || n > 62) return 2; // keep safe and realistic
    const __uint128_t TOTAL = pow2_u128(n);

    const int chunk_id = T->chunk_id;
    const int total_tasks = T->total_tasks;

    const __uint128_t M0 = (TOTAL * (unsigned)chunk_id) / (unsigned)total_tasks;
    const __uint128_t M1 = (TOTAL * (unsigned)(chunk_id + 1)) / (unsigned)total_tasks;
    if (M1 <= M0) return 0;

    Map M;
    if (map_init(&M, 1 << 16) != 0) {
        fprintf(stderr, "n=%d chunk=%d: map init failed\n", n, chunk_id);
        return 2;
    }

    uint64_t elems[64];

    for (__uint128_t mask128 = M0; mask128 < M1; mask128++) {
        // mask fits in 64 bits because n <= 62
        uint64_t mask = (uint64_t)mask128;

        uint32_t k = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1ULL << (unsigned)i)) {
                elems[k++] = T->prefix_divs[i];
            }
        }

        uint32_t add_card = 0, mult_card = 0;

        if (k == 0) {
            add_card = 0;
            mult_card = 0;
        } else {
            uint64_t m64 = ((uint64_t)k * (uint64_t)(k + 1)) / 2;
            uint32_t m = (uint32_t)m64;

            // sums / products for i<=j
            uint32_t idx = 0;
            for (uint32_t i = 0; i < k; i++) {
                uint64_t a = elems[i];
                for (uint32_t j = i; j < k; j++) {
                    uint64_t b = elems[j];

                    // sums fit in uint64 (<= 2*N0 < 2^63)
                    S->sums[idx] = a + b;

                    // products: mpz
                    mpz_set_u64(S->prods[idx].z, a);
                    mpz_mul_ui(S->prods[idx].z, S->prods[idx].z, (unsigned long)b);

                    idx++;
                }
            }

            qsort(S->sums, (size_t)m, sizeof(uint64_t), cmp_u64);
            qsort(S->prods, (size_t)m, sizeof(mpz_wrap), cmp_mpz_wrap);

            add_card = count_unique_sorted_u64(S->sums, m);
            mult_card = count_unique_sorted_mpz(S->prods, m);
        }

        if (map_inc(&M, k, add_card, mult_card, 1) != 0) {
            fprintf(stderr, "n=%d chunk=%d: map_inc failed\n", n, chunk_id);
            map_free(&M);
            return 2;
        }
    }

    // write shard
    char path[4096];
    snprintf(path, sizeof(path), "%s/divprefix_N0_n%d_%04d.csv", T->out_dir, n, chunk_id + 1);

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "n=%d chunk=%d: fopen %s failed: %s\n",
                n, chunk_id, path, strerror(errno));
        map_free(&M);
        return 3;
    }

    fputs("set_cardinality,add_ds_card,mult_ds_card,count\n", f);
    for (size_t i = 0; i < M.cap; i++) {
        if (!M.used[i]) continue;
        const Entry *e = &M.tab[i];
        fprintf(f, "%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu64 "\n",
                e->k, e->add, e->mult, e->count);
    }
    fclose(f);

    map_free(&M);
    return 0;
}

/* ============================================================
   Thread pool driver (reused per n)
   ============================================================ */

typedef struct {
    pthread_mutex_t mu;
    int next_chunk;
    int total_tasks;
    int fail_code;

    Task *tasks;

    // thread-local scratch storage
    int scratch_ready;
    uint32_t scratch_max_n;
} Shared;

typedef struct {
    Shared *S;
    int tid;
} WorkerArg;

static void *worker_main(void *arg) {
    WorkerArg *W = (WorkerArg *)arg;
    Shared *S = W->S;

    // allocate per-thread scratch once
    Scratch scratch = {0};
    if (!S->scratch_ready) {
        // should not happen; main sets it up
    }
    if (scratch_init(&scratch, S->scratch_max_n) != 0) {
        pthread_mutex_lock(&S->mu);
        if (S->fail_code == 0) S->fail_code = 2;
        pthread_mutex_unlock(&S->mu);
        return NULL;
    }

    for (;;) {
        int cid;
        pthread_mutex_lock(&S->mu);
        if (S->next_chunk >= S->total_tasks || S->fail_code != 0) {
            pthread_mutex_unlock(&S->mu);
            break;
        }
        cid = S->next_chunk++;
        pthread_mutex_unlock(&S->mu);

        int rc = process_chunk_and_write(&S->tasks[cid], &scratch);
        if (rc != 0) {
            pthread_mutex_lock(&S->mu);
            if (S->fail_code == 0) S->fail_code = rc;
            pthread_mutex_unlock(&S->mu);
            break;
        }
    }

    scratch_free(&scratch);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <n_max> <out_dir> <jobs> <k>\n", argv[0]);
        return 1;
    }

    int n_max = atoi(argv[1]);
    const char *out_dir = argv[2];
    int jobs = atoi(argv[3]);
    int k = atoi(argv[4]);

    if (n_max < 1 || n_max > 62) {
        fprintf(stderr, "Error: n_max must be in [1,62] (practically you probably want <= 32)\n");
        return 1;
    }
    if (jobs < 1 || jobs > 512) {
        fprintf(stderr, "Error: jobs must be in [1,512]\n");
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

    // Build and sort all divisors of N0
    uint64_t *all_divs = NULL;
    uint32_t all_len = 0;
    if (build_divisors_N0(&all_divs, &all_len) != 0) {
        fprintf(stderr, "Error: failed to build divisors of N0\n");
        return 1;
    }
    if ((uint32_t)n_max > all_len) {
        fprintf(stderr, "Error: n_max=%d exceeds tau(N0)=%" PRIu32 "\n", n_max, all_len);
        free(all_divs);
        return 1;
    }

    const int total_tasks = jobs * k;

    Task *tasks = (Task *)calloc((size_t)total_tasks, sizeof(Task));
    if (!tasks) {
        fprintf(stderr, "Error: allocation failure\n");
        free(all_divs);
        return 1;
    }

    pthread_t *thr = (pthread_t *)calloc((size_t)jobs, sizeof(pthread_t));
    WorkerArg *wargs = (WorkerArg *)calloc((size_t)jobs, sizeof(WorkerArg));
    if (!thr || !wargs) {
        fprintf(stderr, "Error: allocation failure\n");
        free(thr); free(wargs); free(tasks); free(all_divs);
        return 1;
    }

    Shared S;
    pthread_mutex_init(&S.mu, NULL);
    S.total_tasks = total_tasks;
    S.fail_code = 0;
    S.tasks = tasks;
    S.scratch_ready = 1;
    S.scratch_max_n = (uint32_t)n_max;

    // Create worker threads once; they will exit after one run.
    // So we re-create threads for each n (simpler and robust).
    const double t0 = now_seconds();

    for (int n = 1; n <= n_max; n++) {
        // reset shared counters
        S.next_chunk = 0;
        S.fail_code = 0;

        // prepare tasks for this n
        const uint64_t *prefix = all_divs; // smallest divisors already at start
        for (int cid = 0; cid < total_tasks; cid++) {
            tasks[cid].n = n;
            tasks[cid].chunk_id = cid;
            tasks[cid].total_tasks = total_tasks;
            tasks[cid].prefix_divs = prefix;
            tasks[cid].out_dir = out_dir;
        }

        // spawn threads
        for (int i = 0; i < jobs; i++) {
            wargs[i].S = &S;
            wargs[i].tid = i;
            if (pthread_create(&thr[i], NULL, worker_main, &wargs[i]) != 0) {
                fprintf(stderr, "Error: pthread_create failed at n=%d\n", n);
                S.fail_code = 2;
                jobs = i; // join what we started
                break;
            }
        }

        for (int i = 0; i < jobs; i++) pthread_join(thr[i], NULL);

        if (S.fail_code != 0) {
            fprintf(stderr, "Failed at n=%d (code=%d)\n", n, S.fail_code);
            break;
        } else {
            printf("n=%d: wrote %d shards to %s\n", n, total_tasks, out_dir);
        }
    }

    double elapsed = now_seconds() - t0;
    printf("Done in %.2fs\n", elapsed);

    pthread_mutex_destroy(&S.mu);
    free(thr);
    free(wargs);
    free(tasks);
    free(all_divs);
    return 0;
}

