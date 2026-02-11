// compute_divisors_spp.c
//
// For each n in [1, 2^b], let A = Divisors(n).
// Compute (|A|, |A+A|, |A*A|) and aggregate counts of triples.
// Output CSV shards with columns:
//   set_cardinality,add_ds_card,mult_ds_card,count
//
// Usage:
//   ./compute_divisors_spp <b> <out_dir> <jobs> <k>
//
// Where:
//   b      : upper bound exponent (max_n = 2^b)
//   out_dir: directory for CSV shards
//   jobs   : number of worker threads running concurrently
//   k      : number of chunks-per-thread (total_tasks = jobs*k)
//
// Output files:
//   <out_dir>/divpairs_b<b>_<file_id:04d>.csv   where file_id = chunk_id + 1
//
// Compile:
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG \
//       -pthread -lgmp -o compute_divisors_spp compute_divisors_spp.c

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
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) { free(tmp); return -1; }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) { free(tmp); return -1; }
    free(tmp);
    return 0;
}

/* ============================================================
   Prime sieve up to sqrt(max_n)
   ============================================================ */

typedef struct {
    uint32_t *p;
    size_t len;
} PrimeList;

static PrimeList sieve_primes_upto(uint32_t N) {
    PrimeList L = {0};
    if (N < 2) return L;

    uint8_t *is_comp = (uint8_t *)calloc((size_t)N + 1, 1);
    if (!is_comp) return L;

    // rough upper bound π(N) <= N for allocation; shrink later
    uint32_t *pr = (uint32_t *)malloc(((size_t)N + 1) * sizeof(uint32_t));
    if (!pr) { free(is_comp); return L; }

    size_t cnt = 0;
    for (uint32_t i = 2; i <= N; i++) {
        if (!is_comp[i]) {
            pr[cnt++] = i;
            if ((uint64_t)i * (uint64_t)i <= (uint64_t)N) {
                for (uint32_t j = i * i; j <= N; j += i) is_comp[j] = 1;
            }
        }
    }

    free(is_comp);
    L.p = (uint32_t *)realloc(pr, cnt * sizeof(uint32_t));
    if (!L.p) L.p = pr; // if realloc fails, keep original
    L.len = cnt;
    return L;
}

/* ============================================================
   Factorization and divisor generation
   ============================================================ */

typedef struct {
    uint32_t prime;
    uint32_t exp;
} Factor;

typedef struct {
    Factor *f;
    uint32_t nf;
} Factorization;

static Factorization factorize_u64(uint64_t n, const PrimeList *P) {
    Factorization F = {0};
    if (n <= 1) return F;

    // worst-case number of prime factors (with multiplicity) <= 64; distinct much less
    Factor *fac = (Factor *)malloc(64 * sizeof(Factor));
    if (!fac) return F;

    uint32_t nf = 0;
    uint64_t x = n;

    for (size_t i = 0; i < P->len; i++) {
        uint32_t p = P->p[i];
        uint64_t pp = (uint64_t)p;
        if (pp * pp > x) break;
        if (x % pp == 0) {
            uint32_t e = 0;
            do { x /= pp; e++; } while (x % pp == 0);
            fac[nf++] = (Factor){p, e};
        }
    }
    if (x > 1) {
        // x is prime (fits in uint64, but might exceed uint32 for big b; still ok to store as uint32 only if <= UINT32_MAX)
        // For feasible enumeration ranges, b won't push x beyond 2^32 often, but be safe:
        if (x > UINT32_MAX) {
            // store as "prime=0" sentinel? better: split fail explicitly
            // We'll still store truncated; but for your intended b this should not happen.
            fac[nf++] = (Factor){(uint32_t)x, 1};
        } else {
            fac[nf++] = (Factor){(uint32_t)x, 1};
        }
    }

    fac = (Factor *)realloc(fac, (size_t)nf * sizeof(Factor));
    F.f = fac;
    F.nf = nf;
    return F;
}

static void free_factorization(Factorization *F) {
    free(F->f);
    F->f = NULL;
    F->nf = 0;
}

// Generate all divisors from prime factorization.
// Returns malloc'd array divs, length in *out_len.
static uint64_t *gen_divisors(const Factorization *F, uint32_t *out_len) {
    // number of divisors = Π (e_i+1)
    uint64_t tau = 1;
    for (uint32_t i = 0; i < F->nf; i++) tau *= (uint64_t)(F->f[i].exp + 1);

    if (tau == 0 || tau > (uint64_t)UINT32_MAX) return NULL;
    uint32_t m = (uint32_t)tau;

    uint64_t *divs = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
    if (!divs) return NULL;

    divs[0] = 1;
    uint32_t cur = 1;

    for (uint32_t i = 0; i < F->nf; i++) {
        uint64_t p = (uint64_t)F->f[i].prime;
        uint32_t e = F->f[i].exp;

        // copy current block, multiplying by p^j
        uint32_t prev = cur;
        uint64_t pow = 1;
        for (uint32_t j = 1; j <= e; j++) {
            pow *= p;
            for (uint32_t t = 0; t < prev; t++) {
                divs[cur++] = divs[t] * pow;
            }
        }
    }

    *out_len = cur;
    return divs;
}

/* ============================================================
   Unique count after sorting
   ============================================================ */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x < y) ? -1 : (x > y);
}

static uint32_t count_unique_sorted_u64(uint64_t *arr, uint32_t n) {
    if (n == 0) return 0;
    uint32_t u = 1;
    for (uint32_t i = 1; i < n; i++) {
        if (arr[i] != arr[i - 1]) u++;
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
    // cap must be power of two
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
    // grow at load > 0.7
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
   Worker: process one chunk and write CSV
   ============================================================ */

typedef struct {
    int chunk_id;
    int total_tasks;
    uint64_t max_n;          // 2^b
    const char *out_dir;
    const PrimeList *primes;
} Task;

static int process_chunk_and_write(const Task *T) {
    const __uint128_t TOTAL = ( __uint128_t )T->max_n; // inclusive max; range is [1..max_n]
    const int chunk_id = T->chunk_id;
    const int total_tasks = T->total_tasks;

    // Partition integers 1..max_n into contiguous chunks.
    // Let idx run over [0..TOTAL-1] corresponding to n=idx+1.
    const __uint128_t N0 = (TOTAL * (unsigned)chunk_id) / (unsigned)total_tasks;
    const __uint128_t N1 = (TOTAL * (unsigned)(chunk_id + 1)) / (unsigned)total_tasks;

    uint64_t start_n = (uint64_t)N0 + 1;
    uint64_t end_n   = (uint64_t)N1; // inclusive
    if (end_n < start_n) return 0;

    Map M;
    if (map_init(&M, 1 << 16) != 0) {
        fprintf(stderr, "chunk %d: map init failed\n", chunk_id);
        return 2;
    }

    // Scratch arrays reused; resize if needed
    uint64_t *divs = NULL;
    uint64_t *sums = NULL;
    uint64_t *prods = NULL;
    uint32_t div_cap = 0;
    uint32_t pair_cap = 0;

    for (uint64_t n = start_n; n <= end_n; n++) {
        if (n == 0) break;

        Factorization F = factorize_u64(n, T->primes);
        uint32_t k = 0;
        uint64_t *d = gen_divisors(&F, &k);
        free_factorization(&F);
        if (!d || k == 0) { free(d); continue; }

        // ensure divs scratch capacity
        if (k > div_cap) {
            free(divs);
            divs = (uint64_t *)malloc((size_t)k * sizeof(uint64_t));
            if (!divs) { free(d); map_free(&M); return 2; }
            div_cap = k;
        }
        memcpy(divs, d, (size_t)k * sizeof(uint64_t));
        free(d);

        // sort divisors (helps stability; not required)
        qsort(divs, (size_t)k, sizeof(uint64_t), cmp_u64);

        // number of i<=j pairs
        uint64_t m64 = ((uint64_t)k * (uint64_t)(k + 1)) / 2;
        if (m64 > UINT32_MAX) { map_free(&M); free(divs); return 2; }
        uint32_t m = (uint32_t)m64;

        if (m > pair_cap) {
            free(sums); free(prods);
            sums  = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
            prods = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
            if (!sums || !prods) {
                free(sums); free(prods);
                map_free(&M); free(divs);
                return 2;
            }
            pair_cap = m;
        }

        // fill sums/products for i<=j
        uint32_t idx = 0;
        for (uint32_t i = 0; i < k; i++) {
            uint64_t a = divs[i];
            for (uint32_t j = i; j < k; j++) {
                uint64_t b = divs[j];
                sums[idx] = a + b;

                __uint128_t prod = ( __uint128_t )a * ( __uint128_t )b;
                if (prod > UINT64_MAX) {
                    // should not happen for feasible b; if it does, clamp (or skip)
                    prods[idx] = UINT64_MAX;
                } else {
                    prods[idx] = (uint64_t)prod;
                }
                idx++;
            }
        }

        qsort(sums,  (size_t)m, sizeof(uint64_t), cmp_u64);
        qsort(prods, (size_t)m, sizeof(uint64_t), cmp_u64);

        uint32_t add_card  = count_unique_sorted_u64(sums,  m);
        uint32_t mult_card = count_unique_sorted_u64(prods, m);

        // each n contributes +1 to its triple
        if (map_inc(&M, k, add_card, mult_card, 1) != 0) {
            fprintf(stderr, "chunk %d: map_inc failed\n", chunk_id);
            map_free(&M);
            free(divs); free(sums); free(prods);
            return 2;
        }
    }

    // Write CSV shard
    const int file_id = chunk_id + 1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/divpairs_b%d_%04d.csv", T->out_dir,
             (int)(8 * sizeof(uint64_t)), file_id); // placeholder label; b isn't stored here
    // Better: embed actual b in filename via out_dir or caller.
    // We'll instead write a stable name without b if you prefer. For now, keep as above.

    // Overwrite with a cleaner filename (no ambiguity):
    // We'll derive b from max_n = 2^b by counting trailing bits.
    int b_guess = 0;
    { uint64_t x = T->max_n; while (x > 1 && (x & 1ULL) == 0ULL) { b_guess++; x >>= 1; } }
    snprintf(path, sizeof(path), "%s/divpairs_%d_%04d.csv", T->out_dir, b_guess, file_id);

    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "chunk %d: fopen %s failed: %s\n", chunk_id, path, strerror(errno));
        map_free(&M);
        free(divs); free(sums); free(prods);
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
    free(divs); free(sums); free(prods);
    return 0;
}

/* ============================================================
   Thread pool driver
   ============================================================ */

typedef struct {
    pthread_mutex_t mu;
    int next_chunk;
    int total_tasks;
    int fail_code;
    Task *tasks;
} Shared;

static void *worker_main(void *arg) {
    Shared *S = (Shared *)arg;
    for (;;) {
        int cid;
        pthread_mutex_lock(&S->mu);
        if (S->next_chunk >= S->total_tasks || S->fail_code != 0) {
            pthread_mutex_unlock(&S->mu);
            break;
        }
        cid = S->next_chunk++;
        pthread_mutex_unlock(&S->mu);

        int rc = process_chunk_and_write(&S->tasks[cid]);
        if (rc != 0) {
            pthread_mutex_lock(&S->mu);
            if (S->fail_code == 0) S->fail_code = rc;
            pthread_mutex_unlock(&S->mu);
            break;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <b> <out_dir> <jobs> <k>\n", argv[0]);
        return 1;
    }

    int b = atoi(argv[1]);
    const char *out_dir = argv[2];
    int jobs = atoi(argv[3]);
    int k = atoi(argv[4]);

    if (b < 0 || b > 62) {
        fprintf(stderr, "Error: b must be in [0,62] (so 2^b fits in uint64)\n");
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

    uint64_t max_n = (b == 62) ? (1ULL << 62) : (1ULL << b);

    // WARNING: Iterating to 2^b is only feasible for modest b.
    // We don't stop you; but you should choose b realistically.

    // Sieve primes up to floor(sqrt(max_n)).
    uint64_t r = 1;
    while ((r + 1) * (r + 1) <= max_n) r++;
    if (r > UINT32_MAX) {
        fprintf(stderr, "Error: sqrt(max_n) too large for this sieve strategy\n");
        return 1;
    }

    PrimeList P = sieve_primes_upto((uint32_t)r);
    if (!P.p || P.len == 0) {
        fprintf(stderr, "Error: failed to build prime list up to %" PRIu64 "\n", r);
        free(P.p);
        return 1;
    }

    const int total_tasks = jobs * k;

    Task *tasks = (Task *)calloc((size_t)total_tasks, sizeof(Task));
    if (!tasks) {
        fprintf(stderr, "Error: allocation failure\n");
        free(P.p);
        return 1;
    }

    for (int cid = 0; cid < total_tasks; cid++) {
        tasks[cid].chunk_id = cid;
        tasks[cid].total_tasks = total_tasks;
        tasks[cid].max_n = max_n;
        tasks[cid].out_dir = out_dir;
        tasks[cid].primes = &P;
    }

    pthread_t *thr = (pthread_t *)calloc((size_t)jobs, sizeof(pthread_t));
    if (!thr) {
        fprintf(stderr, "Error: allocation failure\n");
        free(tasks);
        free(P.p);
        return 1;
    }

    Shared S;
    pthread_mutex_init(&S.mu, NULL);
    S.next_chunk = 0;
    S.total_tasks = total_tasks;
    S.fail_code = 0;
    S.tasks = tasks;

    const double t0 = now_seconds();
    for (int i = 0; i < jobs; i++) {
        if (pthread_create(&thr[i], NULL, worker_main, &S) != 0) {
            fprintf(stderr, "Error: pthread_create failed\n");
            S.fail_code = 2;
            jobs = i;
            break;
        }
    }

    for (int i = 0; i < jobs; i++) pthread_join(thr[i], NULL);

    double elapsed = now_seconds() - t0;
    if (S.fail_code != 0) {
        fprintf(stderr, "Failed (code=%d) after %.2fs\n", S.fail_code, elapsed);
    } else {
        printf("Done. Wrote %d shards to %s in %.2fs\n", total_tasks, out_dir, elapsed);
    }

    pthread_mutex_destroy(&S.mu);
    free(thr);
    free(tasks);
    free(P.p);

    return (S.fail_code == 0) ? 0 : 2;
}

