// compute_spp_fixedk_up_to.c
//
// Enumerate all k-subsets A of [N] (for N=k..N_max) and count pairs (|A+A|, |A*A|).
// Writes a single merged CSV:  <out_dir>/spp_k_up_to_n.csv
//
// Usage:
//   ./compute_spp_fixedk_up_to <N_max> <k> <out_dir> <jobs> <chunks_per_job>
//
// Output CSV columns:
//   N,set_cardinality,add_ds_card,mult_ds_card,count
//
// Notes:
// - Uses multi-process chunking: total_tasks = jobs * chunks_per_job.
// - Caps concurrency to <= jobs.
// - Partitions combination ranks [0, C(N,k)) into contiguous ranges per task.
// - For each subset, computes distinct sums/products via timestamped mark arrays
//   (no expensive clearing).
//
// Compile (recommended):
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG \
//       -o compute_spp_fixedk_up_to compute_spp_fixedk_up_to.c

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
  #define CTZ64(x) __builtin_ctzll((unsigned long long)(x))
#else
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
   Binomial / combinadic utilities (uint64 safe for small k)
   ============================================================ */

static uint64_t binom_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    __uint128_t res = 1;
    for (int i = 1; i <= k; i++) {
        res = res * (uint64_t)(n - k + i);
        res /= (uint64_t)i;
        if (res > UINT64_MAX) return 0; // overflow guard
    }
    return (uint64_t)res;
}

// Unrank the r-th k-combination of {1..N} in lex order.
// Requires 0 <= r < C(N,k). Output is strictly increasing.
static void unrank_comb_lex(int N, int k, uint64_t r, uint8_t *out) {
    int x = 1;
    for (int i = 0; i < k; i++) {
        // choose out[i]
        for (;;) {
            uint64_t c = binom_u64(N - x, (k - 1) - i); // combos if we pick x at position i
            // Explanation: fixing out[i]=x leaves choose k-1-i from {x+1..N} of size N-x
            if (c == 0) c = 0; // (overflow impossible for our target regime)
            if (r < c) {
                out[i] = (uint8_t)x;
                x++;
                break;
            } else {
                r -= c;
                x++;
            }
        }
    }
}

// Next combination in lex order, in-place, for elements in [1..N].
// Returns 1 if advanced, 0 if it was the last combination.
static int next_comb_lex(uint8_t *a, int k, int N) {
    for (int i = k - 1; i >= 0; i--) {
        if (a[i] < (uint8_t)(N - (k - 1 - i))) {
            a[i]++;
            for (int j = i + 1; j < k; j++) a[j] = (uint8_t)(a[j - 1] + 1);
            return 1;
        }
    }
    return 0;
}

/* ============================================================
   Touched vector (sparse output)
   ============================================================ */

typedef struct {
    uint32_t *data;
    size_t len;
    size_t cap;
} U32Vec;

static int u32vec_init(U32Vec *v, size_t cap0) {
    v->len = 0;
    v->cap = cap0 ? cap0 : 1;
    v->data = (uint32_t *)malloc(v->cap * sizeof(uint32_t));
    return v->data ? 0 : -1;
}

static int u32vec_push(U32Vec *v, uint32_t x) {
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

static void u32vec_free(U32Vec *v) {
    free(v->data);
    v->data = NULL;
    v->len = v->cap = 0;
}

// pack (add, mult) into 32 bits. add <= 126, mult <= 4095 for k<=90-ish, but here k small.
static inline uint32_t pack_am(uint16_t add, uint16_t mult) {
    return ((uint32_t)add << 12) | (uint32_t)(mult & 0x0FFFu);
}
static inline void unpack_am(uint32_t key, uint16_t *add, uint16_t *mult) {
    *add  = (uint16_t)(key >> 12);
    *mult = (uint16_t)(key & 0x0FFFu);
}

/* ============================================================
   Per-task binary output
   ============================================================ */

#if defined(__GNUC__) || defined(__clang__)
  #define PACKED __attribute__((packed))
#else
  #define PACKED
#endif

typedef struct PACKED {
    char     magic[8];     // "SPPKBIN\0"
    uint8_t  version;      // 1
    uint8_t  N;
    uint8_t  k;
    uint8_t  reserved;
    uint16_t max_sum;      // 2N
    uint16_t mult_cap;     // C(k+1,2)
    uint32_t record_cnt;
} TaskHdr;

typedef struct PACKED {
    uint16_t add;
    uint16_t mult;
    uint64_t count;
} TaskRec;

/* ============================================================
   Worker: process rank range [r0, r1) for k-subsets of [N]
   ============================================================ */

static int run_task_fixedk(int chunk_id, int N, int k, int total_tasks, const char *tmp_dir) {
    const uint64_t total = binom_u64(N, k);
    if (total == 0) {
        fprintf(stderr, "Task %d: C(%d,%d) overflow/invalid\n", chunk_id, N, k);
        return 2;
    }

    const uint64_t r0 = (total * (uint64_t)chunk_id) / (uint64_t)total_tasks;
    const uint64_t r1 = (total * (uint64_t)(chunk_id + 1)) / (uint64_t)total_tasks;

    const int max_sum = 2 * N;
    const int mult_cap = (k * (k + 1)) / 2;

    // counts[add][mult] only (k fixed)
    const size_t stride = (size_t)(mult_cap + 1);
    const size_t table_sz = (size_t)(max_sum + 1) * stride;
    uint64_t *counts = (uint64_t *)calloc(table_sz, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to alloc counts (%zu)\n", chunk_id, table_sz);
        return 2;
    }

    U32Vec touched;
    if (u32vec_init(&touched, (size_t)1 << 16) != 0) {
        fprintf(stderr, "Task %d: failed to alloc touched\n", chunk_id);
        free(counts);
        return 2;
    }

    // Timestamped mark arrays: avoid clearing.
    // sum values in [2..2N], product values in [1..N^2]
    const int prod_max = N * N;

    uint32_t *sum_mark  = (uint32_t *)calloc((size_t)(max_sum + 1), sizeof(uint32_t));
    uint32_t *prod_mark = (uint32_t *)calloc((size_t)(prod_max + 1), sizeof(uint32_t));
    if (!sum_mark || !prod_mark) {
        fprintf(stderr, "Task %d: failed to alloc mark arrays\n", chunk_id);
        free(sum_mark);
        free(prod_mark);
        u32vec_free(&touched);
        free(counts);
        return 2;
    }

    uint32_t stamp = 1;

    uint8_t comb[64];
    unrank_comb_lex(N, k, r0, comb);

    for (uint64_t r = r0; r < r1; r++) {
        // Compute |A+A|, |A*A| for current combination comb[0..k-1]
        uint16_t add_card = 0;
        uint16_t mult_card = 0;

        stamp++;
        if (stamp == 0) { // wrapped; reset marks (rare)
            memset(sum_mark, 0, (size_t)(max_sum + 1) * sizeof(uint32_t));
            memset(prod_mark, 0, (size_t)(prod_max + 1) * sizeof(uint32_t));
            stamp = 1;
        }

        for (int i = 0; i < k; i++) {
            const int ai = (int)comb[i];
            for (int j = i; j < k; j++) {
                const int aj = (int)comb[j];

                const int s = ai + aj;
                if (sum_mark[s] != stamp) {
                    sum_mark[s] = stamp;
                    add_card++;
                }

                const int p = ai * aj;
                if (prod_mark[p] != stamp) {
                    prod_mark[p] = stamp;
                    mult_card++;
                }
            }
        }

        // mult_card must be <= C(k+1,2)
        if ((int)mult_card > mult_cap) {
            fprintf(stderr, "Task %d: mult_card=%u > mult_cap=%d (impossible)\n", chunk_id, mult_card, mult_cap);
            free(sum_mark); free(prod_mark);
            u32vec_free(&touched);
            free(counts);
            return 2;
        }

        const size_t idx = (size_t)add_card * stride + (size_t)mult_card;
        uint64_t *c = &counts[idx];
        if ((*c)++ == 0) {
            uint32_t key = pack_am(add_card, mult_card);
            if (u32vec_push(&touched, key) != 0) {
                fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                free(sum_mark); free(prod_mark);
                u32vec_free(&touched);
                free(counts);
                return 2;
            }
        }

        if (r + 1 < r1) {
            if (!next_comb_lex(comb, k, N)) break;
        }
    }

    free(sum_mark);
    free(prod_mark);

    // Write task binary (sparse)
    char path[4096];
    snprintf(path, sizeof(path), "%s/tmp_N%02d_%04d.bin", tmp_dir, N, chunk_id + 1);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Task %d: failed to open %s: %s\n", chunk_id, path, strerror(errno));
        u32vec_free(&touched);
        free(counts);
        return 3;
    }

    TaskHdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "SPPKBIN\0", 8);
    hdr.version = 1;
    hdr.N = (uint8_t)N;
    hdr.k = (uint8_t)k;
    hdr.max_sum = (uint16_t)max_sum;
    hdr.mult_cap = (uint16_t)mult_cap;
    hdr.record_cnt = (uint32_t)touched.len;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Task %d: failed to write header %s\n", chunk_id, path);
        fclose(f);
        u32vec_free(&touched);
        free(counts);
        return 3;
    }

    for (size_t i = 0; i < touched.len; i++) {
        uint16_t add, mult;
        unpack_am(touched.data[i], &add, &mult);
        const size_t idx = (size_t)add * stride + (size_t)mult;
        TaskRec rec = { add, mult, counts[idx] };
        if (fwrite(&rec, sizeof(rec), 1, f) != 1) {
            fprintf(stderr, "Task %d: failed to write rec %s\n", chunk_id, path);
            fclose(f);
            u32vec_free(&touched);
            free(counts);
            return 3;
        }
    }

    fclose(f);
    u32vec_free(&touched);
    free(counts);
    return 0;
}

/* ============================================================
   Parent merge: read task files, accumulate into global counts, append to CSV
   ============================================================ */

static int merge_N_to_csv(int N, int k, const char *tmp_dir, int total_tasks, FILE *csv) {
    const int max_sum = 2 * N;
    const int mult_cap = (k * (k + 1)) / 2;
    const size_t stride = (size_t)(mult_cap + 1);
    const size_t table_sz = (size_t)(max_sum + 1) * stride;

    uint64_t *counts = (uint64_t *)calloc(table_sz, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Merge: failed to alloc counts for N=%d\n", N);
        return 2;
    }

    for (int chunk_id = 0; chunk_id < total_tasks; chunk_id++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/tmp_N%02d_%04d.bin", tmp_dir, N, chunk_id + 1);

        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "Merge: missing task file %s\n", path);
            free(counts);
            return 3;
        }

        TaskHdr hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
            fprintf(stderr, "Merge: failed read hdr %s\n", path);
            fclose(f);
            free(counts);
            return 3;
        }

        if (memcmp(hdr.magic, "SPPKBIN\0", 8) != 0 || hdr.version != 1 || hdr.N != (uint8_t)N || hdr.k != (uint8_t)k) {
            fprintf(stderr, "Merge: bad header in %s\n", path);
            fclose(f);
            free(counts);
            return 3;
        }

        for (uint32_t i = 0; i < hdr.record_cnt; i++) {
            TaskRec rec;
            if (fread(&rec, sizeof(rec), 1, f) != 1) {
                fprintf(stderr, "Merge: failed read rec %s\n", path);
                fclose(f);
                free(counts);
                return 3;
            }
            if (rec.add <= (uint16_t)max_sum && rec.mult <= (uint16_t)mult_cap) {
                counts[(size_t)rec.add * stride + (size_t)rec.mult] += rec.count;
            }
        }

        fclose(f);
        // remove temp file to save disk
        unlink(path);
    }

    // Append nonzero bins to CSV
    for (int add = 0; add <= max_sum; add++) {
        const size_t base = (size_t)add * stride;
        for (int mult = 0; mult <= mult_cap; mult++) {
            uint64_t c = counts[base + (size_t)mult];
            if (c) {
                fprintf(csv, "%d,%d,%d,%d,%" PRIu64 "\n", N, k, add, mult, c);
            }
        }
    }

    free(counts);
    return 0;
}

/* ============================================================
   Main: loop N=k..N_max, run tasks in parallel, merge, append to CSV
   ============================================================ */

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <N_max> <k> <out_dir> <jobs> <chunks_per_job>\n", argv[0]);
        return 1;
    }

    const int N_max = atoi(argv[1]);
    const int k = atoi(argv[2]);
    const char *out_dir = argv[3];
    const int jobs = atoi(argv[4]);
    const int chunks_per_job = atoi(argv[5]);

    if (k < 1 || k > 63) { fprintf(stderr, "Error: k must be in [1,63]\n"); return 1; }
    if (N_max < k || N_max > 63) { fprintf(stderr, "Error: N_max must be in [k,63]\n"); return 1; }
    if (jobs < 1) { fprintf(stderr, "Error: jobs must be >= 1\n"); return 1; }
    if (chunks_per_job < 1) { fprintf(stderr, "Error: chunks_per_job must be >= 1\n"); return 1; }

    // Create output dirs
    if (mkdir_p(out_dir) != 0) {
        fprintf(stderr, "Error: could not create out_dir '%s'\n", out_dir);
        return 1;
    }

    // temp directory inside out_dir
    char tmp_dir[4096];
    snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", out_dir);
    if (mkdir_p(tmp_dir) != 0) {
        fprintf(stderr, "Error: could not create tmp dir '%s'\n", tmp_dir);
        return 1;
    }

    // Open (overwrite) the final CSV
    char out_csv_path[4096];
    snprintf(out_csv_path, sizeof(out_csv_path), "%s/spp_k_up_to_n.csv", out_dir);

    FILE *csv = fopen(out_csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Error: cannot open '%s': %s\n", out_csv_path, strerror(errno));
        return 1;
    }
    fprintf(csv, "N,set_cardinality,add_ds_card,mult_ds_card,count\n");

    const int total_tasks = jobs * chunks_per_job;
    const double t0_all = now_seconds();

    pid_t *pids = (pid_t *)calloc((size_t)total_tasks, sizeof(pid_t));
    if (!pids) {
        fprintf(stderr, "Error: allocation failure\n");
        fclose(csv);
        return 1;
    }

    for (int N = k; N <= N_max; N++) {
        const double t0N = now_seconds();
        const uint64_t total = binom_u64(N, k);
        if (total == 0) {
            fprintf(stderr, "Skipping N=%d: C(N,k) overflow/invalid\n", N);
            continue;
        }

        // Launch at most 'jobs' children concurrently for this N.
        int next_chunk = 0, active = 0, done = 0;

        while (done < total_tasks) {
            while (active < jobs && next_chunk < total_tasks) {
                int chunk_id = next_chunk;

                pid_t pid = fork();
                if (pid < 0) {
                    fprintf(stderr, "Error: fork failed at N=%d chunk=%d: %s\n", N, chunk_id, strerror(errno));
                    next_chunk = total_tasks; // stop launching
                    break;
                }
                if (pid == 0) {
                    int rc = run_task_fixedk(chunk_id, N, k, total_tasks, tmp_dir);
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

            int exit_code = 0;
            if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
            else exit_code = 128;

            if (exit_code != 0) {
                fprintf(stderr, "Error: N=%d task failed (exit=%d)\n", N, exit_code);
                // keep going; merge will likely fail if files missing
            }
        }

        // Merge this N into CSV and delete temp files.
        int mrc = merge_N_to_csv(N, k, tmp_dir, total_tasks, csv);
        if (mrc != 0) {
            fprintf(stderr, "Error: merge failed for N=%d (rc=%d)\n", N, mrc);
            free(pids);
            fclose(csv);
            return 2;
        }

        fflush(csv);

        double elapsedN = now_seconds() - t0N;
        double elapsedAll = now_seconds() - t0_all;
        printf("Finished N=%d (C=%" PRIu64 ") in %.2fs (total %.2fs). Appended to %s\n",
               N, total, elapsedN, elapsedAll, out_csv_path);
        fflush(stdout);
    }

    free(pids);
    fclose(csv);
    return 0;
}

