// compute_spp_fixedk_up_to_gmp.c
//
// Enumerate all k-subsets A of [N] (for N=k..N_max) and count pairs (|A+A|, |A*A|).
// Writes a single merged CSV:  <out_dir>/spp_k<k>_up_to_n<N_max>.csv
//
// Usage:
//   ./compute_spp_fixedk_up_to_gmp <N_max> <k> <out_dir> <jobs> <chunks_per_job>
//
// Output CSV columns:
//   N,set_cardinality,add_ds_card,mult_ds_card,count
//
// Key changes vs earlier version:
// (1) Uses GMP mpz_t for C(N,k) and rank partitioning/unranking, so N can exceed 63.
// (2) Output filenames include the actual k and N_max values.
//
// Performance notes:
// - GMP is used only for computing/partitioning ranks and for the initial unrank per task.
// - The main enumeration loop uses a fast lexicographic next-combination iterator.
//
// Compile:
//   gcc -O3 -march=native -flto -fno-plt -std=c11 -Wall -Wextra -Wpedantic -pipe -DNDEBUG \
//       -o compute_spp_fixedk_up_to_gmp compute_spp_fixedk_up_to_gmp.c -lgmp

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

#include <gmp.h>

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
   Combinations: unrank (mpz) + next (fast)
   ============================================================ */

// Next combination in lex order, in-place, for elements in [1..N].
// Returns 1 if advanced, 0 if it was the last combination.
static int next_comb_lex_u16(uint16_t *a, int k, int N) {
    for (int i = k - 1; i >= 0; i--) {
        int limit = N - (k - 1 - i);
        if ((int)a[i] < limit) {
            a[i]++;
            for (int j = i + 1; j < k; j++) a[j] = (uint16_t)(a[j - 1] + 1);
            return 1;
        }
    }
    return 0;
}

// Unrank the r-th k-combination of {1..N} in lex order, where r is an mpz in [0, C(N,k)).
// Output is strictly increasing; stored as uint16_t (so N must be <= 65535).
static void unrank_comb_lex_mpz(int N, int k, const mpz_t r_in, uint16_t *out) {
    mpz_t r, c;
    mpz_init_set(r, r_in);
    mpz_init(c);

    int x = 1;
    for (int i = 0; i < k; i++) {
        for (;;) {
            // c = C(N-x, (k-1)-i)
            unsigned nn = (unsigned)(N - x);
            unsigned kk = (unsigned)((k - 1) - i);
            mpz_bin_uiui(c, nn, kk);

            if (mpz_cmp(r, c) < 0) {
                out[i] = (uint16_t)x;
                x++;
                break;
            } else {
                mpz_sub(r, r, c);
                x++;
            }
        }
    }

    mpz_clear(r);
    mpz_clear(c);
}

/* ============================================================
   Sparse touched-bins vector
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

// pack (add, mult) into 32 bits. add <= 2N, mult <= C(k+1,2) (fits for modest k).
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
    uint16_t N;
    uint16_t k;
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
    // total = C(N,k)
    mpz_t total, r0, r1, tmp;
    mpz_init(total);
    mpz_init(r0);
    mpz_init(r1);
    mpz_init(tmp);

    mpz_bin_uiui(total, (unsigned)N, (unsigned)k);

    // r0 = floor(total * chunk_id / total_tasks)
    mpz_mul_ui(tmp, total, (unsigned long)chunk_id);
    mpz_fdiv_q_ui(r0, tmp, (unsigned long)total_tasks);

    // r1 = floor(total * (chunk_id+1) / total_tasks)
    mpz_mul_ui(tmp, total, (unsigned long)(chunk_id + 1));
    mpz_fdiv_q_ui(r1, tmp, (unsigned long)total_tasks);

    // chunk_len = r1 - r0 must fit into uint64 for feasible looping
    mpz_sub(tmp, r1, r0);
    if (!mpz_fits_ulong_p(tmp)) {
        fprintf(stderr, "Task %d: chunk length too large for this implementation.\n", chunk_id);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 2;
    }
    uint64_t chunk_len = (uint64_t)mpz_get_ui(tmp);

    const int max_sum  = 2 * N;
    const int mult_cap = (k * (k + 1)) / 2;

    // counts[add][mult] only (k fixed)
    const size_t stride   = (size_t)(mult_cap + 1);
    const size_t table_sz = (size_t)(max_sum + 1) * stride;
    uint64_t *counts = (uint64_t *)calloc(table_sz, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to alloc counts (%zu)\n", chunk_id, table_sz);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 2;
    }

    U32Vec touched;
    if (u32vec_init(&touched, (size_t)1 << 16) != 0) {
        fprintf(stderr, "Task %d: failed to alloc touched\n", chunk_id);
        free(counts);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 2;
    }

    // Timestamped mark arrays (avoid clearing each subset)
    const int prod_max = N * N;
    uint32_t *sum_mark  = (uint32_t *)calloc((size_t)(max_sum + 1), sizeof(uint32_t));
    uint32_t *prod_mark = (uint32_t *)calloc((size_t)(prod_max + 1), sizeof(uint32_t));
    if (!sum_mark || !prod_mark) {
        fprintf(stderr, "Task %d: failed to alloc mark arrays\n", chunk_id);
        free(sum_mark); free(prod_mark);
        u32vec_free(&touched);
        free(counts);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 2;
    }
    uint32_t stamp = 1;

    // Initialize combination at rank r0 (mpz)
    uint16_t *comb = (uint16_t *)malloc((size_t)k * sizeof(uint16_t));
    if (!comb) {
        fprintf(stderr, "Task %d: failed to alloc comb\n", chunk_id);
        free(sum_mark); free(prod_mark);
        u32vec_free(&touched);
        free(counts);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 2;
    }
    unrank_comb_lex_mpz(N, k, r0, comb);

    // Main loop over this chunk using fast next_comb (no GMP)
    for (uint64_t it = 0; it < chunk_len; it++) {
        uint16_t add_card = 0;
        uint16_t mult_card = 0;

        stamp++;
        if (stamp == 0) {
            memset(sum_mark, 0, (size_t)(max_sum + 1) * sizeof(uint32_t));
            memset(prod_mark, 0, (size_t)(prod_max + 1) * sizeof(uint32_t));
            stamp = 1;
        }

        for (int i = 0; i < k; i++) {
            int ai = (int)comb[i];
            for (int j = i; j < k; j++) {
                int aj = (int)comb[j];

                int s = ai + aj;
                if (sum_mark[s] != stamp) { sum_mark[s] = stamp; add_card++; }

                int p = ai * aj;
                if (prod_mark[p] != stamp) { prod_mark[p] = stamp; mult_card++; }
            }
        }

        const size_t idx = (size_t)add_card * stride + (size_t)mult_card;
        uint64_t *c = &counts[idx];
        if ((*c)++ == 0) {
            uint32_t key = pack_am(add_card, mult_card);
            if (u32vec_push(&touched, key) != 0) {
                fprintf(stderr, "Task %d: touched push failed\n", chunk_id);
                free(comb);
                free(sum_mark); free(prod_mark);
                u32vec_free(&touched);
                free(counts);
                mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
                return 2;
            }
        }

        if (it + 1 < chunk_len) {
            if (!next_comb_lex_u16(comb, k, N)) break;
        }
    }

    free(comb);
    free(sum_mark);
    free(prod_mark);

    // Write sparse task binary
    char path[4096];
    snprintf(path, sizeof(path), "%s/tmp_k%d_N%05d_%04d.bin", tmp_dir, k, N, chunk_id + 1);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Task %d: failed to open %s: %s\n", chunk_id, path, strerror(errno));
        u32vec_free(&touched);
        free(counts);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
        return 3;
    }

    TaskHdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "SPPKBIN\0", 8);
    hdr.version   = 1;
    hdr.N         = (uint16_t)N;
    hdr.k         = (uint16_t)k;
    hdr.max_sum   = (uint16_t)max_sum;
    hdr.mult_cap  = (uint16_t)mult_cap;
    hdr.record_cnt = (uint32_t)touched.len;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Task %d: failed to write header %s\n", chunk_id, path);
        fclose(f);
        u32vec_free(&touched);
        free(counts);
        mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
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
            mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
            return 3;
        }
    }

    fclose(f);
    u32vec_free(&touched);
    free(counts);

    mpz_clear(total); mpz_clear(r0); mpz_clear(r1); mpz_clear(tmp);
    return 0;
}

/* ============================================================
   Parent merge: read task files, accumulate into global counts, append to CSV
   ============================================================ */

static int merge_N_to_csv(int N, int k, const char *tmp_dir, int total_tasks, FILE *csv) {
    const int max_sum  = 2 * N;
    const int mult_cap = (k * (k + 1)) / 2;
    const size_t stride   = (size_t)(mult_cap + 1);
    const size_t table_sz = (size_t)(max_sum + 1) * stride;

    uint64_t *counts = (uint64_t *)calloc(table_sz, sizeof(uint64_t));
    if (!counts) {
        fprintf(stderr, "Merge: failed to alloc counts for N=%d\n", N);
        return 2;
    }

    for (int chunk_id = 0; chunk_id < total_tasks; chunk_id++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/tmp_k%d_N%05d_%04d.bin", tmp_dir, k, N, chunk_id + 1);

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

        if (memcmp(hdr.magic, "SPPKBIN\0", 8) != 0 || hdr.version != 1 ||
            hdr.N != (uint16_t)N || hdr.k != (uint16_t)k) {
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
        unlink(path); // delete temp to save disk
    }

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
   Main: loop N=k..N_max, run tasks, merge, append to CSV
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

    if (k < 1) { fprintf(stderr, "Error: k must be >= 1\n"); return 1; }
    if (N_max < k) { fprintf(stderr, "Error: N_max must be >= k\n"); return 1; }
    if (N_max > 65535) { fprintf(stderr, "Error: N_max too large (must be <= 65535)\n"); return 1; }
    if (jobs < 1) { fprintf(stderr, "Error: jobs must be >= 1\n"); return 1; }
    if (chunks_per_job < 1) { fprintf(stderr, "Error: chunks_per_job must be >= 1\n"); return 1; }

    if (mkdir_p(out_dir) != 0) {
        fprintf(stderr, "Error: could not create out_dir '%s'\n", out_dir);
        return 1;
    }

    // Put outputs under out_dir/spp_k<k>/
    char out_subdir[4096];
    snprintf(out_subdir, sizeof(out_subdir), "%s/spp_k%d", out_dir, k);
    if (mkdir_p(out_subdir) != 0) {
        fprintf(stderr, "Error: could not create '%s'\n", out_subdir);
        return 1;
    }

    // temp directory inside that
    char tmp_dir[4096];
    snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp_k%d", out_subdir, k);
    if (mkdir_p(tmp_dir) != 0) {
        fprintf(stderr, "Error: could not create tmp dir '%s'\n", tmp_dir);
        return 1;
    }

    // Output CSV name includes actual k and N_max
    char out_csv_path[4096];
    snprintf(out_csv_path, sizeof(out_csv_path), "%s/spp_k%d_up_to_n%d.csv", out_subdir, k, N_max);

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

        int next_chunk = 0, active = 0, done = 0;

        while (done < total_tasks) {
            while (active < jobs && next_chunk < total_tasks) {
                int chunk_id = next_chunk;

                pid_t pid = fork();
                if (pid < 0) {
                    fprintf(stderr, "Error: fork failed at N=%d chunk=%d: %s\n", N, chunk_id, strerror(errno));
                    next_chunk = total_tasks;
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
            }
        }

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

        // compute total = C(N,k) for reporting
        mpz_t total;
        mpz_init(total);
        mpz_bin_uiui(total, (unsigned)N, (unsigned)k);

        // print total in decimal
        char *tot_str = mpz_get_str(NULL, 10, total);
        printf("Finished N=%d (C=%s) in %.2fs (total %.2fs). Appended to %s\n",
               N, tot_str, elapsedN, elapsedAll, out_csv_path);
        fflush(stdout);

        free(tot_str);
        mpz_clear(total);
    }

    free(pids);
    fclose(csv);
    return 0;
}

