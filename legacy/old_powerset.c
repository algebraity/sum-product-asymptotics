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
#else
static inline int POPCNT64(uint64_t x) {
    int c = 0;
    while (x) { x &= (x - 1); c++; }
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

// Compute |A+A| and |A*A| for A ⊆ [n] given by mask.
static inline void compute_add_mult_cards(uint64_t mask, int n, int *out_add, int *out_mult) {
    // Build A_bits: bit x set iff x ∈ A (x in 1..n). Bit 0 unused.
    // For n<=63, this fits in uint64_t.
    uint64_t A_bits = (mask << 1);

    // ---- sums via shifts: sums_bits has bits at indices in [2..2n] ----
    // 2n <= 126, so store sums in two 64-bit words.
    uint64_t sums0 = 0ULL; // bits 0..63
    uint64_t sums1 = 0ULL; // bits 64..127

    // Iterate elements a in A by scanning mask bits.
    uint64_t mm = mask;
    while (mm) {
        unsigned i = (unsigned)__builtin_ctzll(mm); // i in [0..n-1], element a=i+1
        unsigned a = i + 1;
        mm &= (mm - 1);

        // We want sums_bits |= (A_bits << a), but A_bits is 64-bit and we need 128-bit result.
        // Compute 128-bit shift manually into (sums0,sums1).
        if (a < 64) {
            uint64_t lo = A_bits << a;
            uint64_t hi = (a == 0) ? 0ULL : (A_bits >> (64 - a));
            sums0 |= lo;
            sums1 |= hi;
        } else {
            // a is at most 63 here since a<=n<=63, so this is rarely used but keep correct.
            uint64_t hi = A_bits << (a - 64);
            sums1 |= hi;
        }
    }

    // Mask out bits outside [2..2n] (optional hygiene; popcount is fast anyway).
    // You can omit this if you trust unused bits won’t be set (they can be set from shifts).
    const int max_sum = 2 * n;
    // Clear bits > max_sum
    if (max_sum < 64) {
        uint64_t keep = (max_sum == 63) ? ~0ULL : ((1ULL << (max_sum + 1)) - 1ULL);
        sums0 &= keep;
        sums1 = 0ULL;
    } else {
        int hi_max = max_sum - 64;
        uint64_t keep1 = (hi_max == 63) ? ~0ULL : ((1ULL << (hi_max + 1)) - 1ULL);
        sums1 &= keep1;
    }
    // Clear bits < 2
    sums0 &= ~((1ULL << 2) - 1ULL);

    int add_card = POPCNT64(sums0) + POPCNT64(sums1);

    // ---- products via unordered pairs (i<=j), no heap allocations ----
    const int max_prod  = n * n;
    const int prod_words = (max_prod + 64) / 64; // bits up to max_prod inclusive

    uint64_t prod_bits[63]; // enough for n<=63 (max_prod<=3969 => prod_words<=63)
    for (int w = 0; w < prod_words; w++) prod_bits[w] = 0ULL;

    // Collect elements into array a[0..m-1] quickly by scanning mask bits.
    uint32_t a_arr[63];
    int m = 0;
    mm = mask;
    while (mm) {
        unsigned i = (unsigned)__builtin_ctzll(mm);
        a_arr[m++] = (uint32_t)(i + 1);
        mm &= (mm - 1);
    }

    for (int i = 0; i < m; i++) {
        uint32_t ai = a_arr[i];
        for (int j = i; j < m; j++) {
            uint32_t p = ai * a_arr[j];
            prod_bits[p >> 6] |= 1ULL << (p & 63);
        }
    }

    int mult_card = 0;
    for (int w = 0; w < prod_words; w++) mult_card += POPCNT64(prod_bits[w]);

    *out_add  = add_card;
    *out_mult = mult_card;
}


static int run_task(int chunk_id, int n, int total_tasks, const char *out_dir) {
    // total = 1<<n (n<=63)
    const uint64_t total = (n == 64) ? 0ULL : (1ULL << n);

    const int max_sum  = 2 * n;
    const int max_prod = n * n;

    // Dense distribution table:
    // counts[(add)*(max_prod+1) + mult] is number of subsets with that (add,mult).
    const size_t stride = (size_t)(max_prod + 1);
    const size_t table_size = (size_t)(max_sum + 1) * stride;

    uint32_t *counts = (uint32_t *)calloc(table_size, sizeof(uint32_t));
    if (!counts) {
        fprintf(stderr, "Task %d: failed to allocate counts table (%zu entries)\n", chunk_id, table_size);
        return 2;
    }

    int add_card = 0, mult_card = 0;

    for (uint64_t mask = (uint64_t)chunk_id; mask < total; mask += (uint64_t)total_tasks) {
        if (mask == 0) continue;

        compute_add_mult_cards(mask, n, &add_card, &mult_card);

        // Safety bounds (should always hold)
        if (add_card < 0) add_card = 0;
        if (add_card > max_sum) add_card = max_sum;
        if (mult_card < 0) mult_card = 0;
        if (mult_card > max_prod) mult_card = max_prod;

        counts[(size_t)add_card * stride + (size_t)mult_card] += 1U;
    }

    // Write CSV
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

    // Spawn total_tasks children (mirrors the Python "tasks" list closely)
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
            // Parent: stop spawning; wait for already-spawned children
//            total_tasks; // no-op
            break;
        }
        if (pid == 0) {
            // Child: run task then exit
            int rc = run_task(chunk_id, n, total_tasks, out_dir);
            _exit(rc);
        }
        // Parent
        pids[chunk_id] = pid;
        chunk_of_pid[chunk_id] = chunk_id;
    }

    // Wait for all children, report progress
    int done = 0;
    for (;;) {
        int status = 0;
        pid_t pid = wait(&status);
        if (pid < 0) {
            if (errno == EINTR) continue;
            break; // no more children
        }

        // Find chunk_id (linear search is fine at this scale)
        int chunk_id = -1;
        for (int i = 0; i < total_tasks; i++) {
            if (pids[i] == pid) { chunk_id = chunk_of_pid[i]; break; }
        }

        done += 1;

        int exit_code = 0;
        if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
        else exit_code = 128; // abnormal

        // Match the Python-style print format closely
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

