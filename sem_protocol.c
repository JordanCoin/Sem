#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define OUT_DIR "generated"
#define OUT_FILE "generated/workspace.sem"
#define BUF_SIZE 4096

static int write_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, buf + off, len - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}


static int mkdir_if_missing(const char *path) {
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static int create_sem_file(const char *path) {
    const char *content =
        "#@ {\"type\":\"header\",\"sem_version\":\"0.1\",\"space\":\"R3\",\"units\":\"arb\",\"created_at\":\"2026-02-20T00:00:00Z\"}\n"
        "\n"
        "v 0 0 0\n"
        "#@ {\"type\":\"belief\",\"id\":\"b1\",\"vertex\":1,\"proposition\":\"Belief A\",\"confidence_base\":0.9,\"updated_at\":\"2026-02-20T00:00:00Z\"}\n"
        "\n"
        "v 1 0 0\n"
        "#@ {\"type\":\"belief\",\"id\":\"b2\",\"vertex\":2,\"proposition\":\"Belief B\",\"confidence_base\":0.8,\"updated_at\":\"2026-02-20T00:00:00Z\"}\n"
        "\n"
        "v 0 1 0\n"
        "#@ {\"type\":\"belief\",\"id\":\"b3\",\"vertex\":3,\"proposition\":\"Belief C\",\"confidence_base\":0.7,\"updated_at\":\"2026-02-20T00:00:00Z\"}\n"
        "\n"
        "f 1 2 3\n"
        "#@ {\"type\":\"triangle\",\"id\":\"t1\",\"face\":1,\"rest\":{\"edge_lengths\":[1,1,1.41421356]},\"physics\":{\"stiffness\":1.0,\"damping\":0.25},\"updated_at\":\"2026-02-20T00:00:00Z\"}\n"
        "\n"
        "#@ {\"type\":\"request\",\"id\":\"r1\",\"name\":\"Summarize triangle tensions\",\"input\":{\"select\":{\"mode\":\"triangle\",\"triangle_id\":\"t1\"}},\"agent\":{\"interface\":\"generic\",\"instruction\":\"Explain tensions and propose edits.\",\"output_format\":\"patch_v0\"},\"apply\":{\"mode\":\"suggest\",\"target\":\"workspace\"}}\n";

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    int rc = write_all(fd, content, strlen(content));
    if (close(fd) < 0) return -1;
    return rc;
}

static void consume_line(const char *line, size_t len, int *vertices, int *faces, int *records, bool *has_header) {
    if (len >= 2 && line[0] == 'v' && line[1] == ' ') {
        (*vertices)++;
    } else if (len >= 2 && line[0] == 'f' && line[1] == ' ') {
        (*faces)++;
    } else if (len >= 2 && line[0] == '#' && line[1] == '@') {
        const char needle[] = "\"type\":\"header\"";
        (*records)++;
        for (size_t i = 0; i + sizeof(needle) - 1 <= len; i++) {
            if (memcmp(line + i, needle, sizeof(needle) - 1) == 0) {
                *has_header = true;
                break;
            }
        }
    }
}

static int open_and_validate(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    char buf[BUF_SIZE];
    char line[BUF_SIZE];
    size_t line_len = 0;
    int vertices = 0;
    int faces = 0;
    int records = 0;
    bool has_header = false;

    for (;;) {
        ssize_t r = read(fd, buf, sizeof(buf));
        if (r < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return -1;
        }
        if (r == 0) break;

        for (ssize_t i = 0; i < r; i++) {
            char c = buf[i];
            if (c == '\n') {
                consume_line(line, line_len, &vertices, &faces, &records, &has_header);
                line_len = 0;
            } else if (line_len < sizeof(line) - 1) {
                line[line_len++] = c;
            }
        }
    }

    if (line_len > 0) {
        consume_line(line, line_len, &vertices, &faces, &records, &has_header);
    }

    if (close(fd) < 0) return -1;

    if (!has_header || vertices < 3 || faces < 1 || records < 5) {
        errno = EPROTO;
        return -1;
    }

    char out[256];
    int n = snprintf(out,
                     sizeof(out),
                     "Opened %s successfully. vertices=%d faces=%d semantic_records=%d header=%s\n",
                     path,
                     vertices,
                     faces,
                     records,
                     has_header ? "yes" : "no");
    if (n < 0 || (size_t)n >= sizeof(out)) {
        errno = EOVERFLOW;
        return -1;
    }
    return write_all(STDOUT_FILENO, out, (size_t)n);
}

int main(void) {
    if (mkdir_if_missing(OUT_DIR) != 0) {
        char msg[128];
        int n = snprintf(msg, sizeof(msg), "mkdir failed: %s\n", strerror(errno));
        if (n > 0) write_all(STDERR_FILENO, msg, (size_t)n);
        return 1;
    }

    if (create_sem_file(OUT_FILE) != 0) {
        char msg[128];
        int n = snprintf(msg, sizeof(msg), "write failed: %s\n", strerror(errno));
        if (n > 0) write_all(STDERR_FILENO, msg, (size_t)n);
        return 1;
    }

    if (open_and_validate(OUT_FILE) != 0) {
        char msg[128];
        int n = snprintf(msg, sizeof(msg), "open/validate failed: %s\n", strerror(errno));
        if (n > 0) write_all(STDERR_FILENO, msg, (size_t)n);
        return 1;
    }

    return 0;
}
