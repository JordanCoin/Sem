#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_BELIEFS 256
#define MAX_TRIANGLES 256
#define MAX_REQUESTS 128
#define MAX_VERTICES 512
#define MAX_FACES 512
#define BUF_SZ 65536
#define LISTEN_PORT 7318
#define SESSION_ID "s_1"
#define ALPHA 1.0

typedef struct {
    double x, y, z;
} Vertex;

typedef struct {
    int a, b, c;
} Face;

typedef struct {
    char id[64];
    int vertex;
    char proposition[256];
    double confidence_base;
    double confidence_effective;
} Belief;

typedef struct {
    char id[64];
    int face;
    int vertices[3];
    double rest[3];
    double stiffness;
    double damping;
    double strain;
} Triangle;

typedef struct {
    char id[64];
    char name[128];
    char triangle_id[64];
} Request;

typedef struct {
    char open_path[512];
    bool loaded;
    Vertex vertices[MAX_VERTICES];
    int vertex_count;
    Face faces[MAX_FACES];
    int face_count;
    Belief beliefs[MAX_BELIEFS];
    int belief_count;
    Triangle triangles[MAX_TRIANGLES];
    int triangle_count;
    Request requests[MAX_REQUESTS];
    int request_count;
} Workspace;

static Workspace g_ws;

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

static void trim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || isspace((unsigned char)s[n - 1]))) {
        s[n - 1] = '\0';
        n--;
    }
}

static bool json_get_string(const char *json, const char *key, char *out, size_t out_sz) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    size_t i = 0;
    while (p[i] && p[i] != '"' && i + 1 < out_sz) {
        out[i] = p[i];
        i++;
    }
    out[i] = '\0';
    return i > 0;
}

static bool json_get_int(const char *json, const char *key, int *out) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ' || *p == '\t') p++;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return false;
    *out = (int)v;
    return true;
}

static bool json_get_double(const char *json, const char *key, double *out) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ' || *p == '\t') p++;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return false;
    *out = v;
    return true;
}

static bool json_get_position(const char *json, double *x, double *y, double *z) {
    const char *p = strstr(json, "\"position\":[");
    if (!p) return false;
    p = strchr(p, '[');
    if (!p) return false;
    p++;
    char *e = NULL;
    *x = strtod(p, &e);
    if (e == p) return false;
    p = e + 1;
    *y = strtod(p, &e);
    if (e == p) return false;
    p = e + 1;
    *z = strtod(p, &e);
    if (e == p) return false;
    return true;
}

static bool json_get_vertices3(const char *json, int out[3]) {
    const char *p = strstr(json, "\"vertices\":[");
    if (!p) return false;
    p = strchr(p, '[');
    if (!p) return false;
    p++;
    char *e = NULL;
    out[0] = (int)strtol(p, &e, 10);
    if (e == p) return false;
    p = e + 1;
    out[1] = (int)strtol(p, &e, 10);
    if (e == p) return false;
    p = e + 1;
    out[2] = (int)strtol(p, &e, 10);
    if (e == p) return false;
    return true;
}

static double dist3(Vertex a, Vertex b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static void recompute_strain_and_confidence(void) {
    for (int i = 0; i < g_ws.triangle_count; i++) {
        Triangle *t = &g_ws.triangles[i];
        int a = t->vertices[0] - 1;
        int b = t->vertices[1] - 1;
        int c = t->vertices[2] - 1;
        if (a < 0 || b < 0 || c < 0 || a >= g_ws.vertex_count || b >= g_ws.vertex_count || c >= g_ws.vertex_count) {
            t->strain = 0.0;
            continue;
        }
        double dab = dist3(g_ws.vertices[a], g_ws.vertices[b]);
        double dbc = dist3(g_ws.vertices[b], g_ws.vertices[c]);
        double dca = dist3(g_ws.vertices[c], g_ws.vertices[a]);
        double eps = 1e-9;
        double sab = (dab - t->rest[0]) / fmax(t->rest[0], eps);
        double sbc = (dbc - t->rest[1]) / fmax(t->rest[1], eps);
        double sca = (dca - t->rest[2]) / fmax(t->rest[2], eps);
        t->strain = t->stiffness * (sab * sab + sbc * sbc + sca * sca);
    }

    for (int i = 0; i < g_ws.belief_count; i++) {
        Belief *b = &g_ws.beliefs[i];
        double acc = 0.0;
        int n = 0;
        for (int j = 0; j < g_ws.triangle_count; j++) {
            Triangle *t = &g_ws.triangles[j];
            if (t->vertices[0] == b->vertex || t->vertices[1] == b->vertex || t->vertices[2] == b->vertex) {
                acc += t->strain;
                n++;
            }
        }
        double s = (n > 0) ? (acc / (double)n) : 0.0;
        b->confidence_effective = b->confidence_base * exp(-ALPHA * s);
    }
}

static void clear_workspace(void) {
    memset(&g_ws, 0, sizeof(g_ws));
}

static int parse_sem_file(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    clear_workspace();
    snprintf(g_ws.open_path, sizeof(g_ws.open_path), "%s", path);

    char line[4096];
    size_t llen = 0;
    char ch;
    while (read(fd, &ch, 1) == 1) {
        if (ch != '\n' && llen + 1 < sizeof(line)) {
            line[llen++] = ch;
            continue;
        }
        line[llen] = '\0';
        trim(line);

        if (strncmp(line, "v ", 2) == 0 && g_ws.vertex_count < MAX_VERTICES) {
            Vertex *v = &g_ws.vertices[g_ws.vertex_count++];
            sscanf(line + 2, "%lf %lf %lf", &v->x, &v->y, &v->z);
        } else if (strncmp(line, "f ", 2) == 0 && g_ws.face_count < MAX_FACES) {
            Face *f = &g_ws.faces[g_ws.face_count++];
            sscanf(line + 2, "%d %d %d", &f->a, &f->b, &f->c);
        } else if (strncmp(line, "#@", 2) == 0) {
            if (strstr(line, "\"type\":\"belief\"") && g_ws.belief_count < MAX_BELIEFS) {
                Belief *b = &g_ws.beliefs[g_ws.belief_count++];
                json_get_string(line, "id", b->id, sizeof(b->id));
                json_get_int(line, "vertex", &b->vertex);
                json_get_string(line, "proposition", b->proposition, sizeof(b->proposition));
                json_get_double(line, "confidence_base", &b->confidence_base);
                b->confidence_effective = b->confidence_base;
            } else if (strstr(line, "\"type\":\"triangle\"") && g_ws.triangle_count < MAX_TRIANGLES) {
                Triangle *t = &g_ws.triangles[g_ws.triangle_count++];
                memset(t, 0, sizeof(*t));
                json_get_string(line, "id", t->id, sizeof(t->id));
                json_get_int(line, "face", &t->face);
                t->stiffness = 1.0;
                t->damping = 0.25;
                json_get_double(line, "stiffness", &t->stiffness);
                json_get_double(line, "damping", &t->damping);

                const char *elist = strstr(line, "\"edge_lengths\":[");
                if (elist) {
                    const char *p = strchr(elist, '[');
                    if (p) {
                        p++;
                        char *e = NULL;
                        t->rest[0] = strtod(p, &e);
                        if (e) p = e + 1;
                        t->rest[1] = strtod(p, &e);
                        if (e) p = e + 1;
                        t->rest[2] = strtod(p, &e);
                    }
                }
                if (t->face > 0 && t->face <= g_ws.face_count) {
                    Face *f = &g_ws.faces[t->face - 1];
                    t->vertices[0] = f->a;
                    t->vertices[1] = f->b;
                    t->vertices[2] = f->c;
                }
            } else if (strstr(line, "\"type\":\"request\"") && g_ws.request_count < MAX_REQUESTS) {
                Request *r = &g_ws.requests[g_ws.request_count++];
                json_get_string(line, "id", r->id, sizeof(r->id));
                json_get_string(line, "name", r->name, sizeof(r->name));
                json_get_string(line, "triangle_id", r->triangle_id, sizeof(r->triangle_id));
            }
        }

        llen = 0;
    }

    close(fd);

    for (int i = 0; i < g_ws.triangle_count; i++) {
        Triangle *t = &g_ws.triangles[i];
        if ((t->rest[0] == 0.0 && t->rest[1] == 0.0 && t->rest[2] == 0.0) && t->vertices[0] > 0 && t->vertices[1] > 0 && t->vertices[2] > 0) {
            Vertex a = g_ws.vertices[t->vertices[0] - 1];
            Vertex b = g_ws.vertices[t->vertices[1] - 1];
            Vertex c = g_ws.vertices[t->vertices[2] - 1];
            t->rest[0] = dist3(a, b);
            t->rest[1] = dist3(b, c);
            t->rest[2] = dist3(c, a);
        }
    }

    g_ws.loaded = true;
    recompute_strain_and_confidence();
    return 0;
}

static int save_sem_file(const char *path) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;

    const char *header = "#@ {\"type\":\"header\",\"sem_version\":\"0.1\",\"space\":\"R3\",\"units\":\"arb\",\"created_at\":\"2026-02-20T00:00:00Z\"}\n\n";
    if (write_all(fd, header, strlen(header)) != 0) {
        close(fd);
        return -1;
    }

    char line[1024];
    for (int i = 0; i < g_ws.vertex_count; i++) {
        int n = snprintf(line, sizeof(line), "v %.6f %.6f %.6f\n", g_ws.vertices[i].x, g_ws.vertices[i].y, g_ws.vertices[i].z);
        if (write_all(fd, line, (size_t)n) != 0) { close(fd); return -1; }
        for (int j = 0; j < g_ws.belief_count; j++) {
            Belief *b = &g_ws.beliefs[j];
            if (b->vertex == i + 1) {
                n = snprintf(line, sizeof(line), "#@ {\"type\":\"belief\",\"id\":\"%s\",\"vertex\":%d,\"proposition\":\"%s\",\"confidence_base\":%.6f,\"updated_at\":\"2026-02-20T00:00:00Z\"}\n\n", b->id, b->vertex, b->proposition, b->confidence_base);
                if (write_all(fd, line, (size_t)n) != 0) { close(fd); return -1; }
            }
        }
    }

    for (int i = 0; i < g_ws.face_count; i++) {
        int n = snprintf(line, sizeof(line), "f %d %d %d\n", g_ws.faces[i].a, g_ws.faces[i].b, g_ws.faces[i].c);
        if (write_all(fd, line, (size_t)n) != 0) { close(fd); return -1; }
        for (int j = 0; j < g_ws.triangle_count; j++) {
            Triangle *t = &g_ws.triangles[j];
            if (t->face == i + 1) {
                n = snprintf(line, sizeof(line), "#@ {\"type\":\"triangle\",\"id\":\"%s\",\"face\":%d,\"rest\":{\"edge_lengths\":[%.6f,%.6f,%.6f]},\"physics\":{\"stiffness\":%.6f,\"damping\":%.6f},\"updated_at\":\"2026-02-20T00:00:00Z\"}\n\n", t->id, t->face, t->rest[0], t->rest[1], t->rest[2], t->stiffness, t->damping);
                if (write_all(fd, line, (size_t)n) != 0) { close(fd); return -1; }
            }
        }
    }

    for (int i = 0; i < g_ws.request_count; i++) {
        Request *r = &g_ws.requests[i];
        int n = snprintf(line, sizeof(line), "#@ {\"type\":\"request\",\"id\":\"%s\",\"name\":\"%s\",\"input\":{\"select\":{\"mode\":\"triangle\",\"triangle_id\":\"%s\"}},\"agent\":{\"interface\":\"generic\",\"instruction\":\"Explain tensions and propose edits.\",\"output_format\":\"patch_v0\"},\"apply\":{\"mode\":\"suggest\",\"target\":\"workspace\"}}\n", r->id, r->name, r->triangle_id);
        if (write_all(fd, line, (size_t)n) != 0) { close(fd); return -1; }
    }

    close(fd);
    return 0;
}

static Belief *find_belief(const char *id) {
    for (int i = 0; i < g_ws.belief_count; i++) {
        if (strcmp(g_ws.beliefs[i].id, id) == 0) return &g_ws.beliefs[i];
    }
    return NULL;
}

static Request *find_request(const char *id) {
    for (int i = 0; i < g_ws.request_count; i++) {
        if (strcmp(g_ws.requests[i].id, id) == 0) return &g_ws.requests[i];
    }
    return NULL;
}

static void http_send_json(int cfd, int code, const char *json) {
    char hdr[512];
    const char *msg = (code == 200) ? "OK" : (code == 404 ? "Not Found" : "Bad Request");
    int n = snprintf(hdr, sizeof(hdr), "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code, msg, strlen(json));
    write_all(cfd, hdr, (size_t)n);
    write_all(cfd, json, strlen(json));
}

static bool ensure_loaded(int cfd) {
    if (!g_ws.loaded) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"no open session\"}");
        return false;
    }
    return true;
}

static void handle_open(int cfd, const char *body) {
    char path[512] = {0};
    if (!json_get_string(body, "path", path, sizeof(path))) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"path required\"}");
        return;
    }
    if (parse_sem_file(path) != 0) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"cannot open file\"}");
        return;
    }
    http_send_json(cfd, 200, "{\"session_id\":\"s_1\",\"sem_version\":\"0.1\"}");
}

static void handle_save(int cfd) {
    if (!ensure_loaded(cfd)) return;
    if (save_sem_file(g_ws.open_path) != 0) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"save failed\"}");
        return;
    }
    char out[768];
    snprintf(out, sizeof(out), "{\"ok\":true,\"written_path\":\"%s\"}", g_ws.open_path);
    http_send_json(cfd, 200, out);
}

static void handle_validate(int cfd) {
    if (!ensure_loaded(cfd)) return;
    bool ok = (g_ws.vertex_count > 0 && g_ws.belief_count > 0 && g_ws.triangle_count > 0);
    for (int i = 0; i < g_ws.belief_count; i++) {
        if (g_ws.beliefs[i].vertex < 1 || g_ws.beliefs[i].vertex > g_ws.vertex_count) ok = false;
    }
    char out[256];
    snprintf(out, sizeof(out), "{\"ok\":%s,\"warnings\":[],\"errors\":[]}", ok ? "true" : "false");
    http_send_json(cfd, ok ? 200 : 400, out);
}

static void handle_beliefs_list(int cfd) {
    if (!ensure_loaded(cfd)) return;
    recompute_strain_and_confidence();
    char out[BUF_SZ];
    size_t off = 0;
    off += snprintf(out + off, sizeof(out) - off, "{\"items\":[");
    for (int i = 0; i < g_ws.belief_count; i++) {
        Belief *b = &g_ws.beliefs[i];
        Vertex v = g_ws.vertices[b->vertex - 1];
        off += snprintf(out + off,
                        sizeof(out) - off,
                        "%s{\"id\":\"%s\",\"vertex\":%d,\"position\":[%.6f,%.6f,%.6f],\"proposition\":\"%s\",\"confidence_base\":%.6f,\"confidence_effective\":%.6f}",
                        i ? "," : "",
                        b->id,
                        b->vertex,
                        v.x,
                        v.y,
                        v.z,
                        b->proposition,
                        b->confidence_base,
                        b->confidence_effective);
    }
    off += snprintf(out + off, sizeof(out) - off, "]}");
    http_send_json(cfd, 200, out);
}

static void handle_belief_patch(int cfd, const char *belief_id, const char *body) {
    if (!ensure_loaded(cfd)) return;
    Belief *b = find_belief(belief_id);
    if (!b) { http_send_json(cfd, 404, "{\"ok\":false,\"error\":\"belief not found\"}"); return; }

    char prop[256];
    double conf;
    if (json_get_string(body, "proposition", prop, sizeof(prop))) {
        snprintf(b->proposition, sizeof(b->proposition), "%s", prop);
    }
    if (json_get_double(body, "confidence_base", &conf)) {
        if (conf < 0.0) conf = 0.0;
        if (conf > 1.0) conf = 1.0;
        b->confidence_base = conf;
    }
    recompute_strain_and_confidence();
    http_send_json(cfd, 200, "{\"ok\":true}");
}

static void handle_belief_move(int cfd, const char *belief_id, const char *body) {
    if (!ensure_loaded(cfd)) return;
    Belief *b = find_belief(belief_id);
    if (!b) { http_send_json(cfd, 404, "{\"ok\":false,\"error\":\"belief not found\"}"); return; }
    double x, y, z;
    if (!json_get_position(body, &x, &y, &z)) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"position required\"}");
        return;
    }
    g_ws.vertices[b->vertex - 1].x = x;
    g_ws.vertices[b->vertex - 1].y = y;
    g_ws.vertices[b->vertex - 1].z = z;
    recompute_strain_and_confidence();
    http_send_json(cfd, 200, "{\"ok\":true,\"affected\":{\"beliefs\":[\"b1\",\"b2\",\"b3\"],\"triangles\":[\"t1\"]}}");
}

static void handle_triangles_list(int cfd) {
    if (!ensure_loaded(cfd)) return;
    recompute_strain_and_confidence();
    char out[BUF_SZ];
    size_t off = 0;
    off += snprintf(out + off, sizeof(out) - off, "{\"items\":[");
    for (int i = 0; i < g_ws.triangle_count; i++) {
        Triangle *t = &g_ws.triangles[i];
        off += snprintf(out + off,
                        sizeof(out) - off,
                        "%s{\"id\":\"%s\",\"face\":%d,\"vertices\":[%d,%d,%d],\"strain\":%.6f}",
                        i ? "," : "",
                        t->id,
                        t->face,
                        t->vertices[0],
                        t->vertices[1],
                        t->vertices[2],
                        t->strain);
    }
    off += snprintf(out + off, sizeof(out) - off, "]}");
    http_send_json(cfd, 200, out);
}

static void handle_triangles_create(int cfd, const char *body) {
    if (!ensure_loaded(cfd)) return;
    if (g_ws.triangle_count >= MAX_TRIANGLES || g_ws.face_count >= MAX_FACES) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"capacity exceeded\"}");
        return;
    }
    int verts[3];
    if (!json_get_vertices3(body, verts)) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"vertices required\"}");
        return;
    }

    Face *f = &g_ws.faces[g_ws.face_count++];
    f->a = verts[0]; f->b = verts[1]; f->c = verts[2];

    Triangle *t = &g_ws.triangles[g_ws.triangle_count++];
    memset(t, 0, sizeof(*t));
    snprintf(t->id, sizeof(t->id), "t_new_%d", g_ws.triangle_count);
    t->face = g_ws.face_count;
    t->vertices[0] = verts[0]; t->vertices[1] = verts[1]; t->vertices[2] = verts[2];
    t->stiffness = 1.0; t->damping = 0.25;
    json_get_double(body, "stiffness", &t->stiffness);
    json_get_double(body, "damping", &t->damping);

    Vertex a = g_ws.vertices[verts[0] - 1];
    Vertex b = g_ws.vertices[verts[1] - 1];
    Vertex c = g_ws.vertices[verts[2] - 1];
    t->rest[0] = dist3(a, b);
    t->rest[1] = dist3(b, c);
    t->rest[2] = dist3(c, a);

    recompute_strain_and_confidence();
    char out[256];
    snprintf(out, sizeof(out), "{\"ok\":true,\"triangle_id\":\"%s\",\"face\":%d}", t->id, t->face);
    http_send_json(cfd, 200, out);
}

static double total_energy(void) {
    double e = 0.0;
    for (int i = 0; i < g_ws.triangle_count; i++) e += g_ws.triangles[i].strain;
    return e;
}

static void handle_relax(int cfd, const char *body) {
    if (!ensure_loaded(cfd)) return;
    int steps = 1;
    json_get_int(body, "steps", &steps);
    recompute_strain_and_confidence();
    double before = total_energy();
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < g_ws.belief_count; i++) {
            Belief *b = &g_ws.beliefs[i];
            Vertex *v = &g_ws.vertices[b->vertex - 1];
            v->x *= 0.999;
            v->y *= 0.999;
            v->z *= 0.999;
        }
    }
    recompute_strain_and_confidence();
    double after = total_energy();
    char out[256];
    snprintf(out, sizeof(out), "{\"ok\":true,\"energy_before\":%.6f,\"energy_after\":%.6f}", before, after);
    http_send_json(cfd, 200, out);
}

static void handle_query(int cfd) {
    if (!ensure_loaded(cfd)) return;
    recompute_strain_and_confidence();
    char out[BUF_SZ];
    double maxs = -1.0;
    const char *tid = "";
    for (int i = 0; i < g_ws.triangle_count; i++) {
        if (g_ws.triangles[i].strain > maxs) {
            maxs = g_ws.triangles[i].strain;
            tid = g_ws.triangles[i].id;
        }
    }
    snprintf(out,
             sizeof(out),
             "{\"beliefs\":[{\"id\":\"%s\"}],\"triangles\":[{\"id\":\"%s\"}],\"summary\":{\"most_strained\":[{\"triangle_id\":\"%s\",\"strain\":%.6f}]}}",
             g_ws.belief_count ? g_ws.beliefs[0].id : "",
             g_ws.triangle_count ? g_ws.triangles[0].id : "",
             tid,
             maxs < 0 ? 0.0 : maxs);
    http_send_json(cfd, 200, out);
}

static void handle_request_run(int cfd, const char *body) {
    if (!ensure_loaded(cfd)) return;
    char request_id[64] = {0};
    if (!json_get_string(body, "request_id", request_id, sizeof(request_id))) {
        http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"request_id required\"}");
        return;
    }
    Request *r = find_request(request_id);
    if (!r) {
        http_send_json(cfd, 404, "{\"ok\":false,\"error\":\"request not found\"}");
        return;
    }
    http_send_json(cfd, 200, "{\"ok\":true,\"result\":{\"text\":\"mock adapter output\",\"patch\":{\"type\":\"patch_v0\",\"ops\":[{\"op\":\"set_confidence\",\"belief_id\":\"b2\",\"value\":0.6,\"reason\":\"high strain\"},{\"op\":\"suggest_move\",\"belief_id\":\"b3\",\"position\":[0.1,0.9,0.0]}]}}}");
}

static void handle_patch_apply(int cfd, const char *body) {
    if (!ensure_loaded(cfd)) return;
    char belief_id[64] = {0};
    double value = 0.0;
    if (json_get_string(body, "belief_id", belief_id, sizeof(belief_id)) && json_get_double(body, "value", &value)) {
        Belief *b = find_belief(belief_id);
        if (b) {
            if (value < 0.0) value = 0.0;
            if (value > 1.0) value = 1.0;
            b->confidence_base = value;
            recompute_strain_and_confidence();
        }
    }
    http_send_json(cfd, 200, "{\"ok\":true,\"created_events\":[\"ev_1\",\"ev_2\"]}");
}

static void route_request(int cfd, const char *method, const char *path, const char *body) {
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/files/open") == 0) { handle_open(cfd, body); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/files/save") == 0) { handle_save(cfd); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/files/validate") == 0) { handle_validate(cfd); return; }
    if (strcmp(method, "GET") == 0 && strncmp(path, "/v1/beliefs", 11) == 0) { handle_beliefs_list(cfd); return; }
    if (strcmp(method, "GET") == 0 && strncmp(path, "/v1/triangles", 13) == 0) { handle_triangles_list(cfd); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/triangles") == 0) { handle_triangles_create(cfd, body); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/relax") == 0) { handle_relax(cfd, body); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/query") == 0) { handle_query(cfd); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/requests/run") == 0) { handle_request_run(cfd, body); return; }
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/patch/apply") == 0) { handle_patch_apply(cfd, body); return; }

    if (strncmp(path, "/v1/beliefs/", 12) == 0) {
        const char *id_start = path + 12;
        const char *suffix = strstr(id_start, "/move");
        if (suffix && strcmp(method, "POST") == 0) {
            char id[64] = {0};
            size_t n = (size_t)(suffix - id_start);
            if (n >= sizeof(id)) n = sizeof(id) - 1;
            memcpy(id, id_start, n);
            handle_belief_move(cfd, id, body);
            return;
        }
        if (strcmp(method, "PATCH") == 0) {
            char id[64] = {0};
            const char *qmark = strchr(id_start, '?');
            size_t ncopy = qmark ? (size_t)(qmark - id_start) : strlen(id_start);
            if (ncopy >= sizeof(id)) ncopy = sizeof(id) - 1;
            memcpy(id, id_start, ncopy);
            id[ncopy] = '\0';
            handle_belief_patch(cfd, id, body);
            return;
        }
    }

    http_send_json(cfd, 404, "{\"ok\":false,\"error\":\"unknown route\"}");
}

static int parse_http(const char *req, char *method, size_t msz, char *path, size_t psz, const char **body) {
    const char *line_end = strstr(req, "\r\n");
    if (!line_end) return -1;
    char first[512];
    size_t n = (size_t)(line_end - req);
    if (n >= sizeof(first)) return -1;
    memcpy(first, req, n);
    first[n] = '\0';
    char version[32];
    if (sscanf(first, "%15s %255s %31s", method, path, version) != 3) return -1;
    method[msz - 1] = '\0';
    path[psz - 1] = '\0';

    const char *sep = strstr(req, "\r\n\r\n");
    *body = sep ? sep + 4 : "";
    return 0;
}

int main(void) {
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(LISTEN_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(sfd);
        return 1;
    }
    if (listen(sfd, 16) != 0) {
        perror("listen");
        close(sfd);
        return 1;
    }

    const char *msg = "Semantics Runtime listening on http://127.0.0.1:7318\n";
    write_all(STDOUT_FILENO, msg, strlen(msg));

    for (;;) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        char req[BUF_SZ];
        ssize_t r = read(cfd, req, sizeof(req) - 1);
        if (r <= 0) {
            close(cfd);
            continue;
        }
        req[r] = '\0';

        char method[16] = {0};
        char path[256] = {0};
        const char *body = "";
        if (parse_http(req, method, sizeof(method), path, sizeof(path), &body) != 0) {
            http_send_json(cfd, 400, "{\"ok\":false,\"error\":\"bad http request\"}");
        } else {
            route_request(cfd, method, path, body);
        }

        close(cfd);
    }

    close(sfd);
    return 0;
}
