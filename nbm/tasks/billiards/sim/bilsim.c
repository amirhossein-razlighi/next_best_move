/* Deterministic 2D pool simulator of the billiards task (Python wrapper: nbm/tasks/billiards/physics.py).

   Model (no spin): equal-mass balls of radius R on a rectangular table [0, L] x [0, W] (cushion noses; ball centres stay in
   [R, L-R] x [R, W-R]); every moving ball decelerates at the constant rate A along its velocity (rolling friction) and stops;
   ball-ball collisions are instantaneous, frictionless, equal mass with normal restitution EB (momentum conserved exactly);
   cushions reflect the normal velocity component with restitution EC (tangential component unchanged); six pockets = points
   (0,0) (L/2,0) (L,0) (0,W) (L/2,W) (L,W): a ball is potted (removed) the moment its centre comes within RC (corner) / RS (side)
   of a pocket point.

   Integration: event driven inside fixed steps of at most H seconds. Over a step each moving ball is moved along its exact
   decelerating straight-line displacement; events (ball-ball contact, cushion, pocket entry) are located on the chord of that
   step (exact contact geometry; the chord deviates from the true trajectory by <= A H^2 / 8 = 1e-5 table widths at the default
   H = 0.01 s), the earliest event is resolved and a new step starts at the event time. Recording frames (rec_t) reads the exact
   trajectories and never changes the step sequence, so outcomes do not depend on what is recorded. Pure IEEE double arithmetic
   (built with -O2 -ffp-contract=off -fno-fast-math, no FMA): identical results for identical inputs on every machine.

   Parameters P[]: L W R A EB EC RC RS H TMAX MAXEV. Ball 0 = cue ball, ball 1 = target. Angles in degrees, counter-clockwise from +x
   with y up. Speed v0 in table widths per second. */
#include <math.h>
#include <string.h>

#define MAXB 16
#define INFO_N 12
#define BIL_VERSION 3

int bil_version(void) { return BIL_VERSION; }
int bil_info_n(void) { return INFO_N; }

static double disp(double v, double a, double t) {
    double ts = v / a;
    if (t >= ts) return v * v / (2.0 * a);
    return v * t - 0.5 * a * t * t;
}

/* earliest root tau in [0, hmax] of |q + w tau|^2 = r^2 for an approaching / entering configuration, or -1 */
static double first_contact(double qx, double qy, double wx, double wy, double r, double hmax) {
    double a2 = wx * wx + wy * wy;
    double b = 2.0 * (qx * wx + qy * wy);
    double c = qx * qx + qy * qy - r * r;
    if (b >= 0.0) return -1.0;          /* separating / tangential */
    if (c <= 0.0) return 0.0;           /* already touching or overlapping while approaching */
    if (a2 <= 0.0) return -1.0;
    double disc = b * b - 4.0 * a2 * c;
    if (disc < 0.0) return -1.0;
    double tau = (-b - sqrt(disc)) / (2.0 * a2);
    if (tau < 0.0) tau = 0.0;
    return tau <= hmax ? tau : -1.0;
}

/* info: [0] first ball the cue touches (-1 none) [1] cue cushions before that contact (-1 if none) [2] target cushions before it
   was potted (-1 not potted) [3] target ball contacts before potted [4] first ball that touched the target (-1) [5] events
   [6] overflow (event cap or TMAX reached) [7] cue cushions total [8] ball-ball collisions total [9] target pocket (-1)
   [10] cue pocket (-1) [11] number of object balls potted */
int bil_sim(const double *P, int n, const double *xy0, const int *alive0, double ang_deg, double v0, int nrec, const double *rec_t,
            double *rec_xy, int *rec_alive, double *out_xy, int *out_pocket, double *out_ptime, int *info, double *out_trest) {
    const double L = P[0], W = P[1], R = P[2], A = P[3], EB = P[4], EC = P[5], RC = P[6], RS = P[7], H = P[8], TMAX = P[9];
    const int MAXEV = (int)P[10];
    double px[MAXB], py[MAXB], dx[MAXB], dy[MAXB], sp[MAXB], wx[MAXB], wy[MAXB];
    int al[MAXB], ncush[MAXB], nhit[MAXB], fhit[MAXB];
    const double pkx[6] = {0.0, 0.5 * L, L, 0.0, 0.5 * L, L}, pky[6] = {0.0, 0.0, 0.0, W, W, W};
    const double prc[6] = {RC, RS, RC, RC, RS, RC};
    int i, j, k, ri = 0, nev = 0, overflow = 0, ncol = 0;
    double t = 0.0;
    if (n > MAXB) return -2;
    for (i = 0; i < n; i++) {
        px[i] = xy0[2 * i]; py[i] = xy0[2 * i + 1]; al[i] = alive0[i];
        dx[i] = 1.0; dy[i] = 0.0; sp[i] = 0.0; ncush[i] = 0; nhit[i] = 0; fhit[i] = -1;
        out_pocket[i] = -1; out_ptime[i] = -1.0;
    }
    for (k = 0; k < INFO_N; k++) info[k] = -1;
    info[3] = 0; info[5] = 0; info[6] = 0; info[8] = 0; info[11] = 0;
    {
        double th = ang_deg * (M_PI / 180.0);
        dx[0] = cos(th); dy[0] = sin(th); sp[0] = v0 > 0.0 ? v0 : 0.0;
    }
    for (;;) {
        int moving = 0;
        for (i = 0; i < n; i++) if (al[i] && sp[i] > 0.0) moving = 1;
        if (!moving) break;
        if (t >= TMAX || nev >= MAXEV) { overflow = 1; break; }
        for (i = 0; i < n; i++) {
            if (al[i] && sp[i] > 0.0) {
                double s = disp(sp[i], A, H);
                wx[i] = dx[i] * s / H; wy[i] = dy[i] * s / H;
            } else { wx[i] = 0.0; wy[i] = 0.0; }
        }
        double best = H + 1.0;
        int et = -1, ea = -1, eb = -1;  /* type 0 cushion (ea ball, eb axis 0..3), 1 pocket (ea ball, eb pocket), 2 pair (ea, eb) */
        for (i = 0; i < n; i++) {
            if (!al[i] || sp[i] <= 0.0) continue;
            double tau;
            if (wx[i] < 0.0) { tau = (px[i] - R) / (-wx[i]); if (tau < 0.0) tau = 0.0; if (tau < best) { best = tau; et = 0; ea = i; eb = 0; } }
            if (wx[i] > 0.0) { tau = (L - R - px[i]) / wx[i]; if (tau < 0.0) tau = 0.0; if (tau < best) { best = tau; et = 0; ea = i; eb = 1; } }
            if (wy[i] < 0.0) { tau = (py[i] - R) / (-wy[i]); if (tau < 0.0) tau = 0.0; if (tau < best) { best = tau; et = 0; ea = i; eb = 2; } }
            if (wy[i] > 0.0) { tau = (W - R - py[i]) / wy[i]; if (tau < 0.0) tau = 0.0; if (tau < best) { best = tau; et = 0; ea = i; eb = 3; } }
            for (k = 0; k < 6; k++) {
                tau = first_contact(px[i] - pkx[k], py[i] - pky[k], wx[i], wy[i], prc[k], H);
                if (tau >= 0.0 && tau < best) { best = tau; et = 1; ea = i; eb = k; }
            }
        }
        for (i = 0; i < n; i++) {
            if (!al[i]) continue;
            for (j = i + 1; j < n; j++) {
                if (!al[j]) continue;
                if (sp[i] <= 0.0 && sp[j] <= 0.0) continue;
                /* real relative velocity must approach along the centre line (the chord of a ball that stops inside the step
                   is not proportional to its velocity) */
                double rel = (sp[i] * dx[i] - sp[j] * dx[j]) * (px[j] - px[i]) + (sp[i] * dy[i] - sp[j] * dy[j]) * (py[j] - py[i]);
                if (rel <= 0.0) continue;
                double tau = first_contact(px[i] - px[j], py[i] - py[j], wx[i] - wx[j], wy[i] - wy[j], 2.0 * R, H);
                if (tau >= 0.0 && tau < best) { best = tau; et = 2; ea = i; eb = j; }
            }
        }
        double dt = (et >= 0 && best <= H) ? best : H;
        if (dt > H) dt = H;
        if (et >= 0 && best > H) et = -1;
        /* frames inside (t, t + dt]: exact trajectories from the step start */
        while (ri < nrec && rec_t[ri] <= t + dt) {
            double tr = rec_t[ri] - t;
            if (tr < 0.0) tr = 0.0;
            for (i = 0; i < n; i++) {
                double s = (al[i] && sp[i] > 0.0) ? disp(sp[i], A, tr) : 0.0;
                rec_xy[(ri * n + i) * 2] = px[i] + dx[i] * s;
                rec_xy[(ri * n + i) * 2 + 1] = py[i] + dy[i] * s;
                rec_alive[ri * n + i] = al[i];
            }
            ri++;
        }
        for (i = 0; i < n; i++) {
            if (!al[i] || sp[i] <= 0.0) continue;
            if (dt == H) {  /* full step: exact displacement */
                double s = disp(sp[i], A, H);
                px[i] += dx[i] * s; py[i] += dy[i] * s;
            } else {
                px[i] += wx[i] * dt; py[i] += wy[i] * dt;
            }
            sp[i] -= A * dt;
            if (sp[i] <= 1e-12) sp[i] = 0.0;
        }
        t += dt;
        if (et < 0) continue;
        nev++;
        if (et == 0) {
            i = ea;
            double ux = sp[i] * dx[i], uy = sp[i] * dy[i];
            if (eb == 0) { px[i] = R; if (ux < 0.0) ux = -EC * ux; }
            else if (eb == 1) { px[i] = L - R; if (ux > 0.0) ux = -EC * ux; }
            else if (eb == 2) { py[i] = R; if (uy < 0.0) uy = -EC * uy; }
            else { py[i] = W - R; if (uy > 0.0) uy = -EC * uy; }
            double s = sqrt(ux * ux + uy * uy);
            sp[i] = s;
            if (s > 0.0) { dx[i] = ux / s; dy[i] = uy / s; }
            ncush[i]++;
        } else if (et == 1) {
            i = ea;
            al[i] = 0; sp[i] = 0.0; out_pocket[i] = eb; out_ptime[i] = t;
            if (i == 1) { info[2] = ncush[1]; info[3] = nhit[1]; info[9] = eb; }
            if (i == 0) info[10] = eb; else info[11]++;
        } else {
            i = ea; j = eb;
            double qx = px[j] - px[i], qy = py[j] - py[i];
            double d = sqrt(qx * qx + qy * qy);
            if (d <= 0.0) continue;
            double nx = qx / d, ny = qy / d;
            double uix = sp[i] * dx[i], uiy = sp[i] * dy[i], ujx = sp[j] * dx[j], ujy = sp[j] * dy[j];
            double uin = uix * nx + uiy * ny, ujn = ujx * nx + ujy * ny;
            if (uin - ujn > 0.0) {
                double del = 0.5 * (1.0 + EB) * (uin - ujn);
                uix -= del * nx; uiy -= del * ny; ujx += del * nx; ujy += del * ny;
                double si = sqrt(uix * uix + uiy * uiy), sj = sqrt(ujx * ujx + ujy * ujy);
                sp[i] = si; sp[j] = sj;
                if (si > 0.0) { dx[i] = uix / si; dy[i] = uiy / si; }
                if (sj > 0.0) { dx[j] = ujx / sj; dy[j] = ujy / sj; }
                ncol++;
                if ((i == 0 || j == 0) && info[0] < 0) { info[0] = (i == 0) ? j : i; info[1] = ncush[0]; }
                nhit[i]++; nhit[j]++;
                if (fhit[i] < 0) fhit[i] = j;
                if (fhit[j] < 0) fhit[j] = i;
            }
        }
    }
    for (; ri < nrec; ri++) {
        for (i = 0; i < n; i++) {
            rec_xy[(ri * n + i) * 2] = px[i]; rec_xy[(ri * n + i) * 2 + 1] = py[i]; rec_alive[ri * n + i] = al[i];
        }
    }
    for (i = 0; i < n; i++) { out_xy[2 * i] = px[i]; out_xy[2 * i + 1] = py[i]; }
    info[4] = fhit[1]; info[5] = nev; info[6] = overflow; info[7] = ncush[0]; info[8] = ncol;
    *out_trest = t;
    return overflow ? -1 : nev;
}

/* many shots from one layout: out_pmask[s] = bit i set iff ball i potted; out_info[s * INFO_N ..] as bil_sim */
void bil_batch(const double *P, int n, const double *xy0, const int *alive0, int ns, const double *ang, const double *v, int *out_pmask,
               int *out_info) {
    double oxy[2 * MAXB], pt[MAXB], tr;
    int pk[MAXB];
    for (int s = 0; s < ns; s++) {
        bil_sim(P, n, xy0, alive0, ang[s], v[s], 0, 0, 0, 0, oxy, pk, pt, out_info + s * INFO_N, &tr);
        int m = 0;
        for (int i = 0; i < n; i++) if (pk[i] >= 0) m |= 1 << i;
        out_pmask[s] = m;
    }
}
