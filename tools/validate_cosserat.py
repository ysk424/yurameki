"""Offline validation for the Cosserat rod solver (no Blender required).

Run:  python tools/validate_cosserat.py [cpu|cuda:0]

Layer 1 -- a self-contained numpy reference of the same energy and Gauss-Newton
           math, finite-difference checked (gradients / rhs).
Layer 2 -- the actual Warp kernels from _cosserat.py, checked to reproduce the
           numpy reference trajectory, plus rest-stability and gravity-drape
           behaviour with rod length preserved *without* any FK reconnection.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp  # noqa: E402
import _cosserat as cr  # noqa: E402

np.set_printoptions(suppress=True, precision=6)
E3 = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# numpy quaternion helpers (xyzw), mirroring the device convention
# --------------------------------------------------------------------------- #
def qmul(a, b):
    return cr._quat_mul(a, b)


def qconj(a):
    return cr._quat_conj_np(a)


def qrot(q, v):
    x, y, z, w = q
    r = qmul(qmul(q, np.array([v[0], v[1], v[2], 0.0])), qconj(q))
    return r[:3]


def dir3(q):
    return qrot(q, E3)


def expq(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        q = np.array([0.5 * w[0], 0.5 * w[1], 0.5 * w[2], 1.0])
        return q / np.linalg.norm(q)
    half = 0.5 * theta
    s = np.sin(half) / theta
    return np.array([w[0] * s, w[1] * s, w[2] * s, np.cos(half)])


def rel_darboux(qa, qb):
    return cr._rel_darboux_np(qa, qb)


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


# --------------------------------------------------------------------------- #
# numpy reference energy and local Gauss-Newton (mirror of the kernels)
# --------------------------------------------------------------------------- #
def energy(P, Q, seg_rest, dar_rest, pps, k_ss, k_bt,
           inertial=None, mass=None, inv_dt2=0.0, invm=None):
    P = P.reshape(-1, pps, 3)
    ns = P.shape[0]
    segs = pps - 1
    bends = pps - 2
    Q = Q.reshape(ns, segs, 4)
    dar_rest = dar_rest.reshape(ns, bends, 3)
    seg_rest = seg_rest.reshape(ns, segs)
    E = 0.0
    for s in range(ns):
        for e in range(segs):
            d3 = dir3(Q[s, e])
            cs = (P[s, e + 1] - P[s, e]) - seg_rest[s, e] * d3
            E += 0.5 * k_ss * float(cs @ cs)
        for e in range(bends):
            cb = rel_darboux(Q[s, e], Q[s, e + 1])[:3] - dar_rest[s, e]
            E += 0.5 * k_bt * float(cb @ cb)
    if inertial is not None:
        Pf = P.reshape(-1, 3)
        If = inertial.reshape(-1, 3)
        for k in range(Pf.shape[0]):
            if invm[k] > 0.0:
                m = 1.0 / invm[k]
                d = Pf[k] - If[k]
                E += 0.5 * m * inv_dt2 * float(d @ d)
    return E


def orient_rhs_A(P, Q, dar_rest, seg_rest, s, local_seg, pps, k_ss, k_bt, damping):
    """Analytic (A, rhs) for one segment -- exact mirror of the Warp kernel."""
    segs = pps - 1
    bends = pps - 2
    Ps = P.reshape(-1, pps, 3)[s]
    Qs = Q.reshape(-1, segs, 4)[s]
    dr = dar_rest.reshape(-1, bends, 3)[s]
    sr = seg_rest.reshape(-1, segs)[s]
    q = Qs[local_seg]
    i, j = local_seg, local_seg + 1

    A = np.zeros((3, 3))
    rhs = np.zeros(3)
    I3 = np.eye(3)

    d3 = dir3(q)
    sl = sr[local_seg]
    s_res = (Ps[j] - Ps[i]) - sl * d3
    dd0 = qrot(q, np.cross([1.0, 0, 0], E3))  # (0,-1,0) rotated
    dd1 = qrot(q, np.cross([0, 1.0, 0], E3))  # (1,0,0) rotated
    Js = np.zeros((3, 3))
    Js[:, 0] = -sl * dd0
    Js[:, 1] = -sl * dd1
    A += k_ss * (Js.T @ Js)
    rhs += -k_ss * (Js.T @ s_res)

    if local_seg < segs - 1:
        d = rel_darboux(q, Qs[local_seg + 1])
        dv, dw = d[:3], d[3]
        b_r = dv - dr[local_seg]
        Jr = 0.5 * (-dw * I3 + skew(dv))
        A += k_bt * (Jr.T @ Jr)
        rhs += -k_bt * (Jr.T @ b_r)

    if local_seg > 0:
        g = rel_darboux(Qs[local_seg - 1], q)
        gv, gw = g[:3], g[3]
        b_l = gv - dr[local_seg - 1]
        Jl = 0.5 * (gw * I3 + skew(gv))
        A += k_bt * (Jl.T @ Jl)
        rhs += -k_bt * (Jl.T @ b_l)

    A += damping * I3
    return A, rhs


def numpy_position_solve(predicted, inertial, invm, Q, seg_rest, pps, k_ss, inv_dt2):
    """Exact per-strand position solve (independent dense tridiagonal build)."""
    P = predicted.copy()
    ns = P.shape[0] // pps
    segs = pps - 1
    for s in range(ns):
        base = s * pps
        f0 = 0
        while f0 < pps and invm[base + f0] <= 0.0:
            f0 += 1
        if f0 >= pps:
            continue
        nfree = pps - f0
        D = np.zeros(nfree); A = np.zeros(nfree); C = np.zeros(nfree)
        B = np.zeros((nfree, 3))
        for idx in range(nfree):
            r = f0 + idx
            gi = base + r
            has_left = r > 0; has_right = r < pps - 1
            m = 1.0 / invm[gi]
            diag = m * inv_dt2
            rhs = m * inv_dt2 * inertial[gi].copy()
            if has_right:
                diag += k_ss
                se = s * segs + r
                rhs = rhs - k_ss * seg_rest[se] * dir3(Q[se])
            if has_left:
                diag += k_ss
                se = s * segs + (r - 1)
                rhs = rhs + k_ss * seg_rest[se] * dir3(Q[se])
            D[idx] = diag
            if has_right:
                C[idx] = -k_ss
            if has_left and idx > 0:
                A[idx] = -k_ss
            elif has_left and idx == 0:
                rhs = rhs + k_ss * P[gi - 1]
            B[idx] = rhs
        M = np.diag(D) + np.diag(A[1:], -1) + np.diag(C[:-1], 1)
        X = np.linalg.solve(M, B)
        P[base + f0:base + pps] = X
    return P


def numpy_step(P, Q, seg_rest, dar_rest, pps, k_ss, k_bt,
               invm, target, gravity, dt, damping, iters,
               pos_relax=1.0, omega_relax=1.0, gn_damping=1e-7, max_omega=0.5,
               vel=None):
    """One substep of the numpy reference solver (predict/solve/velocity)."""
    ns = P.shape[0] // pps
    segs = pps - 1
    inv_dt2 = 1.0 / (dt * dt)
    if vel is None:
        vel = np.zeros_like(P)

    # predict
    vel = vel.copy()
    free = invm > 0.0
    vel[free] += gravity * dt
    inertial = P + vel * dt
    predicted = inertial.copy()
    predicted[~free] = target[~free]
    inertial[~free] = target[~free]

    Q = Q.copy()
    for _ in range(iters):
        # position: exact per-strand tridiagonal solve (dense reference)
        predicted = numpy_position_solve(predicted, inertial, invm, Q, seg_rest,
                                         pps, k_ss, inv_dt2)
        # orientation: red/black over local segment index
        for parity in (0, 1):
            newQ = Q.copy()
            for e in range(ns * segs):
                local_seg = e % segs
                if local_seg % 2 != parity:
                    continue
                strand = e // segs
                A, rhs = orient_rhs_A(predicted, Q, dar_rest, seg_rest, strand,
                                      local_seg, pps, k_ss, k_bt, gn_damping)
                if abs(np.linalg.det(A)) < 1e-20:
                    continue
                omega = omega_relax * np.linalg.solve(A, rhs)
                mag = np.linalg.norm(omega)
                if mag > max_omega and mag > 1e-9:
                    omega *= max_omega / mag
                qn = qmul(Q[e], expq(omega))
                newQ[e] = qn / np.linalg.norm(qn)
            Q = newQ

    new_vel = (predicted - P) / dt * (1.0 - damping)
    new_vel[~free] = 0.0
    return predicted, Q, new_vel


# --------------------------------------------------------------------------- #
# Warp driver mirroring numpy_step, using the real kernels
# --------------------------------------------------------------------------- #
@wp.kernel
def _predict(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
             predicted: wp.array(dtype=wp.vec3), inertial: wp.array(dtype=wp.vec3),
             target: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
             g: wp.vec3, dt: float):
    i = wp.tid()
    if invm[i] <= 0.0:
        pos[i] = target[i]
        predicted[i] = target[i]
        inertial[i] = target[i]
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v = vel[i] + g * dt
        vel[i] = v
        inertial[i] = pos[i] + v * dt
        predicted[i] = inertial[i]


@wp.kernel
def _velupd(pos: wp.array(dtype=wp.vec3), predicted: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
            dt: float, damping: float):
    i = wp.tid()
    if invm[i] <= 0.0:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        vel[i] = (predicted[i] - pos[i]) / dt * (1.0 - damping)
    pos[i] = predicted[i]


def warp_run(P0, Q0, seg_rest, dar_rest, pps, k_ss, k_bt, invm, target,
             gravity, dt, damping, iters, nsteps, device,
             pos_relax=1.0, omega_relax=1.0, gn_damping=1e-7, max_omega=0.5):
    n = P0.shape[0]
    segs = pps - 1
    pos = wp.array(P0.astype(np.float32), dtype=wp.vec3, device=device)
    vel = wp.zeros(n, dtype=wp.vec3, device=device)
    predicted = wp.array(P0.astype(np.float32), dtype=wp.vec3, device=device)
    inertial = wp.array(P0.astype(np.float32), dtype=wp.vec3, device=device)
    tgt = wp.array(target.astype(np.float32), dtype=wp.vec3, device=device)
    im = wp.array(invm.astype(np.float32), dtype=float, device=device)
    orient = wp.array(Q0.astype(np.float32), dtype=wp.quat, device=device)
    sr = wp.array(seg_rest.astype(np.float32), dtype=float, device=device)
    dr = wp.array(dar_rest.astype(np.float32), dtype=wp.vec3, device=device)
    sc_c = wp.zeros(n, dtype=float, device=device)
    sc_d = wp.zeros(n, dtype=wp.vec3, device=device)
    g = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))
    inv_dt2 = 1.0 / (dt * dt)
    nstr = n // pps

    for _ in range(nsteps):
        wp.launch(_predict, dim=n, inputs=[pos, vel, predicted, inertial, tgt, im, g, dt], device=device)
        for _ in range(iters):
            wp.launch(cr.cosserat_position_tridiagonal_kernel, dim=nstr,
                      inputs=[predicted, inertial, im, orient, sr, sc_c, sc_d, pps,
                              float(k_ss), float(inv_dt2)], device=device)
            for parity in (0, 1):
                wp.launch(cr.cosserat_orientation_kernel, dim=n_segments(n, pps),
                          inputs=[predicted, orient, dr, sr, pps, parity,
                                  float(k_ss), float(k_bt), float(gn_damping),
                                  float(omega_relax), float(max_omega)], device=device)
        wp.launch(_velupd, dim=n, inputs=[pos, predicted, vel, im, dt, float(damping)], device=device)
    wp.synchronize()
    return pos.numpy().copy(), orient.numpy().copy()


def n_segments(n, pps):
    return (n // pps) * (pps - 1)


# --------------------------------------------------------------------------- #
# Test scenes
# --------------------------------------------------------------------------- #
def make_strand(pps, curve=0.0, seg=0.02, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((pps, 3))
    ang = 0.0
    for i in range(1, pps):
        ang += curve
        d = np.array([np.sin(ang), 0.0, -np.cos(ang)]) * seg
        d += (rng.standard_normal(3) * jitter if jitter else 0.0)
        P[i] = P[i - 1] + d
    return P


def seg_lengths(P, pps):
    S = P.reshape(-1, pps, 3)
    return np.linalg.norm(S[:, 1:] - S[:, :-1], axis=2)


# --------------------------------------------------------------------------- #
def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    wp.init()
    print(f"=== Cosserat solver validation on device={device} ===")
    fails = 0

    # ---- Test 1: finite-difference of orientation rhs (= -dE/domega) ------- #
    print("\n[1] FD check: orientation rhs == -grad(E) wrt omega")
    pps = 6
    P = make_strand(pps, curve=0.35, jitter=0.002, seed=1).reshape(-1, 3)
    Q, dar, sr = cr.init_cosserat(P, pps)
    Q = Q.astype(np.float64); dar = dar.astype(np.float64); sr = sr.astype(np.float64)
    # perturb Q away from rest so bend residual is non-zero
    rng = np.random.default_rng(2)
    for e in range(len(Q)):
        Q[e] = qmul(Q[e], expq(rng.standard_normal(3) * 0.05))
        Q[e] /= np.linalg.norm(Q[e])
    P += rng.standard_normal(P.shape) * 0.001
    k_ss, k_bt = 120.0, 5.0
    h = 1e-6
    max_rel = 0.0
    for local_seg in range(pps - 1):
        _, rhs = orient_rhs_A(P, Q, dar, sr, 0, local_seg, pps, k_ss, k_bt, 0.0)
        fd = np.zeros(3)
        for k in range(3):
            for sign in (+1, -1):
                Qp = Q.copy()
                dw = np.zeros(3); dw[k] = sign * h
                qn = qmul(Q[0 * (pps - 1) + local_seg], expq(dw))
                Qp[local_seg] = qn / np.linalg.norm(qn)
                Ep = energy(P, Qp, sr, dar, pps, k_ss, k_bt)
                fd[k] += sign * Ep
            fd[k] /= (2 * h)
        grad = -rhs  # rhs = -grad
        denom = max(np.linalg.norm(grad), 1e-6)
        rel = np.linalg.norm(grad - fd) / denom
        max_rel = max(max_rel, rel)
    ok = max_rel < 2e-4
    print(f"    max relative error = {max_rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- Test 2: FD of position gradient ----------------------------------- #
    print("\n[2] FD check: position gradient (inertia + stretch/shear)")
    invm = np.ones(pps); invm[0] = 0.0
    mass = 1.0
    dt = 0.01
    inv_dt2 = 1.0 / dt**2
    inertial = P + rng.standard_normal(P.shape) * 0.01
    inertial[0] = P[0]

    def Efull(Pv):
        return energy(Pv, Q, sr, dar, pps, k_ss, k_bt, inertial=inertial,
                      mass=mass, inv_dt2=inv_dt2, invm=invm)

    max_rel = 0.0
    for k in range(1, pps):
        e = k  # right seg index within strand
        grad = (1.0 / invm[k]) * inv_dt2 * (P[k] - inertial[k])
        if k < pps - 1:
            cs = (P[k + 1] - P[k]) - sr[k] * dir3(Q[k])
            grad = grad - k_ss * cs
        if k > 0:
            cs = (P[k] - P[k - 1]) - sr[k - 1] * dir3(Q[k - 1])
            grad = grad + k_ss * cs
        fd = np.zeros(3)
        for c in range(3):
            for sign in (+1, -1):
                Pp = P.copy(); Pp[k, c] += sign * h
                fd[c] += sign * Efull(Pp)
            fd[c] /= (2 * h)
        rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(grad), 1e-6)
        max_rel = max(max_rel, rel)
    ok = max_rel < 2e-4
    print(f"    max relative error = {max_rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- Test 3: Warp kernels reproduce numpy reference -------------------- #
    print("\n[3] Warp kernels vs numpy reference (20 substeps, gravity drape)")
    pps = 8
    P0 = make_strand(pps, curve=0.15, seg=0.03, seed=5).reshape(-1, 3)
    Q0, dar0, sr0 = cr.init_cosserat(P0, pps)
    invm = np.ones(pps); invm[0] = invm[1] = 0.0
    invm = invm / 1.0
    target = P0.copy()
    grav = np.array([0.0, 0.0, -9.81])
    dt = 0.01
    k_ss, k_bt = 200.0, 2.0
    iters = 15
    # numpy
    Pn = P0.copy(); Qn = Q0.astype(np.float64).copy(); veln = np.zeros_like(P0)
    for _ in range(20):
        Pn, Qn, veln = numpy_step(Pn, Qn, sr0.astype(np.float64), dar0.astype(np.float64),
                                  pps, k_ss, k_bt, invm, target, grav, dt, 0.02, iters, vel=veln)
    # warp
    Pw, Qw = warp_run(P0, Q0, sr0, dar0, pps, k_ss, k_bt, invm, target, grav, dt,
                      0.02, iters, 20, device)
    dmax = np.max(np.linalg.norm(Pn - Pw, axis=1)) * 1000.0  # mm
    ok = dmax < 0.05
    print(f"    max |numpy - warp| position diff = {dmax:.4f} mm  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- Test 4: rest stability (no gravity, locked root) ------------------ #
    print("\n[4] Rest stability: no drift over 100 substeps, gravity off")
    pps = 12
    P0 = make_strand(pps, curve=0.2, seg=0.025, seed=7).reshape(-1, 3)
    Q0, dar0, sr0 = cr.init_cosserat(P0, pps)
    invm = np.ones(pps); invm[:3] = 0.0
    Pw, _ = warp_run(P0, Q0, sr0, dar0, pps, 300.0, 5.0, invm, P0.copy(),
                     np.zeros(3), 0.01, 0.0, 12, 100, device)
    drift = np.max(np.linalg.norm(Pw - P0, axis=1)) * 1000.0
    ok = drift < 0.05
    print(f"    max drift from rest = {drift:.5f} mm  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- Test 5: length preservation under gravity WITHOUT FK -------------- #
    # Horizontal cantilever clamped at the root: gravity should droop it into a
    # catenary-like curve (clear tip drop) while the stiff stretch keeps every
    # segment at rest length -- the property that removes the FK reconnection.
    print("\n[5] Horizontal cantilever drape: length preserved without FK")
    pps = 12
    seg = 0.03
    P0 = np.zeros((pps, 3))
    P0[:, 0] = np.arange(pps) * seg  # horizontal along +x
    P0 = P0.reshape(-1, 3)
    Q0, dar0, sr0 = cr.init_cosserat(P0, pps)
    mass = 5.0e-4
    invm = np.ones(pps) / mass; invm[:2] = 0.0
    k_ss, k_bt = 1.0e4, 2.0e-3
    dt = 1.0 / 120.0
    rest_len = seg_lengths(P0, pps)
    total_len = float(rest_len.sum())
    Pw, Qw = warp_run(P0, Q0, sr0, dar0, pps, k_ss, k_bt, invm, P0.copy(),
                      np.array([0.0, 0.0, -9.81]), dt, 0.02, 15, 600, device)
    final_len = seg_lengths(Pw, pps)
    rel_err = np.max(np.abs(final_len - rest_len) / rest_len) * 100.0
    tip_drop = (P0.reshape(-1, pps, 3)[0, -1, 2] - Pw.reshape(-1, pps, 3)[0, -1, 2])
    nan = not np.all(np.isfinite(Pw))
    ok = (rel_err < 0.2) and (tip_drop > 0.03) and (not nan)
    print(f"    m/h^2 = {mass/dt**2:.2f}   k_ss = {k_ss:.0f}   (ratio {k_ss/(mass/dt**2):.0f}:1)"
          f"   total rod length = {total_len*100:.1f} cm")
    print(f"    max segment length error = {rel_err:.4f} %   tip drop = {tip_drop*100:.2f} cm"
          f"   nan={nan}  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- Test 6: internal (strain-rate) velocity damping ------------------- #
    print("\n[6] Internal damping: preserves bulk velocity, damps jitter")
    pps = 12
    invm6 = np.ones(pps, np.float32); invm6[:3] = 0.0
    im6 = wp.array(invm6, dtype=float, device=device)
    tmp6 = wp.zeros(pps, dtype=wp.vec3, device=device)

    def apply_damp(vel_np, mu, sweeps):
        v = wp.array(vel_np.astype(np.float32), dtype=wp.vec3, device=device)
        for _ in range(sweeps):
            wp.copy(tmp6, v)
            wp.launch(cr._internal_damp_kernel, dim=pps, inputs=[tmp6, v, im6, pps, float(mu)], device=device)
        wp.synchronize()
        return v.numpy()

    # (a) a spatially uniform velocity is a fixed point -> bulk motion preserved
    vel_u = np.tile([1.0, 0.0, 0.0], (pps, 1))
    out_u = apply_damp(vel_u, 0.4, 30)
    bulk_change = float(np.max(np.abs(out_u[3:] - vel_u[3:])))
    # (b) mean + alternating jitter -> jitter decays, per-strand mean is exact
    vel_j = np.array([[0.5, 0.0, 0.0]] * pps, dtype=float)
    for i in range(pps):
        vel_j[i, 2] = (-1.0) ** i
    jit0 = float(np.std(vel_j[3:], axis=0).sum())
    mean0 = vel_j[3:].mean(axis=0)
    out_j = apply_damp(vel_j, 0.4, 30)
    jit1 = float(np.std(out_j[3:], axis=0).sum())
    mean_drift = float(np.linalg.norm(out_j[3:].mean(axis=0) - mean0))
    ok = (bulk_change < 1e-5) and (jit1 < 0.1 * jit0) and (mean_drift < 1e-5)
    print(f"    uniform-vel change = {bulk_change:.2e} (bulk preserved)   "
          f"jitter {jit0:.3f} -> {jit1:.3f}   mean drift = {mean_drift:.2e}  -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    print(f"\n=== {'ALL TESTS PASSED' if fails == 0 else str(fails) + ' TEST(S) FAILED'} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
