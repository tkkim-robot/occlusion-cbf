from base_control.dynamic_env.robot import BaseRobotDyn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Polygon, Point, LineString
from shapely import is_valid_reason
from base_control.utils.geometry import custom_merge

class BaseRobotDyn_OCC(BaseRobotDyn):

    def __init__(self, X0, robot_spec, dt, ax):
        super().__init__(X0, robot_spec, dt, ax)

        # Occlusion visualization handles
        self.occlusion_patches = []
        self.occlusion_smax_contours = []
        self.occlusion_future_contours = []
        self._occ_arc_lines = []
        self._occ_barrier_cb = None

        # Force OCC model when requested to avoid import path collisions
        self._ensure_occ_robot()

    def _ensure_occ_robot(self):
        if not (self.robot_spec.get("use_occ", False) and self.robot_spec.get("model") == "DoubleIntegrator2D"):
            return False
        if hasattr(self.robot, "set_occ_barrier_fn") and hasattr(self.robot, "simulate_backup_trajectory"):
            return True
        # Do not force-load a local OCC robot implementation.
        # The project uses the in-tree base_control DoubleIntegrator2D as the
        # only DoubleIntegrator2D model, and occlusion backup logic is handled
        # in the controller layer.
        return False

    def simulate_backup_trajectory(self, *args, **kwargs):
        """Forward backup rollout to the underlying robot."""
        if hasattr(self.robot, "simulate_backup_trajectory"):
            return self.robot.simulate_backup_trajectory(*args, **kwargs)
        self._ensure_occ_robot()
        if hasattr(self.robot, "simulate_backup_trajectory"):
            return self.robot.simulate_backup_trajectory(*args, **kwargs)
        raise AttributeError("Underlying robot does not support simulate_backup_trajectory")

    def _circle_tangents_local(self, p, c, R):
        p = np.asarray(p, float).reshape(2,)
        c = np.asarray(c, float).reshape(2,)
        v = p - c
        d2 = float(v @ v)
        R2 = R * R
        if d2 <= R2:   # No tangents if the robot is inside/on the expanded circle.
            return None, None
        x1, y1 = v
        x0 = R2 * x1 / d2
        y0 = R2 * y1 / d2
        k = R * np.sqrt(d2 - R2) / d2
        t1 = np.array([x0 - y1 * k, y0 + x1 * k]) + c
        t2 = np.array([x0 + y1 * k, y0 - x1 * k]) + c
        return t1, t2

    def _build_occlusion_U0(self, p, c, R_o, R_s, t1, t2, n_arc=40):
        """
        Construct the exact initial occlusion set U0 as:
          - outer arc : circle centered at p, radius R_s
          - inner arc : circle centered at c, radius R_o
          - 2 tangential lines connecting arcs via t1, t2

        Returns:
            poly : (N,2) polygon vertices in CCW order, or None
        """
        import numpy as np
        import math

        p = np.asarray(p, float)
        c = np.asarray(c, float)
        t1 = np.asarray(t1, float)
        t2 = np.asarray(t2, float)

        def norm(v):
            return float(np.linalg.norm(v))

        def normalize_angle(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def arc_with_mid(a1, a2, amid, n):
            # choose orientation so that 'amid' is on the arc
            a1 = normalize_angle(a1)
            a2 = normalize_angle(a2)
            amid = normalize_angle(amid)

            # ccw span
            ccw = normalize_angle(a2 - a1)
            if ccw <= 0:
                ccw += 2.0 * math.pi

            d_mid = normalize_angle(amid - a1)
            if 0.0 <= d_mid <= ccw:
                # use ccw
                return np.linspace(a1, a1 + ccw, n)
            else:
                # use cw (complement)
                cw = 2.0 * math.pi - ccw
                return np.linspace(a1, a1 - cw, n)

        # directions from p through tangent points -> outer intersection
        direction_a = t1 - p
        direction_b = t2 - p
        if norm(direction_a) < 1e-9 or norm(direction_b) < 1e-9:
            return None
        d1 = direction_a / norm(direction_a)
        d2 = direction_b / norm(direction_b)

        f1 = p + R_s * d1
        f2 = p + R_s * d2

        # outer arc angles (center p)
        phi1 = math.atan2(f1[1] - p[1], f1[0] - p[0])
        phi2 = math.atan2(f2[1] - p[1], f2[0] - p[0])
        phi_c = math.atan2(c[1] - p[1], c[0] - p[0])  # direction to obstacle

        outer_phis = arc_with_mid(phi1, phi2, phi_c, n_arc)
        outer_arc = np.vstack([
            p[0] + R_s * np.cos(outer_phis),
            p[1] + R_s * np.sin(outer_phis)
        ]).T

        # inner arc on obstacle circle: choose orientation that does NOT include robot
        theta1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
        theta2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
        theta_p = math.atan2(p[1] - c[1], p[0] - c[0])

        arc_inc = arc_with_mid(theta1, theta2, theta_p, n_arc)
        # complementary arc excludes robot direction
        arc_exc = arc_inc[::-1]

        inner_arc = np.vstack([
            c[0] + R_o * np.cos(arc_exc),
            c[1] + R_o * np.sin(arc_exc)
        ]).T

        # Orient inner_arc so that it starts near t2 (for a nice closed loop)
        if np.linalg.norm(inner_arc[0] - t2) > np.linalg.norm(inner_arc[-1] - t2):
            inner_arc = inner_arc[::-1]

        # Build polygon: outer arc(f1→f2) -> line to t2 -> inner arc(t2→t1) -> line to f1
        poly = []
        poly.extend(outer_arc.tolist())
        poly.append(t2.tolist())
        poly.extend(inner_arc.tolist())
        poly.append(t1.tolist())

        return np.asarray(poly)

    def _line_circle_intersections(self, n, beta, p, R):
        """
        n: (2,) unit normal vector
        line: n·x = beta
        circle: center p, radius R
        returns: two intersections (x_plus, x_minus) or None
        """
        import numpy as np
        n = np.asarray(n, float).reshape(2,)
        p = np.asarray(p, float).reshape(2,)
        d = float(beta - n @ p)               # Signed distance from circle center to the line.
        if abs(d) > R + 1e-9:
            return None
        # Closest point on the line to the circle center
        x0 = p + d * n
        # Unit tangent direction of the line
        t = np.array([-n[1], n[0]], dtype=float)
        l = float(np.sqrt(max(R*R - d*d, 0.0)))
        return x0 + l * t, x0 - l * t

    def _build_occlusion_UT_offset(self, p, c, R_o, R_s, t1, t2, d, n_arc=60):
        """
        Offset the two tangents of U0 by distance d toward the obstacle to build U_T
        The inner arc uses radius R_o + d, the outer arc stays at R_s.
        """
        import numpy as np, math

        p = np.asarray(p, float); c = np.asarray(c, float)
        t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)

        def arc_with_mid(a1, a2, amid, n):
            def normang(a): return (a + math.pi) % (2*math.pi) - math.pi
            a1 = normang(a1); a2 = normang(a2); amid = normang(amid)
            ccw = normang(a2 - a1);  ccw = ccw + 2*math.pi if ccw <= 0 else ccw
            dmid = normang(amid - a1)
            # Choose the arc that includes the obstacle direction
            if 0.0 <= dmid <= ccw:
                return np.linspace(a1, a1 + ccw, n)
            else:
                cw = 2*math.pi - ccw
                return np.linspace(a1, a1 - cw, n)

        # (1) Tangent normals at contact points
        n1 = t1 - c; n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = t2 - c; n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        # U0 tangents: n_i · x = beta0_i, beta0_i = n_i · t_i
        beta01 = float(n1 @ t1)
        beta02 = float(n2 @ t2)

        # (2) Offset by d along the normal: beta' = beta0 + d
        beta1 = beta01 + d
        beta2 = beta02 + d
        R_eff = R_o + d

        # (3) Tangency points on the expanded inner circle: q_i' = c + R_eff * n_i
        q1 = c + R_eff * n1
        q2 = c + R_eff * n2

        # (4) Intersections with the outer circle: pick the one aligned with p→t_i
        u1 = (t1 - p); u1 = u1 / (np.linalg.norm(u1) + 1e-12)
        u2 = (t2 - p); u2 = u2 / (np.linalg.norm(u2) + 1e-12)

        ints1 = self._line_circle_intersections(n1, beta1, p, R_s)
        ints2 = self._line_circle_intersections(n2, beta2, p, R_s)
        if (ints1 is None) or (ints2 is None):
            return None  # Skip if out of sensing range

        cand11, cand12 = ints1
        cand21, cand22 = ints2
        # Pick the intersection most aligned with u_i
        def pick_dir(c1, c2, u):
            direction_a = c1 - p
            direction_a = direction_a / (np.linalg.norm(direction_a) + 1e-12)
            direction_b = c2 - p
            direction_b = direction_b / (np.linalg.norm(direction_b) + 1e-12)
            return c1 if (direction_a @ u) >= (direction_b @ u) else c2

        f1 = pick_dir(cand11, cand12, u1)  # Outer arc endpoint (L1')
        f2 = pick_dir(cand21, cand22, u2)  # Outer arc endpoint (L2')

        # (5) Outer arc (f1→f2), choose the side containing the obstacle
        phi1 = math.atan2(f1[1]-p[1], f1[0]-p[0])
        phi2 = math.atan2(f2[1]-p[1], f2[0]-p[0])
        phi_c = math.atan2(c[1]-p[1],  c[0]-p[0])
        outer_phis = arc_with_mid(phi1, phi2, phi_c, n_arc)
        outer_arc = np.vstack([p[0] + R_s*np.cos(outer_phis),
                            p[1] + R_s*np.sin(outer_phis)]).T

        # (6) Inner expanded arc: complementary arc excluding robot direction (q2→q1)
        th1 = math.atan2(q1[1]-c[1], q1[0]-c[0])
        th2 = math.atan2(q2[1]-c[1], q2[0]-c[0])
        thp = math.atan2(p[1]-c[1],  p[0]-c[0])
        inc = arc_with_mid(th1, th2, thp, n_arc)    # Arc including robot direction
        exc = inc[::-1]                             # Complementary arc (exclude robot direction)
        inner_arc = np.vstack([c[0] + R_eff*np.cos(exc),
                            c[1] + R_eff*np.sin(exc)]).T
        # Ensure inner_arc starts near q2 (direction q2→q1)
        if np.linalg.norm(inner_arc[0]-q2) > np.linalg.norm(inner_arc[-1]-q2):
            inner_arc = inner_arc[::-1]

        # (7) Final polygon: outer_arc(f1→f2) → q2 → inner_arc(q2→q1) → q1
        poly = []
        poly.extend(outer_arc.tolist())
        poly.append(q2.tolist())
        poly.extend(inner_arc.tolist())
        poly.append(q1.tolist())
        return np.asarray(poly), (outer_arc, inner_arc, f1, f2, q1, q2)

    def _ut_halfspaces_params(self, p, c, R_o, t1, t2, d):
        """
        Half-space parameters (n_i, beta_i) for tangents at τ=T,
        and the expanded radius R_eff.
        """
        import numpy as np
        p = np.asarray(p, float); c = np.asarray(c, float)
        t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)

        n1 = t1 - c; n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = t2 - c; n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        beta01 = float(n1 @ t1)
        beta02 = float(n2 @ t2)
        beta1 = beta01 + float(d)
        beta2 = beta02 + float(d)

        R_eff = float(R_o + d)
        return (n1, beta1), (n2, beta2), R_eff

    def _softmin_field_UT(self, XX, YY, p, c, R_s, n1, beta1, n2, beta2, R_eff, kappa):
        R = float(self.robot_radius)

        # c_i(x) >= 0 inside UT
        c1 = (beta1 + R) - (XX * n1[0] + YY * n1[1])                 # halfspace 1
        c2 = (beta2 + R) - (XX * n2[0] + YY * n2[1])                 # halfspace 2
        c3 = (R_s + R) - np.hypot(XX - p[0], YY - p[1])              # inside sensing disc
        c4 = np.hypot(XX - c[0], YY - c[1]) - (R_eff + R)            # outside expanded obstacle

        C = np.stack([c1, c2, c3, c4], axis=2)   # (..., 4)

        # softmin_kappa(C) = Cmin - (1/kappa) * ( log(sum(exp(-kappa*(C - Cmin)))) - log(4) )
        Cmin = np.min(C, axis=2, keepdims=True)
        sumexp = np.exp(-kappa * (C - Cmin)).sum(axis=2, keepdims=False)
        h_tilde = (Cmin.squeeze(-1)
                - (np.log(sumexp) - np.log(4.0)) / float(kappa))
        return h_tilde

    def _softmin_field_UT_masked(self, XX, YY, p, c, R_s,
                                n1, beta1, n2, beta2, R_eff,
                                kappa, wedge_eps=1e-3, band_eps=None):
        """
        Compute h̃_T and mask everything except:
        near-boundary band ∩ inside wedge ∩ inside sensing disk ∩ outside expanded obstacle.
        """
        R = float(self.robot_radius)

        # Signed distances for each constraint (>=0 is inside)
        c1 = (beta1 + R) - (XX * n1[0] + YY * n1[1])              # halfspace 1 (inside when >=0)
        c2 = (beta2 + R) - (XX * n2[0] + YY * n2[1])              # halfspace 2
        c3 = (R_s + R) - np.hypot(XX - p[0], YY - p[1])           # inside sensing disc
        c4 = np.hypot(XX - c[0], YY - c[1]) - (R_eff + R)         # outside expanded obstacle

        # Softmin for intersection (same as main field)
        C = np.stack([c1, c2, c3, c4], axis=2)                    # (..., 4)
        Cmin = np.min(C, axis=2, keepdims=True)
        h_tilde = ( Cmin.squeeze(-1)
                    - ( np.log(np.exp(-kappa*(C - Cmin)).sum(axis=2)) - np.log(4.0) ) / float(kappa) )

        # ----- Masking -----
        # Inside wedge (two halfspaces) with a small margin
        mask_wedge = (c1 >= -wedge_eps) & (c2 >= -wedge_eps)
        # Circle constraints with a small margin.
        ring_eps = 3.0 * max(1e-3, float((XX[0,1]-XX[0,0])))      # ≈ 3 * grid_res
        mask_rings = (c3 >= -ring_eps) & (c4 >= -ring_eps)

        # Keep only a band near the boundary (softmin ~ 0)
        if band_eps is None:
            band_eps = 2.5 * max(1e-3, float((XX[0,1]-XX[0,0])))  # ≈ 2.5 * grid_res
        mask_band = np.abs(h_tilde) <= band_eps

        mask = mask_wedge & mask_rings & mask_band

        Z = h_tilde.copy()
        Z[~mask] = np.nan
        return Z

    def _curved_smax_value(self, pos, scenario, R_s, kappa):
        """
        h̃(x) from the same definition as BackupCBFQP._occlusion_barrier_softmax_curved
        but at tau = 0, using:
          - first two halfspaces (tangent lines)
          - sensing circle
          - obstacle circle
        """
        import numpy as np

        A = scenario.get('A', None)
        b0 = scenario.get('b0', None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None

        p = scenario['robot_pos']
        c = scenario['obs_center']
        R_o = scenario['obs_radius']
        R = self.robot_radius

        pos = np.asarray(pos, float)
        x, y = pos

        a1, a2 = A[3], A[1]
        beta1, beta2 = b0[3], b0[1]

        h1 = a1 @ pos - beta1 - R
        h2 = a2 @ pos - beta2 - R

        # dx_s, dy_s = x - p[0], y - p[1]
        # h3 = dx_s*dx_s + dy_s*dy_s - (R_s + R)**2

        # dx_o, dy_o = x - c[0], y - c[1]
        # h4 = dx_o*dx_o + dy_o*dy_o - (R_o + R)**2
        dx_s, dy_s = x - p[0], y - p[1]
        d_s = np.hypot(dx_s, dy_s)
        h3 = d_s - (R_s + R)           # sensing arc

        dx_o, dy_o = x - c[0], y - c[1]
        d_o = np.hypot(dx_o, dy_o)
        h4 = d_o - (R_o + R)           # obstacle arc

        h_vec = np.array([h1, h2, h3, h4], dtype=float)
        if not np.all(np.isfinite(h_vec)):
            return None

        M = h_vec.size
        if not np.all(np.isfinite(h_vec)):
            return None

        max_h = np.max(h_vec)
        z = np.exp(kappa * (h_vec - max_h))
        Z = np.sum(z)
        if not np.isfinite(Z) or Z <= 0.0:
            return None

        lse = max_h + np.log(Z)
        return float((lse - np.log(M)) / kappa)

    def _clear_artists(self, lst):
        for a in lst:
            try:
                # For contour/contourf artists
                if hasattr(a, "collections"):
                    for c in a.collections:
                        try: c.remove()
                        except: pass
                else:
                    a.remove()
            except Exception:
                pass
        lst.clear()

    def _halfspace_intersection_polygon(self, A, b, eps=1e-9):
        """
        Build polygon vertices of the convex polytope {p | A p <= b}
        by enumerating pairwise line intersections (2D).
        Assumes each row of A is an outward unit normal (as in occlusion._polygon_to_halfspaces).
        """
        import numpy as np

        A = np.asarray(A, float)
        b = np.asarray(b, float).reshape(-1,)
        K = A.shape[0]
        if K < 3:
            return None

        pts = []
        for i in range(K):
            for j in range(i + 1, K):
                Ai = np.stack([A[i], A[j]], axis=0)      # (2,2)
                bi = np.array([b[i], b[j]], dtype=float) # (2,)
                det = np.linalg.det(Ai)
                if abs(det) < 1e-10:
                    continue
                x = np.linalg.solve(Ai, bi)              # (2,)
                if np.all(A @ x <= b + eps):
                    pts.append(x)

        if len(pts) < 3:
            return None

        pts = np.unique(np.round(np.array(pts), 12), axis=0)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        return pts[order]

    def _smax_field_poly(self, XX, YY, A, b0, delta, kappa):
        """
        Vectorized LSE smooth-max field for h_vec = A p - b0 - delta.
        XX,YY: meshgrid arrays
        returns H with same shape as XX
        """
        import numpy as np

        A = np.asarray(A, float)                 # (K,2)
        b0 = np.asarray(b0, float).reshape(-1,)  # (K,)
        K = A.shape[0]

        P = np.stack([XX, YY], axis=2)           # (H,W,2)
        h = P @ A.T - b0[None, None, :] - delta # (H,W,K)

        m = np.max(h, axis=2, keepdims=True)     # (H,W,1)
        z = np.exp(float(kappa) * (h - m))
        Z = np.sum(z, axis=2)                    # (H,W)
        H = (m.squeeze(-1) + np.log(Z) - np.log(float(K))) / float(kappa)
        return H


    def update_occlusion_polygons(self, occlusion_scenarios,
                                  kappa=10.0,
                                  show_true_occ=True,
                                  show_true_occ_T=True,
                                  show_smax_occ_T=True,
                                  T_rollout=None,
                                  grid_res=0.05):

        self._clear_artists(self.occlusion_smax_contours)
        self._clear_artists(self._occ_arc_lines)  # legacy list (no longer used)
        if not show_true_occ and not show_true_occ_T and not show_smax_occ_T:
            for patch in self.occlusion_patches:
                patch.set_visible(False)
            for patch in self.occlusion_future_contours:
                patch.set_visible(False)
            return
        if occlusion_scenarios is None or len(occlusion_scenarios) == 0:
            for patch in self.occlusion_patches:
                patch.set_visible(False)
            for patch in self.occlusion_future_contours:
                patch.set_visible(False)
            return

        import numpy as np
        import matplotlib.patches as patches

        num_scenarios = len(occlusion_scenarios)
        if show_true_occ:
            while len(self.occlusion_patches) < num_scenarios:
                patch = patches.Polygon(
                    np.zeros((3, 2)),
                    closed=True,
                    fill=True,
                    facecolor='gray',
                    edgecolor='none',
                    alpha=0.25,
                    zorder=1
                )
                patch.set_visible(False)
                self.ax.add_patch(patch)
                self.occlusion_patches.append(patch)
        else:
            for patch in self.occlusion_patches:
                patch.set_visible(False)

        if show_true_occ_T:
            while len(self.occlusion_future_contours) < num_scenarios:
                patchT = patches.Polygon(
                    np.zeros((3, 2)),
                    closed=True,
                    fill=True,
                    facecolor="#e41d45",
                    edgecolor='none',
                    alpha=0.22,
                    zorder=0.9
                )
                patchT.set_visible(False)
                self.ax.add_patch(patchT)
                self.occlusion_future_contours.append(patchT)
        else:
            for patch in self.occlusion_future_contours:
                patch.set_visible(False)

        for i, sc in enumerate(occlusion_scenarios):
            # Plot the exact visibility occlusion region when available.
            # Rollout over-approximation remains in sc["poly"] / A / b0.
            poly = sc.get('visibility_poly', sc.get('poly', None))
            A    = sc.get('A', None)
            b0   = sc.get('b0', None)
            if A is None or b0 is None:
                if show_true_occ and i < len(self.occlusion_patches):
                    self.occlusion_patches[i].set_visible(False)
                if show_true_occ_T and i < len(self.occlusion_future_contours):
                    self.occlusion_future_contours[i].set_visible(False)
                continue

            if show_true_occ:
                patch = self.occlusion_patches[i]
                if poly is None:
                    patch.set_visible(False)
                else:
                    patch.set_xy(np.asarray(poly, float))
                    patch.set_visible(True)

            # if no rollout requested, skip T-level plots
            if T_rollout is None or T_rollout <= 0.0:
                if show_true_occ_T and i < len(self.occlusion_future_contours):
                    self.occlusion_future_contours[i].set_visible(False)
                continue

            if 'v_expand_vec' in sc:
                v_exp = sc['v_expand_vec']
            else:
                v_exp = float(sc.get('v_adv_max', 0.0))

            tauT = float(T_rollout)
            deltaT = v_exp * tauT  # matches controller: r_rob + pi_adv*tau

            # --- (B) Build expanded polygon UT from shifted halfspaces: A p <= b0 + deltaT ---
            bT = np.asarray(b0, float).reshape(-1,) + deltaT
            UT = self._halfspace_intersection_polygon(A, bT, eps=1e-8)

            if UT is not None:
                # --- (B1) True UT patch ---
                if show_true_occ_T:
                    patchT = self.occlusion_future_contours[i]
                    patchT.set_xy(UT)
                    patchT.set_visible(True)

                # --- (B2) Smooth-max levelset (h=0) for tau=T ---
                if show_smax_occ_T:
                    xmin, xmax = UT[:, 0].min(), UT[:, 0].max()
                    ymin, ymax = UT[:, 1].min(), UT[:, 1].max()
                    pad = 0.75
                    xs = np.arange(xmin - pad, xmax + pad, grid_res)
                    ys = np.arange(ymin - pad, ymax + pad, grid_res)
                    XX, YY = np.meshgrid(xs, ys)

                    H = self._smax_field_poly(XX, YY, A, b0, deltaT, kappa=float(kappa))

                    ut_color = "#f80101ea"
                    csT = self.ax.contour(
                        XX, YY, H,
                        levels=[0.0],
                        colors=ut_color,
                        linestyles='-',
                        linewidths=1.5,
                        zorder=5
                    )
                    for coll in csT.collections:
                        self.occlusion_smax_contours.append(coll)
            else:
                if show_true_occ_T and i < len(self.occlusion_future_contours):
                    self.occlusion_future_contours[i].set_visible(False)

        if show_true_occ:
            for j in range(num_scenarios, len(self.occlusion_patches)):
                self.occlusion_patches[j].set_visible(False)
        if show_true_occ_T:
            for j in range(num_scenarios, len(self.occlusion_future_contours)):
                self.occlusion_future_contours[j].set_visible(False)

    def set_occ_barrier_fn(self, fn):
        """
        Forward occlusion barrier callback into the underlying robot model
        (e.g., DoubleIntegrator2D).
        """
        self._occ_barrier_cb = fn
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "set_occ_barrier_fn"):
            inner.set_occ_barrier_fn(fn)
        # The base DoubleIntegrator2D does not provide OCC-specific hooks.
        # Keep callback locally for visualization helpers and let controller-layer
        # backup logic handle occlusion rollout.

    def _draw_smax_levelset(self, scenario, tau, bbox, grid_res, color, lw=1.8, ls='-'):
        """
        scenario: dict used by BackupCBFQP (A, b0, robot_pos, obs_center, obs_radius, v_adv_max, ...)
        tau     : 0.0 or T
        bbox    : (xmin, xmax, ymin, ymax)
        """
        if self._occ_barrier_cb is None:
            return  # No callback registered

        xmin, xmax, ymin, ymax = bbox
        xs = np.arange(xmin, xmax, grid_res)
        ys = np.arange(ymin, ymax, grid_res)
        XX, YY = np.meshgrid(xs, ys)

        H = np.full_like(XX, np.nan, dtype=float)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                h_val, _, _ = self._occ_barrier_cb((XX[i, j], YY[i, j]), scenario, tau=tau)
                if (h_val is not None) and np.isfinite(h_val):
                    H[i, j] = h_val

        cs = self.ax.contour(
            XX, YY, H,
            levels=[0.0], colors=color, linewidths=lw, linestyles=ls, zorder=5
        )
        for coll in cs.collections:
            self.occlusion_smax_contours.append(coll)

    def set_terminal_backup_context(self, occlusion_scenarios, T, kappa=None, rho_T=0.05):
        """
        Forward terminal backup context to the underlying model
        If the model does not implement it, store a safe no-op context
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "set_terminal_backup_context"):
            return inner.set_terminal_backup_context(occlusion_scenarios, T, kappa=kappa, rho_T=rho_T)

        # Fallback no-op (store state for velocity-based backup)
        self._term_occ_scenarios = occlusion_scenarios if occlusion_scenarios else []
        self._term_T = float(T)
        self._term_kappa = kappa
        self._term_rho = float(rho_T)
        return None

    def _occ_terminal_set(self, X):
        """
        Terminal backup set value h_b(x_T)
        Delegate to the model if available; otherwise use a speed-based fallback
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "h_b_stop"):
            return inner.h_b_stop(X)

        # Fallback: v_safe^2 - ||v||^2
        v_sq = float(X[2, 0]**2 + X[3, 0]**2)
        v_safe_sq = (0.3)**2
        return v_safe_sq - v_sq

    def grad_occ_terminal(self, X):
        """
        Gradient of h_b_stop
        Delegate to the model if available; otherwise use the speed-based fallback
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "grad_h_b_stop"):
            return inner.grad_h_b_stop(X)

        return np.array([[0, 0, -2 * X[2, 0], -2 * X[3, 0]]])
