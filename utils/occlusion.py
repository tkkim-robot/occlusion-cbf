import numpy as np
from matplotlib.path import Path

class OcclusionUtils:
    def __init__(self, robot, robot_spec, sensing_range, barrier_fn=None):
        self.robot = robot
        self.robot_spec = robot_spec
        self.sensing_range = sensing_range
        self._occ_barrier_fn = barrier_fn

    def set_occ_barrier_fn(self, fn):
        self._occ_barrier_fn = fn

    def _circle_tangents(self, p, c, R):
        """
        Compute tangent points from point p to a circle centered at c with radius R.

        t1, t2 :
            Tangent points on the circle (each (2,))
            if p is inside/on the circle (no valid tangents).
        """
        p = np.asarray(p, dtype=float).reshape(2,)
        c = np.asarray(c, dtype=float).reshape(2,)
        v = p - c
        d2 = float(v @ v)
        R2 = R * R

        # No tangent if p is inside or on the circle
        if d2 <= R2:
            return None, None

        x1, y1 = v
        x0 = R2 * x1 / d2
        y0 = R2 * y1 / d2
        k = R * np.sqrt(d2 - R2) / d2

        # Tangent points in global coordinates
        t1 = np.array([x0 - y1 * k, y0 + x1 * k]) + c
        t2 = np.array([x0 + y1 * k, y0 - x1 * k]) + c
        return t1, t2
    
    def _polygon_to_halfspaces(self, poly):
        """
        Convert a convex polygon into half-space form: { z | A z <= b }.
        """
        poly = np.asarray(poly, dtype=float)
        M = poly.shape[0]
        if M < 3:
            return None, None

        centroid = np.mean(poly, axis=0)

        A_list = []
        b_list = []

        for i in range(M):
            p1 = poly[i]
            p2 = poly[(i + 1) % M]
            edge = p2 - p1
            if np.linalg.norm(edge) < 1e-9:
                continue

            # Outward normal candidate (right-hand normal)
            n = np.array([edge[1], -edge[0]], dtype=float)
            n /= np.linalg.norm(n)
            
            b = float(n @ p1)

            # Ensure centroid is inside: n^T centroid <= b
            if n @ centroid > b + 1e-9:
                n = -n
                b = -b

            A_list.append(n)
            b_list.append(b)
            
        if len(A_list) == 0:
            return None, None

        A = np.vstack(A_list)        # (M_eff, 2)
        b0 = np.array(b_list)        # (M_eff,)
        
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b0))):
            return None, None
        
        return A, b0
    
    def _point_in_poly(self, pt, poly):
        # poly: (N,2), pt: (2,)
        path = Path(poly)
        return path.contains_point((float(pt[0]), float(pt[1])), radius=1e-12)
    
    def _circle_fully_in_halfspaces(self, center, radius, A, b0, eps=1e-9):
        """
        Return True if a circle is fully inside the convex polygon A z <= b0.
        Assumes rows of A are unit normals (as constructed in _polygon_to_halfspaces).
        """
        if A is None or b0 is None:
            return False
        center = np.asarray(center, dtype=float).reshape(2,)
        radius = float(radius)
        vals = A @ center
        return np.all(vals <= (b0 - radius + eps))

    def _normalize_occ_version(self, version, default="v1"):
        key = default if version is None else str(version).strip().lower()
        if key not in {"v1", "v2"}:
            key = str(default).strip().lower()
        if key not in {"v1", "v2"}:
            key = "v1"
        return key

    def _visibility_occ_version(self):
        return self._normalize_occ_version(
            self.robot_spec.get("occ_visibility_version", "v1"),
            default="v1",
        )

    def _rollout_occ_version(self):
        return self._normalize_occ_version(
            self.robot_spec.get(
                "occ_rollout_version",
                self.robot_spec.get("occ_version", "v1"),
            ),
            default="v1",
        )

    def _occlusion_polygon_v1(self, p, c, R_o, sensing_R, dir1, dir2, t1, t2):
        """
        Legacy occlusion polygon:
        front facet is the chord connecting tangent points [t1, t2].
        """
        far1 = p + sensing_R * dir1
        far2 = p + sensing_R * dir2
        poly_pts = np.vstack([t1, t2, far2, far1])
        return poly_pts, {"far1": far1, "far2": far2, "front1": t1, "front2": t2}

    def _occlusion_polygon_v2(self, p, c, R_o, sensing_R, dir1, dir2, t1, t2):
        """
        Conservative occlusion polygon:
        keep side/rear facets, but move the front facet toward the robot to the
        supporting line perpendicular to the robot-obstacle line-of-sight and
        tangent to the obstacle at the nearest point. This makes the polygon
        include the full visible obstacle body.
        """
        del t1, t2  # front facet no longer uses the tangent chord directly

        los = np.asarray(c, dtype=float).reshape(2,) - np.asarray(p, dtype=float).reshape(2,)
        d = float(np.linalg.norm(los))
        if d <= max(float(R_o), 1e-9):
            return None, None
        n = los / d
        support_pt = np.asarray(c, dtype=float).reshape(2,) - float(R_o) * n
        rhs = float(n @ support_pt)
        p_rhs = float(n @ p)

        denom1 = float(n @ dir1)
        denom2 = float(n @ dir2)
        if abs(denom1) <= 1e-9 or abs(denom2) <= 1e-9:
            return None, None

        lam1 = (rhs - p_rhs) / denom1
        lam2 = (rhs - p_rhs) / denom2
        if (not np.isfinite(lam1)) or (not np.isfinite(lam2)):
            return None, None
        if lam1 <= 1e-9 or lam2 <= 1e-9:
            return None, None
        if lam1 >= sensing_R or lam2 >= sensing_R:
            return None, None

        front1 = p + lam1 * dir1
        front2 = p + lam2 * dir2
        far1 = p + sensing_R * dir1
        far2 = p + sensing_R * dir2
        poly_pts = np.vstack([front1, front2, far2, far1])
        return poly_pts, {
            "far1": far1,
            "far2": far2,
            "front1": front1,
            "front2": front2,
            "support_point": support_pt,
            "los_unit": n,
        }
    
    def _build_occlusion_scenario(self, robot_state, obs, is_static=False, occ_version_override=None):
        """
        Build an occlusion scenario for a single circular obstacle.

        The occlusion region is a wedge from the robot through the tangent points to a far range, then converted to half-spaces.

        scenario :
            {
              'A'         : (M_k, 2) half-space normals
              'b0'        : (M_k,) offsets
              'v_adv_max' : float, adversary speed bound
              'poly'      : (4, 2) occlusion polygon vertices
            }
            Returns None if no valid occlusion is formed.
        Polygon Order: [front1, front2, far2, far1] where
        # Edge 0: front1 -> front2 (Front Facet)
        # Edge 1: front2 -> far2   (Side Facet 1)
        # Edge 2: far2 -> far1     (Back Facet)
        # Edge 3: far1 -> front1   (Side Facet 2)
        """

        px = float(robot_state[0, 0])
        py = float(robot_state[1, 0])
        p = np.array([px, py])

        obs = np.asarray(obs).flatten()
        ox, oy, r_obs = obs[:3]
        c = np.array([ox, oy])
        R_o = float(r_obs)

        sensing_R = self.sensing_range
        v_adv = float(self.robot_spec.get('v_adv_max_occ', 0.5))
        
        p_rel = np.array([[px - ox], 
                        [py - oy]])
        
        p_rel_mag = np.linalg.norm(p_rel)
        p_rel_mag = max(p_rel_mag, 1e-6)
        arc_adv = v_adv * p_rel / p_rel_mag
        arc_adv = arc_adv.flatten()

        # Ignore obstacle if it is outside sensing range
        d = np.linalg.norm(c - p)
        if d >= sensing_R:
            return None

        # Compute tangent points from robot to obstacle
        t1, t2 = self._circle_tangents(p, c, R_o)
        if t1 is None or t2 is None:
            return None

        # Extend tangent directions to sensing range to form occlusion wedge
        dir1 = t1 - p
        n1 = np.linalg.norm(dir1)
        if n1 < 1e-6:
            return None
        dir1 /= n1  # tangent 1 unit vector
        
        dir2 = t2 - p
        n2 = np.linalg.norm(dir2)
        if n2 < 1e-6:
            return None
        dir2 /= n2  # tangenet 2 unit vector

        occ_version = self._rollout_occ_version()
        if occ_version_override is not None:
            occ_version = self._normalize_occ_version(occ_version_override, default=occ_version)

        if occ_version == "v2":
            poly_pts, geom_meta = self._occlusion_polygon_v2(p, c, R_o, sensing_R, dir1, dir2, t1, t2)
            if poly_pts is None:
                poly_pts, geom_meta = self._occlusion_polygon_v1(p, c, R_o, sensing_R, dir1, dir2, t1, t2)
                occ_version = "v1_fallback"
        else:
            poly_pts, geom_meta = self._occlusion_polygon_v1(p, c, R_o, sensing_R, dir1, dir2, t1, t2)

        A, b0 = self._polygon_to_halfspaces(poly_pts)
        if A is None:
            return None

        if occ_version.startswith("v2") and not self._circle_fully_in_halfspaces(c, R_o, A, b0):
            poly_pts, geom_meta = self._occlusion_polygon_v1(p, c, R_o, sensing_R, dir1, dir2, t1, t2)
            A, b0 = self._polygon_to_halfspaces(poly_pts)
            if A is None:
                return None
            occ_version = "v1_fallback"

        # generate expand velocity vectors (default is all v_adv)
        v_expand_vec = np.full(len(b0), v_adv)

        # Optional experiment mode:
        # disable expansion only on the front facet (edge t1->t2), i.e., row 0.
        # This facet is the one facing the robot along the current LoS geometry.
        if bool(self.robot_spec.get("occ_disable_front_facet_expand", False)):
            if len(v_expand_vec) > 0:
                v_expand_vec[0] = 0.0

        if is_static:
            # Keep legacy behavior for static occluders.
            v_expand_vec[0] = 0.0
        
        scenario = {
            'A': A,
            'b0': b0,
            'v_expand_vec': v_expand_vec,
            'v_adv_max': v_adv,
            'arc_adv': arc_adv,
            'poly': poly_pts,
            'occ_version': occ_version,
            ## For arc softmax
            'robot_pos': p,
            'obs_center': c,
            'obs_radius': R_o,
            't1': t1,
            't2': t2,
        }
        if geom_meta:
            scenario.update(geom_meta)
        
        fn = getattr(self, "_occ_barrier_fn", None)
        if fn is None:
            return scenario

        _, _, risk_vec = fn(p, scenario, tau=0.0)
        if risk_vec is not None:
            risk_vec = risk_vec * v_adv
            scenario['risk_normal_vec'] = risk_vec
        
        return scenario
    
    def _filter_visible_and_build_occ(self, robot_state, obs_list, return_indices=False):

        visible_obs = []
        occl_scenarios = []
        visibility_scenarios = []
        visible_indices = []

        if obs_list is None:
            if return_indices:
                return visible_obs, occl_scenarios, visible_indices
            return visible_obs, occl_scenarios

        obs_arr = np.array(obs_list, dtype=float)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        px, py = float(robot_state[0, 0]), float(robot_state[1, 0])
        p = np.array([px, py])
        R_sense2 = self.sensing_range ** 2

        keep = []
        for k, o in enumerate(obs_arr):
            if (o[0]-px)**2 + (o[1]-py)**2 <= R_sense2:
                keep.append(k)
        if not keep:
            if return_indices:
                return visible_obs, occl_scenarios, visible_indices
            return visible_obs, occl_scenarios

        obs_arr = obs_arr[keep]

        occ_types = self.robot_spec.get('occlusion_types', None)
        if occ_types is not None:
            occ_types = {int(t) for t in occ_types}

        vis_scale = float(self.robot_spec.get('occ_visible_scale', 1.0))
        if not np.isfinite(vis_scale):
            vis_scale = 1.0
        if vis_scale < 0.1:
            vis_scale = 0.1
        if vis_scale > 1.0:
            vis_scale = 1.0

        dists = np.linalg.norm(obs_arr[:, :2] - p[None, :], axis=1)
        order = np.argsort(dists)

        visibility_occ_version = self._visibility_occ_version()
        rollout_occ_version = self._rollout_occ_version()

        for idx in order:
            obs = obs_arr[idx]
            c = obs[:2]
            r_obs = float(obs[2]) if obs.shape[0] >= 3 else 0.0
            r_occ = r_obs * vis_scale

            occluded = any(
                self._circle_fully_in_halfspaces(c, r_occ, sc.get('A'), sc.get('b0'))
                for sc in visibility_scenarios
            )
            if occluded:
                continue

            visible_obs.append(obs)
            if return_indices:
                visible_indices.append(int(keep[int(idx)]))

            # verify type flag
            obs_type = None
            if len(obs) >= 8:
                obs_type = int(obs[7])
                is_static_obs = (obs_type == 0) # 0: Static, 1: Dynamic
            else:
                is_static_obs = False

            if occ_types is not None:
                if obs_type is None or obs_type not in occ_types:
                    continue

            vis_sc = self._build_occlusion_scenario(
                robot_state,
                obs,
                is_static=is_static_obs,
                occ_version_override=visibility_occ_version,
            )
            if vis_sc is not None and vis_sc.get('poly') is not None:
                visibility_scenarios.append(vis_sc)

            rollout_sc = vis_sc
            if rollout_occ_version != visibility_occ_version:
                rollout_sc = self._build_occlusion_scenario(
                    robot_state,
                    obs,
                    is_static=is_static_obs,
                    occ_version_override=rollout_occ_version,
                )
            if rollout_sc is not None and rollout_sc.get('poly') is not None:
                occl_scenarios.append(rollout_sc)

        if return_indices:
            return visible_obs, occl_scenarios, visible_indices
        return visible_obs, occl_scenarios
    
    def _u_pi_at(self, x, scenarios, t=0.0):
        if hasattr(self.robot, "backup_input_at"):
            return self.robot.backup_input_at(x, scenarios,t=t)

        if (scenarios is not None) and hasattr(self.robot, "backup_input_occlusion"):
            u = self.robot.backup_input_occlusion(x, scenarios, t=t)
            if u is not None:
                return u

        if hasattr(self.robot, "backup_input"):
            u = self.robot.backup_input(x)
            if u is not None:
                return u

        if hasattr(self.robot, "stop"):
            return self.robot.stop(x)
        return np.zeros((2,1), dtype=float)
