"""Continuous-time CBF terms for moving circular obstacles."""

from __future__ import annotations

import numpy as np


def dynamic_agent_barrier(model, X, obs, robot_radius, beta=1.01):
    """Return barrier terms, including explicit time derivatives.

    Obstacles use ``[x, y, radius, vx, vy, ...]``. The return shape follows
    the model's relative degree: Unicycle2D returns ``(h, grad_h, h_t)``;
    DoubleIntegrator2D and DynamicUnicycle2D return
    ``(h, h_dot, grad_h_dot, h_dot_t)``.
    """
    state = np.asarray(X, dtype=float).reshape(-1)
    obstacle = np.asarray(obs, dtype=float).reshape(-1)
    if obstacle.size < 5:
        raise ValueError("Moving-circle obstacles require x, y, radius, vx, vy.")

    rel = state[:2] - obstacle[:2]
    obs_vel = obstacle[3:5]
    radius = float(obstacle[2]) + float(robot_radius)
    h_circle = float(rel @ rel - float(beta) * radius**2)
    model_name = type(model).__name__

    if model_name == "Unicycle2D":
        theta = float(state[2])
        heading = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        heading_perp = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
        projection = float(rel @ heading)
        sigma = float(np.asarray(model.sigma(projection)).reshape(-1)[0])
        sigma_der = float(np.asarray(model.sigma_der(projection)).reshape(-1)[0])
        grad_h = np.array(
            [
                2.0 * rel[0] - sigma_der * heading[0],
                2.0 * rel[1] - sigma_der * heading[1],
                -sigma_der * float(rel @ heading_perp),
            ],
            dtype=float,
        ).reshape(1, -1)
        h_t = float(-2.0 * rel @ obs_vel + sigma_der * (obs_vel @ heading))
        return h_circle - sigma, grad_h, h_t

    if model_name == "DoubleIntegrator2D":
        robot_vel = state[2:4]
        vel_rel = robot_vel - obs_vel
        h_dot = float(2.0 * rel @ vel_rel)
        grad_h_dot = np.concatenate((2.0 * vel_rel, 2.0 * rel)).reshape(1, -1)
        h_dot_t = float(-2.0 * obs_vel @ vel_rel)
        return h_circle, h_dot, grad_h_dot, h_dot_t

    if model_name == "DynamicUnicycle2D":
        theta = float(state[2])
        speed = float(state[3])
        heading = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        heading_perp = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
        vel_rel = speed * heading - obs_vel
        h_dot = float(2.0 * rel @ vel_rel)
        grad_h_dot = np.array(
            [
                2.0 * vel_rel[0],
                2.0 * vel_rel[1],
                2.0 * float(rel @ (speed * heading_perp)),
                2.0 * float(rel @ heading),
            ],
            dtype=float,
        ).reshape(1, -1)
        h_dot_t = float(-2.0 * obs_vel @ vel_rel)
        return h_circle, h_dot, grad_h_dot, h_dot_t

    raise ValueError(f"Unsupported moving-circle model: {model_name}")
