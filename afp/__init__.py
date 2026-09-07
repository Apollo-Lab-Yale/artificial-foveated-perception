"""
AFP (Artificial Foveated Perception) - inference-only package.

A compact task-conditioned mask predictor. Given RGB frames and a task
description it predicts a continuous mask in [0, 1] over the task-relevant
objects and the robot end-effector. The mask is used as an auxiliary
attention-grounding signal when fine-tuning a robotic foundation model
(see afp_integrations/).
"""

from .inference import AFPModel

__all__ = ['AFPModel']
