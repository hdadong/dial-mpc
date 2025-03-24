from brax import math

import jax.numpy as jnp
import jax


def global_to_body_velocity(v, q):
    """Transforms global velocity to body velocity."""
    # rotate v by inverse of q
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    """Transforms body velocity to global velocity."""
    return math.rotate(v, q)


@jax.jit
def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps

# @jax.jit
# def get_foot_step(duty_ratio, cadence, amplitude, phases, time, gravity=9.81):
#     """
#     Compute the foot step height for a jump where both feet lift off the ground at the same time.
#     Args:
#         amplitude: The peak height of the jump.
#         cadence: The cadence of the jump (jumps per second).
#         time: The current time in seconds.
#         gravity: Gravitational constant (default is 9.81 m/s^2).
#     """
#     amplitude = amplitude # Assume symmetry in jump motion
#     # Total time for one jump (up and down)
#     jump_duration = 1 / cadence
    
#     # Time spent in the air (assuming symmetry in jump motion)
#     air_time = jump_duration / 2

#     # Calculate upward and downward motion based on time
#     def jump_motion(t, amplitude, air_time, gravity):
#         # If in the upward phase
#         height_up = amplitude * (t / air_time)
        
#         # If in the falling phase
#         height_down = amplitude - 0.5 * gravity * ((t - air_time) ** 2)
        
#         # Use height_up if within air_time, otherwise switch to height_down
#         foot_height = jnp.where(t < air_time, height_up, height_down)
        
#         # Ensure foot height doesn’t go below ground level
#         return jnp.maximum(foot_height, 0.0)

#     # Calculate foot height at the given time
#     h_jump = jump_motion(time % jump_duration, amplitude, air_time, gravity)
    
#     return h_jump

# @jax.jit
# def get_foot_step(duty_ratio, cadence, amplitude, phases, time, gravity=9.81):
#     """
#     Compute the squat motion height for a humanoid robot without lifting feet.
#     Args:
#         amplitude: The maximum depth of the squat.
#         cadence: The cadence of the squat (squats per second).
#         time: The current time in seconds.
#     """
#     # Squat duration based on cadence
#     squat_duration = 1 / cadence
    
#     # Time within each squat cycle
#     t_rel = time % squat_duration

#     # Downward and upward motion for a smooth squat
#     descent = amplitude * (t_rel / (squat_duration / 2))
#     ascent = amplitude * (1 - ((t_rel - squat_duration / 2) / (squat_duration / 2)))

#     # Use descent if in the first half, otherwise use ascent
#     height = jnp.where(t_rel < squat_duration / 2, descent, ascent)
    
#     # Ensure feet stay grounded by returning only positive height values
#     return jnp.clip(height, 0, amplitude)
