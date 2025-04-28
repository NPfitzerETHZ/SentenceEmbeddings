import jax.numpy as jnp


# TODO: The robot coordinates should be (N, E) not (E, N)
STATE_IDX = {
    "n_pos": jnp.array([0]), 
    "e_pos": jnp.array([1]),
    "pos": jnp.array([0, 1, 2]) ,
    "n_vel": jnp.array([3]), 
    "e_vel": jnp.array([4]),
    "vel": jnp.array([3, 4, 5]) 
}
IDX_STATE = {
    0: "n_pos",
    1: "e_pos",
    3: "n_vel",
    4: "e_vel",
}

# TODO Why are S, N swapped in dataset?
ACTION_IDX = {
    "0": jnp.array(0),
    "W": jnp.array(1),
    "SW": jnp.array(2),
    "N": jnp.array(3),
    "SE": jnp.array(4),
    "E": jnp.array(5),
    "NE": jnp.array(6),
    "S": jnp.array(7),
    "NW": jnp.array(8),
}
ACTION_VEL = {
    "0": jnp.array([0, 0]),
    "W": jnp.array([0, -1]),
    "SW": jnp.array([-1, -1]),
    "S": jnp.array([-1, 0]),
    "SE": jnp.array([-1, 1]),
    "E": jnp.array([0, 1]),
    "NE": jnp.array([1, 1]),
    "N": jnp.array([1, 0]),
    "NW": jnp.array([1, -1]),
}
VELOCITY = 0.3
ACTION_VEL = {k: VELOCITY * (v / jnp.linalg.norm(v)) for k, v in ACTION_VEL.items()}
ACTION_VEL["0"] = jnp.array([0.0, 0.0])
ACTION_MAPPING = {ACTION_IDX[s].item(): ACTION_VEL[s] for s in ACTION_IDX}

ARENA_BOUNDS_N = (-100, 100) # meters
ARENA_BOUNDS_E = (-100, 100) # meters
ROBOT_DIAMETER = 0.4 # meters

NUM_DSCR_ACTIONS = 9
ACTION_TABLE = {
    0 :    jnp.array([     0,   -0.6283]),
    1 :    jnp.array([0.2500,   -0.6283]),
    2 :    jnp.array([0.5000,   -0.6283]),
    3 :         jnp.array([0,         0]),
    4 :    jnp.array([0.2500,         0]),
    5 :    jnp.array([0.5000,         0]),
    6 :         jnp.array([0,    0.6283]),
    7 :    jnp.array([0.2500,    0.6283]),
    8 :    jnp.array([0.5000,    0.6283]),
}
