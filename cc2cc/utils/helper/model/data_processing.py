import numpy as np

from cc2cc.utils.helper.runtime.env_var import CUBE_MIDDLE

CUBE_INDEX = {
    "cube5": np.array(
        [(0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 2, 0), (2, 0, 2)],
        dtype=np.int64,
    ),
    "cube9": np.array(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (1, 1, 1),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ],
        dtype=np.int64,
    ),
}


def process_input(x, cube_type, input_level, cube_size):
    if cube_type == "center_4":
        return x[:, :input_level, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
    if cube_type == "cube":
        return x[:, :input_level, :, :, :].reshape(
            x.shape[0], input_level, cube_size
        )
    if cube_type in CUBE_INDEX:
        idx = CUBE_INDEX[cube_type]
        return x[
            :,
            :input_level,
            idx[:, 0],
            idx[:, 1],
            idx[:, 2],
        ]
    raise ValueError(f"Unknown cube type: {cube_type}")


def process_grad2force(x, cube_type, input_level, cube_size):
    # tipabcx -> tipCx -> piCtx
    if cube_type == "center_4":
        x = x[
            :, :input_level, :, [CUBE_MIDDLE], CUBE_MIDDLE, CUBE_MIDDLE, :
        ]
    elif cube_type == "cube":
        x = x[:, :input_level, :, :, :, :, :].reshape(
            x.shape[0],
            input_level,
            x.shape[2],
            cube_size,
            x.shape[-1],
        )
    elif cube_type in CUBE_INDEX:
        idx = CUBE_INDEX[cube_type]
        x = x[
            :,
            :input_level,
            :,
            idx[:, 0],
            idx[:, 1],
            idx[:, 2],
            :,
        ]
    else:
        raise ValueError(f"Unknown cube type: {cube_type}")
    # tipCx -> piCtx
    x = np.transpose(x, (2, 1, 3, 0, 4))
    shape_x = x.shape
    return x.reshape(shape_x[0], shape_x[1], shape_x[2], -1)
