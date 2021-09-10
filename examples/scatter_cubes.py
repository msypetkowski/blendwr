import random

from .. import edit as blwr_edit
from .. import object as blwr_obj


def scatter_cubes(seed, n=50):
    base_obj = blwr_obj.create_empty_mesh()
    with blwr_edit.EditMesh(base_obj) as editor:
        editor.make_box((.5, .5, .5))

    rand = random.Random(seed)
    cubes = []
    for _ in range(n):
        cubes.append(blwr_obj.duplicate(base_obj))
        blwr_obj.translate(cubes[-1], [rand.random() for _ in range(3)])

    blwr_obj.remove(base_obj)
    ret = blwr_obj.join(*cubes)
    ret.name = 'splattered_cubes'
    return ret
