import random

from .. import edit as blwr_edit
from .. import object as blwr_obj


def simple_terrain(seed, detail_levels=9, height_multiplier=.4):
    rand = random.Random(seed)
    obj = blwr_obj.create_empty_mesh()
    with blwr_edit.EditMesh(obj) as editor:
        editor.bm.faces.new([editor.bm.verts.new(co + [0]) for co in [[1, 0], [1, 1], [0, 1], [0, 0]]])
        for detail_level in range(detail_levels):
            editor.select_all()
            editor.subdivide()
            editor.deselect_all()
            editor.select(rand.sample(list(editor.bm.verts), k=len(editor.bm.verts) // 3))
            scale = 2 ** (-detail_level)
            editor.translate([*[rand.uniform(-scale / 5, scale / 5) for _ in range(2)],
                              height_multiplier * scale])
    blwr_obj.shade_smooth(obj)
    return obj
