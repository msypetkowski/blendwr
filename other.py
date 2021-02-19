import time
from collections import defaultdict
from functools import reduce

import bpy
import mathutils


class TimerScope:
    def __init__(self, name, times):
        self.name = name
        self.times = times

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, _type, value, tb):
        self.times[self.name].append(time.time() - self.start_time)


class Timer:
    def __init__(self):
        self.times = defaultdict(list)

    def scope(self, name):
        return TimerScope(name, self.times)

    def print_stats(self):
        print('---------------------------------------')
        for k, v in sorted(self.times.items(), key=lambda x: sum(x[1])):
            print(k, sum(v))
        print('---------------------------------------')


def check_blender_version_ge(ver):
    return tuple(map(int, bpy.app.version_string.split()[0].split('.'))) >= tuple(map(int, ver.split('.')))


def scene_update():
    if check_blender_version_ge('2.80'):
        bpy.context.view_layer.update()
    else:
        bpy.context.scene.update()


def get_cursor_location():
    if check_blender_version_ge('2.80'):
        return bpy.context.scene.cursor.location
    else:
        return bpy.context.scene.cursor_location


def set_cursor_location(loc):
    if check_blender_version_ge('2.80'):
        bpy.context.scene.cursor.location = loc
    else:
        bpy.context.scene.cursor_location = loc


def loc_rot_scale_to_matrix(loc, rot, scale):
    loc = mathutils.Matrix.Translation(loc)
    # rot = [mathutils.Matrix.Rotation(rot[i], 4, axis) for i, axis in enumerate('XYZ')]
    # rot = reduce(matmul, rot[::-1])
    rot = mathutils.Euler(rot, 'XYZ').to_matrix().to_4x4()
    scale = [mathutils.Matrix.Scale(scale[i], 4, axis) for i, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)])]
    scale = reduce(matmul, scale)
    # return list(reduce(matmul, [rot, loc]))
    return reduce(matmul, [loc, rot, scale])
    # return list(reduce(matmul, [scale, rot, loc]))


def _matmul(m1, m2):
    if check_blender_version_ge('2.80'):
        return m1 @ m2
    return m1 * m2


def matmul(*args):
    assert len(args) >= 2
    i = args[0]
    for j in args[1:]:
        i = _matmul(i, j)
    return i


class no_logging:
    def __enter__(self):
        import os, sys
        # redirect output to log file
        logfile = 'blender.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        self.old = old

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os
        # disable output redirection
        os.close(1)
        os.dup(self.old)
        os.close(self.old)


def set_edit_mode():
    bpy.ops.object.mode_set(mode='EDIT')


def set_object_mode():
    bpy.ops.object.mode_set(mode='OBJECT')


def set_vertex_paint_mode():
    bpy.ops.object.mode_set(mode='VERTEX_PAINT')


def set_weight_paint_mode():
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')


def set_pose_mode():
    bpy.ops.object.mode_set(mode='POSE')


def set_pivot(value):
    if check_blender_version_ge('2.80'):
        bpy.context.scene.tool_settings.transform_pivot_point = value
    else:
        bpy.context.space_data.pivot_point = value
