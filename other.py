import bpy
import mathutils
import time
from collections import defaultdict
from functools import reduce
from pathlib import Path


class TimerScope:
    """ Profiling is an inseparable part of developing software using blender API.
    This is just a simple util. This library may contain more complex utils in the future. """

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


def loc_rot_scale_to_matrix(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    loc = mathutils.Matrix.Translation(loc)
    rot = mathutils.Euler(rot, 'XYZ').to_matrix().to_4x4()
    scale = [mathutils.Matrix.Scale(scale[i], 4, axis) for i, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)])]
    scale = reduce(matmul, scale)
    return reduce(matmul, [loc, rot, scale])


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
    """ Some blender operators output variety of annoying logs,
    use this context manager to mute them """

    def __enter__(self):
        import os, sys, tempfile
        # redirect output to log file
        self.temp_dir = tempfile.TemporaryDirectory()
        logfile = self.temp_dir.name + '/blender.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        self.old = old

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os
        # disable output redirection
        self.temp_dir.cleanup()
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


def scene_raycast(origin, direction):
    return bpy.context.scene.ray_cast(
        origin=origin,
        direction=direction,
        depsgraph=bpy.context.view_layer.depsgraph)


def save_image(path, img):
    import cv2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def read_image(path):
    import cv2
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _copy_atttributes_generic(attributes, source, target):
    """ Only copies attributes that are copyable (not read-only)"""
    for attr in attributes:
        if hasattr(target, attr):
            try:
                setattr(target, attr, getattr(source, attr))
            except AttributeError:
                pass


def copy_nodes(src_nodes, group):
    """ Copy given nodes into a given node group """
    # TODO: refactor
    input_attributes = ("default_value", "name")
    output_attributes = ("default_value", "name")

    for src_node in src_nodes:
        target_node = group.nodes.new(src_node.bl_idname)
        attributes = []
        for attr in src_node.bl_rna.properties:
            if not attr.identifier.split("_")[0] == "bl":
                attributes.append(attr.identifier)
        _copy_atttributes_generic(attributes, src_node, target_node)

        for i, inp in enumerate(src_node.inputs):
            _copy_atttributes_generic(input_attributes, inp, target_node.inputs[i])

        for i, out in enumerate(src_node.outputs):
            _copy_atttributes_generic(output_attributes, out, target_node.outputs[i])


def mirror_links(nodes, group):
    """ Setup links in group as it is in given nodes list.
    Use node names to find corresponding ones. """
    for node in nodes:
        group_node = group.nodes[node.name]
        for i, inp in enumerate(node.inputs):
            for link in inp.links:
                if link.from_node.name not in group.nodes:
                    continue
                corresponding_node = group.nodes[link.from_node.name]
                group.links.new(corresponding_node.outputs[link.from_socket.name], group_node.inputs[i])


def create_node_group(nodes, name='NodeGroup'):
    """ Create new node group by copying given nodes """
    group = bpy.data.node_groups.new(name=name, type="ShaderNodeTree")
    copy_nodes(nodes, group)
    mirror_links(nodes, group)
    return group


def make_blender_image(img, img_save_path):
    Path(img_save_path).parent.mkdir(exist_ok=True, parents=True)
    save_image(img_save_path, img)
    return bpy.data.images.load(str(img_save_path))


def render_to_numpy(disable_logging=True):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        ff = bpy.context.scene.render.image_settings.file_format
        if ff == 'OPEN_EXR':
            extension = '.exr'
        elif ff == 'JPEG':
            extension = '.jpg'
        elif ff == 'PNG':
            extension = '.png'
        else:
            raise NotImplementedError(f'Unsupported file format {ff}')
        path = tmpdirname + f'/render{extension}'
        bpy.data.scenes["Scene"].render.filepath = path
        scene_update()
        if disable_logging:
            with no_logging():
                bpy.ops.render.render(write_still=True)
        else:
            bpy.ops.render.render(write_still=True)
        return read_image(path)
