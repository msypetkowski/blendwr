import math
from pathlib import Path

import bpy
import cv2
import mathutils
import numpy as np

from . import edit as blwr_edit
from . import other as blwr_oth


def multi_object(fun):
    """
    Allow passing list of objects to decorated function.
    Applies function for each object in list and returns list of results.
    """

    def optional_list(*args, **kwargs):
        objs, *args = args
        if isinstance(objs, list):
            return [fun(obj, *args, **kwargs) for obj in objs]
        return fun(objs, *args, **kwargs)

    return optional_list


def get_selected():
    return [o for o in bpy.data.objects if select_get(o)]


def get_single_selected():
    objects = get_selected()
    if len(objects) != 1:
        raise IndexError("Exactly one object should be selected.")
    return objects[0]


def select_source_target(*objs):
    """ Deselect all objects, and make the last one active.
    """
    deselect_all()
    for obj in objs:
        select_set(obj, True)
    set_active(objs[-1])
    blwr_oth.scene_update()


@multi_object
def select(obj):
    select_set(obj, True)


def focus(obj, update=False):
    """ Make Object active, select only this object.
    """
    deselect_all()
    select_set(obj, True)
    set_active(obj)
    if update:
        blwr_oth.scene_update()


@multi_object
def remove(obj):
    bpy.data.objects.remove(obj, do_unlink=True)


def get_child_objects(obj):
    return [o for o in bpy.data.objects if o.parent == obj]


@multi_object
def apply_modifier(obj, modifier):
    focus(obj)
    if blwr_oth.check_blender_version_ge('2.90'):
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    else:
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifier.name)


@multi_object
def use_modifier(obj, mod_type, apply=True, use_as_first=False, **kwargs):
    name = 'temp_modifier' if apply else mod_type.lower()
    modifier = obj.modifiers.new(name=name, type=mod_type)

    for k, v in kwargs.items():
        if isinstance(v, dict):
            attr = getattr(modifier, k)
            for k, v in v.items():
                setattr(attr, k, v)
        else:
            setattr(modifier, k, v)
    focus(obj)

    if use_as_first:
        for _ in range(len(obj.modifiers) - 1):
            bpy.ops.object.modifier_move_up(modifier=name)
        blwr_oth.scene_update()

    if apply:
        apply_modifier(obj, modifier)
    else:
        return modifier


@multi_object
def select_set(obj, val):
    if blwr_oth.check_blender_version_ge('2.80'):
        obj.select_set(val)
    else:
        obj.select = val


@multi_object
def select_get(obj):
    if blwr_oth.check_blender_version_ge('2.80'):
        return obj.select_get()
    return obj.select


def set_active(obj):
    if blwr_oth.check_blender_version_ge('2.80'):
        bpy.context.view_layer.objects.active = obj
    else:
        bpy.context.scene.objects.active = obj


def get_active():
    if blwr_oth.check_blender_version_ge('2.80'):
        return bpy.context.view_layer.objects.active
    else:
        return bpy.context.scene.objects.active


def create_empty_mesh(name='EmptyMeshObject'):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([[]], [], [])
    obj = bpy.data.objects.new(name, mesh)
    add_to_scene(obj)
    blwr_oth.scene_update()
    focus(obj)
    with blwr_edit.EditMesh(obj) as editor:
        editor.select_all()
        editor.remove_selected()  # TODO: why sometimes there is anything at all
    return obj


def create_text_object(name='EmptyTextObject', text=""):
    font_curve = bpy.data.curves.new(type="FONT", name="font_curve")
    obj = bpy.data.objects.new(name, font_curve)
    obj.data.body = text
    add_to_scene(obj)
    return obj


def create_sphere(radius=1, rings=16, segments=32, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, ring_count=rings, segments=segments,
                                         align='WORLD', location=location, scale=(1, 1, 1))
    return get_single_selected()


@multi_object
def add_to_scene(obj):
    if blwr_oth.check_blender_version_ge('2.80'):
        bpy.context.collection.objects.link(obj)
    else:
        bpy.context.scene.objects.link(obj)


@multi_object
def duplicate(obj, reference=False, name='Object'):
    """ Returns newly created object
    """
    if reference:
        new_obj = bpy.data.objects.new(name, obj.data)
        new_obj.matrix_world = obj.matrix_world
    else:
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        new_obj.name = name
    add_to_scene(new_obj)
    return new_obj


@multi_object
def apply_transform(obj):
    focus(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


@multi_object
def translate(obj, value):
    focus(obj)
    bpy.ops.transform.translate(value=value, orient_type='GLOBAL')


@multi_object
def resize(obj, value, orient_type='GLOBAL'):
    focus(obj)
    bpy.ops.transform.resize(value=value, orient_type=orient_type)


@multi_object
def rotate(obj, value, pivot='MEDIAN_POINT'):
    blwr_oth.set_pivot(pivot)
    focus(obj)
    mat = mathutils.Matrix()
    for v, ax in zip(value, 'XYZ'):
        mat = blwr_oth.matmul(mathutils.Matrix.Rotation(v, 4, ax), mat)
    obj.matrix_world = blwr_oth.matmul(mat, obj.matrix_world)


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')


@multi_object
def break_to_components(obj):
    ret = []
    with blwr_edit.EditMesh(obj) as editor:
        editor.deselect_all()
        editor.set_selection_mode(True, False, False)
        while len(editor.bm.verts):
            editor.bm.verts[0].select = True
            editor.select_linked()
            ret.append(editor.separate_selection(duplicate=False))
            editor.update()
    remove(obj)
    return ret


@multi_object
def boolean(obj, other_obj, operation='DIFFERENCE', break_obj=False, **kwargs):
    """
    :param break_obj: apply boolean separately to each connected component of obj.
    """
    if not break_obj:
        use_modifier(obj, 'BOOLEAN',
                     operation=operation,
                     object=other_obj,
                     **kwargs)
        return obj

    result = []
    for obj in break_to_components(obj):
        use_modifier(obj, 'BOOLEAN',
                     operation=operation,
                     object=other_obj,
                     **kwargs)
        result.append(obj)

    return join(*result)


def join(*objects, boolean_union=False):
    """ Join given objects to last of them.
    """
    if len(objects) == 0:
        return None
    if len(objects) == 1:
        return objects[0]
    if not boolean_union:
        bpy.ops.object.select_all(action='TOGGLE')
        select_source_target(*objects)
        bpy.ops.object.join()
        return objects[-1]
    *objs, last = objects
    for obj in objs:
        boolean(last, obj, operation='UNION')
        focus(obj)
        bpy.ops.object.delete(use_global=False)
    focus(last)
    return last


@multi_object
def mirror(obj, **kwargs):
    return use_modifier(obj, 'MIRROR', **kwargs)


@multi_object
def bevel(obj, offset=0.02, segments=1, clamp_overlap=False, **kwargs):
    return use_modifier(obj, 'BEVEL', width=offset, segments=segments,
                        use_clamp_overlap=clamp_overlap, **kwargs)


@multi_object
def shrinkwrap(obj, target, vertex_group=None, wrap_mode='ON_SURFACE', offset=0, **kwargs):
    if vertex_group:
        kwargs['vertex_group'] = vertex_group
    use_modifier(obj, 'SHRINKWRAP', wrap_mode=wrap_mode, offset=offset, target=target, **kwargs)


@multi_object
def bounding_box(obj):
    return [[f([obj.bound_box[j][i] for j in range(8)]) for f in [min, max]] for i in range(3)]


@multi_object
def bounding_box_center(obj):
    bbox = bounding_box(obj)
    return np.mean(bbox, axis=1)


def get_image_as_plane_obj(img, path):
    """
    :type path: Path
    :type img: np.ndarray
    :return: object with material using image texture
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    if len(img.shape) == 3:
        cv2.imwrite(str(path), img[..., ::-1])
    else:
        cv2.imwrite(str(path), img)
    img = bpy.data.images.load(str(path))
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane_obj = get_single_selected()
    focus(plane_obj)
    bpy.ops.object.material_slot_add()

    material = bpy.data.materials.new(str(path))
    plane_obj.data.materials[0] = material
    material.use_nodes = True
    tree = material.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs['Roughness'].default_value = 1
    image_node = tree.nodes.new('ShaderNodeTexImage')
    image_node.image = img
    tree.links.new(image_node.outputs[0], principled.inputs['Base Color'])

    return plane_obj


@multi_object
def hide(obj, hide=True):
    obj.hide_set(hide)


@multi_object
def subdivide(obj, times=1, **kwargs):
    if not times:
        return
    kwargs = dict()
    if blwr_oth.check_blender_version_ge('2.80'):
        kwargs['uv_smooth'] = "NONE"
    else:
        kwargs['use_subsurf_uv'] = False
    kwargs['levels'] = times
    kwargs['render_levels'] = times
    use_modifier(obj, 'SUBSURF', **kwargs)


@multi_object
def symmetrize(obj):
    with blwr_edit.EditMesh(obj) as editor:
        editor.select_all()
        editor.symmetrize()


@multi_object
def origin_to_geometry(obj):
    focus(obj)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')


@multi_object
def flip_normals(obj):
    focus(obj)
    blwr_oth.set_edit_mode()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    blwr_oth.set_object_mode()


@multi_object
def solidify(obj, thickness, offset=0.0, use_rim_only=False, vgroup_name='', **kwargs):
    use_modifier(obj, 'SOLIDIFY',
                 thickness=thickness,
                 offset=offset,
                 use_rim_only=use_rim_only,
                 vertex_group=vgroup_name,
                 **kwargs)


@multi_object
def shade_smooth(obj):
    focus(obj)
    bpy.ops.object.shade_smooth()


@multi_object
def shade_flat(obj):
    focus(obj)
    bpy.ops.object.shade_flat()


@multi_object
def set_origin(obj, loc):
    old_loc = blwr_oth.get_cursor_location()
    blwr_oth.set_cursor_location(loc)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    blwr_oth.set_cursor_location(old_loc)


def place_in_grid(objs, margin=1, nrows=5):
    offset_x = 0
    offset_y = 0
    max_dim1 = 0
    for i, o in enumerate(objs):
        max_dim1 = max(max_dim1, o.dimensions[1])
        o.location[0] += offset_x
        o.location[1] += offset_y
        offset_x += round(o.dimensions[0] + margin)
        if (i + 1) % nrows == 0:
            offset_y += round(max_dim1 + margin)
            offset_x = 0
            max_dim1 = 0


@multi_object
def reparent(obj, new_parent=None, keep_transform=True):
    if new_parent is None:
        focus(obj)
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM' if keep_transform else 'CLEAR')
        return
    select_source_target(obj, new_parent)
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=keep_transform)


@multi_object
def edge_split(obj, split_angle=math.pi / 6, **kwargs):
    use_modifier(obj, 'EDGE_SPLIT', split_angle=split_angle, **kwargs)


@multi_object
def array(obj, count, offset, **kwargs):
    use_modifier(obj, 'ARRAY',
                 count=count,
                 constant_offset_displace=offset,
                 use_constant_offset=True,
                 use_relative_offset=False,
                 **kwargs)


@multi_object
def assign_material(obj, material):
    focus(obj)
    if not len(obj.material_slots):
        bpy.ops.object.material_slot_add()
    obj.material_slots[0].material = material


def make_joined_duplicate(objs):
    duplicates = [duplicate(o) for o in objs]
    return join(*duplicates)


@multi_object
def remove_material_slot(obj, idx):
    focus(obj)
    for x in bpy.context.object.material_slots:
        bpy.context.object.active_material_index = idx
        bpy.ops.object.material_slot_remove()
