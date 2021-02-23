import math
import queue
from itertools import product

import bmesh
import bpy
import numpy as np

from . import object as blwr_obj
from . import other as blwr_oth


def get_connected_components(bm, vert_indices):
    vset = set(vert_indices)
    colors = [-1 for v in range(len(bm.verts))]

    def bfs(i, color):
        q = queue.Queue()
        q.put(i)
        colors[i] = color
        while not q.empty():
            i = q.get(i)
            for e in bm.verts[i].link_edges:
                v, v2 = e.verts
                if v.index == i:
                    v = v2
                if v.index not in vset:
                    continue
                if colors[v.index] == -1:
                    q.put(v.index)
                    colors[v.index] = color

    cur_color = 0
    for v in vert_indices:
        if colors[v] == -1:
            bfs(v, cur_color)
            cur_color += 1

    ret = [[] for _ in range(cur_color)]
    for vi in vert_indices:
        assert colors[vi] >= 0
        ret[colors[vi]].append(vi)
    return ret


class EditMesh(object):

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        blwr_obj.focus(self.obj, update=True)
        ret = MeshEditor(self.obj)
        blwr_oth.set_edit_mode()
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        blwr_oth.set_object_mode()


class MeshEditor:
    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.set_selection_mode(True, False, False)
        self.update()

        self.use_proportional_edit = False
        self.proportional_edit_falloff = 'LINEAR'
        self.use_proportional_connected = False

        self.bvh = None

    def select_all(self):
        bpy.ops.mesh.select_all(action='SELECT')

    def deselect_all(self):
        bpy.ops.mesh.select_all(action='DESELECT')

    def select(self, elements, select=True):
        for e in elements:
            e.select = select

    def select_verts(self, indices, select=True):
        for vi in indices:
            self.bm.verts[vi].select = select

    def select_edges(self, indices, select=True):
        for i in indices:
            self.bm.edges[i].select = select

    def select_faces(self, indices=None, select=True, lambd=None):
        if indices is not None:
            for i in indices:
                self.bm.faces[i].select = select
        if lambd is not None:
            for f in self.bm.faces:
                if lambd(f):
                    f.select = select

    def update(self):
        if hasattr(self, 'bm'):
            self.select_flush()
        blwr_oth.set_object_mode()
        blwr_oth.set_edit_mode()
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()

        self.bvh = None

    def extrude(self, value, select_old=False):
        """ TODO: think of a method to avoid calling self.update every time
        """
        old_selection_mode = self.get_selection_mode()
        self.set_selection_mode(True, True, True)

        selected = self.get_selected_verts(indices=False)
        selected.extend(self.get_selected_edges(indices=False))
        selected.extend(self.get_selected_faces(indices=False))

        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={
            "value": value, "orient_type": 'GLOBAL'})

        if select_old:
            for f in selected:
                f.select = True

        self.set_selection_mode(*old_selection_mode)
        self.update()

    def extrude_along_normals(self, value, select_old=False):
        bmesh.update_edit_mesh(self.mesh)
        if select_old:
            old = self.get_selected_verts()

        bpy.ops.mesh.extrude_region_shrink_fatten(
            MESH_OT_extrude_region={"use_normal_flip": False, "mirror": False},
            TRANSFORM_OT_shrink_fatten={
                "value": -value,
                "use_even_offset": False,
                "mirror": False,
                "use_proportional_edit": False,
                "proportional_edit_falloff": 'SMOOTH',
                "proportional_size": 1,
                "use_proportional_connected": False,
                "use_proportional_projected": False,
                "snap": False,
                "snap_target": 'CLOSEST',
                "snap_point": (0, 0, 0),
                "snap_align": False, "snap_normal": (0, 0, 0),
                "release_confirm": False,
                "use_accurate": False
            }
        )

        if select_old:
            self.update()
            self.select_verts(old)
            self.update()

    def add_vert(self, loc):
        index = self.bm.verts.new(loc).index
        index = len(self.bm.verts) - 1  # TODO: is it correct?
        self.bm.verts.ensure_lookup_table()
        return index

    def get_selected_verts(self, indices=True):
        if indices:
            return [i for i, v in enumerate(self.bm.verts) if v.select]
        return [v for v in self.bm.verts if v.select]

    def get_selected_edges(self, indices=True):
        if indices:
            return [i for i, e in enumerate(self.bm.edges) if e.select]
        return [e for e in self.bm.edges if e.select]

    def get_selected_faces(self, indices=True):
        if indices:
            return [i for i, f in enumerate(self.bm.faces) if f.select]
        return [f for f in self.bm.faces if f.select]

    def remove_selected(self, mode='VERT'):
        blwr_oth.scene_update()
        with blwr_oth.no_logging():
            bpy.ops.mesh.delete(type=mode)
        self.update()

    def translate(self, value, **kwargs):
        kwargs['value'] = value
        if blwr_oth.check_blender_version_ge('2.80'):
            rename_map = [('constraint_orientation', 'orient_type')]
            for k, v in rename_map:
                if k in kwargs:
                    kwargs[v] = kwargs[k]
                    kwargs.pop(k)
            if 'proportional' in kwargs:
                kwargs['use_proportional_edit'] = kwargs['proportional'] == 'ENABLED'
                kwargs.pop('proportional')
        bpy.ops.transform.translate(**kwargs)

    def resize(self, value, proportional_size=1):
        kwargs = dict(value=value, proportional_size=proportional_size,
                      use_proportional_edit=self.use_proportional_edit,
                      proportional_edit_falloff=self.proportional_edit_falloff,
                      use_proportional_connected=self.use_proportional_connected)
        if blwr_oth.check_blender_version_ge('2.80'):
            # TODO: fix constraint_orientation
            rename_map = [('constraint_orientation', 'orient_type')]
            for k, v in rename_map:
                if k in kwargs:
                    kwargs[v] = kwargs[k]
                    kwargs.pop(k)
            if 'proportional' in kwargs:
                kwargs['use_proportional_edit'] = kwargs['proportional'] == 'ENABLED'
                kwargs.pop('proportional')
        bpy.ops.transform.resize(**kwargs)

    def rotate(self, value):
        for v, axis in zip(value, 'XYZ'):
            bpy.ops.transform.rotate(value=v, orient_axis=axis, orient_type='GLOBAL')

    def set_pivot_median(self):
        bpy.context.scene.tool_settings.transform_pivot_point = 'MEDIAN_POINT'

    def set_pivot_individual(self):
        # TODO: doesn't work when running blender with startup script (--python)
        bpy.context.scene.tool_settings.transform_pivot_point = 'INDIVIDUAL_ORIGINS'

    def bevel_edges(self, value, offset_type='OFFSET'):
        if blwr_oth.check_blender_version_ge('2.90'):
            bpy.ops.mesh.bevel(offset=value, offset_pct=value, affect='EDGES', offset_type=offset_type)
        else:
            if offset_type != 'OFFSET':
                raise NotImplementedError()
            bpy.ops.mesh.bevel(offset=value, offset_pct=0, vertex_only=False)

    def make_polygon(self, n, radius=1):
        verts = []
        for i in range(n):
            ang = i * (2 * math.pi / n)
            verts.append(self.add_vert([math.cos(ang), math.sin(ang), 0]))
            self.bm.verts[verts[-1]].co *= radius

        verts = [self.bm.verts[v] for v in verts]
        for v1, v2 in zip(verts, verts[1:] + [verts[0]]):
            assert v1 != v2
            self.bm.edges.new([v1, v2])
        self.bm.faces.new(verts)
        self.update()
        return len(self.bm.faces) - 1

    def insert_faces(self, thickness, depth=0, individual=False):
        bpy.ops.mesh.inset(thickness=thickness, depth=depth, use_individual=individual)

    def smooth_vertices(self, factor, repeat=1):
        bpy.ops.mesh.vertices_smooth(factor=factor, repeat=repeat)

    def select_similar_normal(self, threshold=0.01):
        bpy.ops.mesh.select_similar(type='NORMAL', threshold=threshold)
        self.update()

    def set_selection_mode(self, vert, edge, face):
        bpy.context.tool_settings.mesh_select_mode = (vert, edge, face)
        blwr_oth.scene_update()

    def get_selection_mode(self):
        return tuple(bpy.context.tool_settings.mesh_select_mode)

    def intersect(self):
        bpy.ops.mesh.intersect()
        self.update()

    def select_more(self):
        bpy.ops.mesh.select_more()

    def select_less(self):
        bpy.ops.mesh.select_less()

    def select_lambda(self, fun, elements='VERT'):
        if elements == 'VERT':
            for v in self.bm.verts:
                if fun(v):
                    v.select = True
        elif elements == 'FACE':
            for f in self.bm.faces:
                if fun(f):
                    f.select = True
        elif elements == 'EDGE':
            for e in self.bm.edges:
                if fun(e):
                    e.select = True
        else:
            raise NotImplementedError()

    def remove_doubles(self, thr=None):
        with blwr_oth.no_logging():
            if thr is None:
                bpy.ops.mesh.remove_doubles()
            else:
                bpy.ops.mesh.remove_doubles(threshold=thr)
            self.update()

    def remove_internal(self):
        self.deselect_all()
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type='FACE')
        self.update()

    def edge_face_add(self):
        bpy.ops.mesh.edge_face_add()
        self.update()

    def make_box(self, dims):
        verts = []
        for z, y, x in product([0, dims[2]], [0, dims[1]], [0, dims[0]]):
            verts.append(self.bm.verts.new((x, y, z)))

        faces = [self.bm.faces.new((verts[1], verts[0], verts[2], verts[3])),
                 self.bm.faces.new((verts[4], verts[5], verts[7], verts[6])),
                 self.bm.faces.new((verts[2], verts[0], verts[4], verts[6])),
                 self.bm.faces.new((verts[1], verts[3], verts[7], verts[5])),
                 self.bm.faces.new((verts[4], verts[0], verts[1], verts[5])),
                 self.bm.faces.new((verts[2], verts[6], verts[7], verts[3]))]

        for v in verts:
            v.select = True
        for f in faces:
            f.select = True

        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()

        return self.get_selected_verts()

    def shrink_fatten(self, value):
        bpy.ops.transform.shrink_fatten(
            value=value,
            use_even_offset=False,
            mirror=True,
            use_proportional_edit=False,
            proportional_edit_falloff='SMOOTH',
            proportional_size=1,
            use_proportional_connected=False,
            use_proportional_projected=False
        )

    def convex_hull(self):
        bpy.ops.mesh.convex_hull()
        self.update()

    def to_tris(self):
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        self.update()

    def get_vgroups(self):
        return [g.name for g in bpy.context.object.vertex_groups]

    def select_vgroup(self, name, threshold=None, select=True):
        obj = blwr_obj.get_single_selected()
        vgroups = bpy.context.object.vertex_groups
        vgrp = vgroups[name]
        vgroups.active_index = vgrp.index

        if threshold is None:
            if select:
                bpy.ops.object.vertex_group_select()
            else:
                bpy.ops.object.vertex_group_deselect()
        else:
            for poly in obj.data.polygons:
                for loop in poly.loop_indices:
                    vertindex = obj.data.loops[loop].vertex_index
                    try:
                        weight = vgrp.weight(vertindex)
                    except RuntimeError:
                        # vertex not in any group
                        continue
                    if weight > threshold:
                        obj.data.vertices[vertindex].select = select
                        self.bm.verts[vertindex].select = select
        blwr_oth.scene_update()

    def remove_vgroup(self, name):
        self.obj.vertex_groups.remove(self.obj.vertex_groups[name])

    def deselect_vgroup(self, name):
        self.select_vgroup(name, select=False)

    def add_vertex_group_from_selection(self, name, replace=False):
        if replace and name in self.obj.vertex_groups:
            self.remove_vgroup(name)
        vgroups = bpy.context.object.vertex_groups
        obj = bpy.context.object
        if name not in obj.vertex_groups:
            obj.vertex_groups.new(name=name)
        vgroups.active_index = vgroups[name].index
        bpy.ops.object.vertex_group_assign()

    def select_boundary_loop(self):
        bpy.ops.mesh.region_to_loop()

    def hide_selection(self):
        bpy.ops.mesh.hide(unselected=False)

    def reveal_hidden(self):
        bpy.ops.mesh.reveal()

    def get_connected_components(self):
        return get_connected_components(self.bm, self.get_selected_verts())

    def make_normals_consistent(self, inside=False):
        bpy.ops.mesh.normals_make_consistent(inside=inside)

    def invert_selection(self):
        bpy.ops.mesh.select_all(action='INVERT')

    def enable_proportional(self, enable=True, falloff='LINEAR', connected=False):
        self.use_proportional_edit = enable
        self.proportional_edit_falloff = falloff
        self.use_proportional_connected = connected

    def get_bounds(self, vertices):
        verts = np.array([list(self.bm.verts[i].co) for i in vertices])
        return [(np.min(verts[:, dim]), np.max(verts[:, dim])) for dim in (0, 1, 2)]

    def get_center(self, vertices):
        return np.mean([list(self.bm.verts[i].co) for i in vertices], axis=0)

    def select_sharp_edges(self, angle):
        bpy.ops.mesh.edges_select_sharp(sharpness=angle)

    def assign_material_slot(self, slot_index):
        for _ in range(max(slot_index + 1 - len(self.obj.material_slots), 0)):
            bpy.ops.object.material_slot_add()
        bpy.context.object.active_material_index = slot_index
        bpy.ops.object.material_slot_assign()

    def shade_smooth(self):
        bpy.ops.mesh.faces_shade_smooth()

    def shade_flat(self):
        bpy.ops.mesh.faces_shade_flat()

    def skin_resize(self, value):
        bpy.ops.transform.skin_resize(value=value,
                                      orient_type='GLOBAL',
                                      orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                      orient_matrix_type='GLOBAL',
                                      mirror=True)

    def separate_selection(self, duplicate=True):
        if duplicate:
            bpy.ops.mesh.duplicate_move()
        old_objs = blwr_obj.get_selected()
        bpy.ops.mesh.separate(type='SELECTED')
        selected = blwr_obj.get_selected()
        selected = [s for s in selected if s not in old_objs]
        assert len(selected) == 1
        return selected[0]

    def fill_holes(self, sides):
        bpy.ops.mesh.fill_holes(sides=sides)

    def get_bvh(self):
        from mathutils.bvhtree import BVHTree
        if self.bvh is None:
            self.bvh = BVHTree.FromBMesh(self.bm, epsilon=0.0001)
        return self.bvh

    def looptools_space(self):
        bpy.ops.mesh.looptools_space(influence=100,
                                     input='selected',
                                     interpolation='cubic',
                                     lock_x=False, lock_y=False, lock_z=False)

    def subdivide(self, number_cuts=1):
        if number_cuts == 0:
            return
        bpy.ops.mesh.subdivide(number_cuts=number_cuts)
        self.update()

    def bridge_edge_loops(self, cuts=10, profile_factor=0, smoothness=1):
        bpy.ops.mesh.bridge_edge_loops(number_cuts=cuts,
                                       profile_shape_factor=profile_factor,
                                       smoothness=smoothness)

    def looptools_bridge_edge_loops(self, cuts=10):
        bpy.ops.mesh.looptools_bridge(cubic_strength=1,
                                      interpolation='cubic',
                                      loft=False,
                                      loft_loop=False,
                                      min_width=0,
                                      mode='shortest',
                                      remove_faces=True,
                                      reverse=False,
                                      segments=cuts,
                                      twist=0)

    def looptools_circle(self):
        bpy.ops.mesh.looptools_circle(custom_radius=False,
                                      fit='best',
                                      flatten=True,
                                      influence=100,
                                      lock_x=False,
                                      lock_y=False,
                                      lock_z=False,
                                      radius=1,
                                      regular=True)

    def looptools_flatten(self, plane='best_fit'):
        bpy.ops.mesh.looptools_flatten(influence=100,
                                       lock_x=False,
                                       lock_y=False,
                                       lock_z=False,
                                       plane=plane,
                                       restriction='none')

    def select_edge_loops(self):
        bpy.ops.mesh.loop_multi_select(ring=False)

    def select_edge_rings(self):
        bpy.ops.mesh.loop_multi_select(ring=True)

    def dissolve_vert(self):
        bpy.ops.mesh.dissolve_verts()
        self.update()

    def dissolve_edges(self):
        bpy.ops.mesh.dissolve_edges()
        self.update()

    def dissolve_faces(self):
        bpy.ops.mesh.dissolve_faces()
        self.update()

    def delete_loose(self):
        bpy.ops.mesh.delete_loose()
        self.update()

    def merge(self, type='COLLAPSE'):
        bpy.ops.mesh.merge(type=type)
        self.update()

    def mark_seam(self, clear=False):
        bpy.ops.mesh.mark_seam(clear=clear)

    def select_seam(self, select=True):
        for e in self.bm.edges:
            if e.seam:
                e.select = select

    def select_linked(self):
        bpy.ops.mesh.select_linked(delimit={'SEAM'})

    def edge_collapse(self):
        bpy.ops.mesh.edge_collapse()
        self.update()

    def shortest_path_select(self):
        bpy.ops.mesh.shortest_path_select()

    def update_edit_mesh(self):
        bmesh.update_edit_mesh(self.obj.data)

    def select_flush(self, select=False):
        self.bm.select_flush(select)

    def check_vgroup_overlap(self, name1, name2):
        self.deselect_all()
        self.select_vgroup(name1)
        count1 = len(self.get_selected_verts())
        self.deselect_vgroup(name2)
        return count1 != len(self.get_selected_verts())

    def symmetrize(self):
        bpy.ops.mesh.symmetrize()
        self.update()

    def select_mirror(self, extend=False):
        with blwr_oth.no_logging():
            bpy.ops.mesh.select_mirror(extend=extend)

    def get_selection_center(self):
        return np.mean(np.array([v.co for v in self.bm.verts if v.select]), axis=0)
