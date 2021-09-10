import bpy
import colorsys
import math
import numpy as np
import random
from itertools import chain

from .. import edit as blwr_edit
from .. import examples as blwr_examples
from .. import object as blwr_obj
from .. import object as blwr_obj
from .. import other as blwr_oth


def create_scatter_ps(obj, target_mesh, seed, count, density_vgroup):
    blwr_obj.focus(target_mesh)
    bpy.ops.object.particle_system_add()
    ps = target_mesh.particle_systems[-1]
    ps.seed = seed
    settings = ps.settings
    settings.type = 'HAIR'
    settings.use_advanced_hair = True
    settings.render_type = 'OBJECT'
    settings.particle_size = 0.4
    settings.size_random = 0.5
    settings.instance_object = obj
    settings.use_rotations = True
    settings.use_rotation_instance = True
    settings.rotation_mode = 'NOR'
    settings.rotation_factor_random = 0.1
    settings.phase_factor_random = 2
    settings.count = count
    ps.vertex_group_density = density_vgroup
    return ps


def assign_solid_material(obj, color, roughness):
    blwr_obj.focus(obj)
    m = bpy.data.materials.new(f'SolidMaterial_{color}_{roughness}')
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color
    m.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness
    blwr_obj.assign_material(obj, m)


def make_initial_terrain(seed):
    terrain = blwr_examples.simple_terrain(seed, 7)
    blwr_obj.decimate_collapse(terrain, 0.3)
    with blwr_edit.EditMesh(terrain) as editor:
        editor.select_all()
        editor.select_boundary_loop()
        editor.extrude((0, 0, -1))
        for v in editor.get_selected_verts(indices=False):
            v.co[2] = 0
        editor.select_more()
        editor.add_vertex_group_from_selection('boundary')
    assign_solid_material(terrain, (0.2, 0.1, 0.1, 1), 0.8)
    blwr_obj.resize(terrain, [170] * 3)
    blwr_obj.apply_transform(terrain)
    return terrain


def color_grasses(seed, grasses):
    rand = random.Random(seed)
    for spec_id, species_instances in enumerate(grasses):
        species_base_hsv = np.array([rand.uniform(0.0, 0.5), rand.uniform(0.2, 0.8), rand.uniform(0.1, 0.4)])
        species_base_roughness = rand.uniform(0.2, 0.6)
        for inst_id, grass in enumerate(species_instances):
            hsv = [v * rand.uniform(0.8, 1.2) for v in species_base_hsv]
            roughness = species_base_roughness * rand.uniform(0.8, 1.2)
            assign_solid_material(grass, (*colorsys.hsv_to_rgb(*hsv), 1), roughness)


def scatter_grass(seed, grasses, terrain):
    rand = random.Random(seed)
    for spec_id, species_instances in enumerate(grasses):

        preferred_normal = np.array([*[2 * rand.random() - 1] * 2, 3 * rand.random()])
        preferred_normal /= np.linalg.norm(preferred_normal)
        with blwr_edit.EditMesh(terrain) as editor:
            editor.deselect_all()
            for v in editor.bm.verts:
                if np.linalg.norm(np.array(v.normal) - preferred_normal) < 0.3:
                    v.select = True
            vgroup_name = f'species_{spec_id}'
            editor.add_vertex_group_from_selection(vgroup_name)
            editor.smooth_vgroup(vgroup_name, factor=.5, repeat=rand.randint(15, 30), expand=0.1)
            editor.deselect_all()
            editor.select_vgroup('boundary')
            editor.remove_selection_from_vgroup(vgroup_name)

        for inst_id, grass_inst in enumerate(species_instances):
            import math
            grass_inst.rotation_euler[1] = math.pi / 2
            create_scatter_ps(grass_inst, terrain, seed=inst_id, count=1000, density_vgroup=vgroup_name)


def detail_terrain(seed, terrain):
    rand = random.Random(seed)
    with blwr_edit.EditMesh(terrain) as editor:
        editor.set_selection_mode(False, True, False)
        editor.deselect_all()
        editor.select_sharp_edges(math.pi / 6)
        editor.crease_selected_edges(1.0)
    for detail_level in range(4):
        with blwr_edit.EditMesh(terrain) as editor:
            editor.deselect_all()
            editor.select(rand.sample(list(editor.bm.verts), k=len(editor.bm.verts) // 2))
            scale = 1.5 * 2 ** (-detail_level)
            editor.translate([rand.uniform(-scale, scale) for _ in range(3)])
        if detail_level < 3:
            blwr_obj.subdivide(terrain, 1)


def prepare_to_render():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.cycles.samples = 1024
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.threads_mode = 'AUTO'
    bpy.context.scene.cycles.volume_step_rate = 1.0
    bpy.context.scene.cycles.volume_bounces = 0


def setup_scene_and_camera(terrain):
    bpy.ops.object.camera_add(enter_editmode=False, location=(0, 0, 0),
                              rotation=(math.pi / 4, 0, - math.pi / 4), scale=(1, 1, 1))
    camera = blwr_obj.get_single_selected()
    blwr_obj.focus(camera)
    blwr_obj.deselect_all()
    blwr_obj.select(terrain)
    bpy.context.scene.camera = camera
    bpy.ops.view3d.camera_to_view_selected()
    blwr_obj.deselect_all()
    return camera


def setup_lighting(seed, camera):
    rand = random.Random(seed)
    bpy.context.scene.render.film_transparent = False

    world = bpy.data.worlds.new(name='ExampleWorld')
    world.use_nodes = True
    tree = world.node_tree

    sky_node = tree.nodes.new(type="ShaderNodeTexSky")
    bg_node = tree.nodes['Background']
    tree.links.new(sky_node.outputs[0], bg_node.inputs[0])

    bg_node.inputs['Strength'].default_value = 0.3

    sky_node.sun_elevation = (15 / 360) * 2 * math.pi * rand.uniform(0.7, 2.1)
    sky_node.sun_rotation = -camera.rotation_euler[2] + math.pi * rand.choice([
        rand.uniform(0.04, 0.6), -rand.uniform(0.04, 0.6)])
    sky_node.sun_intensity = rand.gauss(0.32, 0.04)
    sky_node.ozone_density = rand.gauss(1, 0.1)
    sky_node.altitude = rand.gauss(10, 1)

    bpy.context.scene.world = world


def make_fog(terrain):
    fog_box = blwr_obj.create_empty_mesh('fog')
    with blwr_edit.EditMesh(fog_box) as editor:
        editor.make_box([2] * 3)
        editor.select_all()
        editor.translate([-.5] * 3)

    blwr_obj.resize(fog_box, [terrain.dimensions[0], terrain.dimensions[1],
                              terrain.dimensions[2] * 1])

    m = bpy.data.materials.new(f'Fog material')
    m.use_nodes = True
    tree = m.node_tree
    tree.nodes.remove(tree.nodes['Principled BSDF'])
    output_node = tree.nodes['Material Output']
    volume_node = tree.nodes.new(type='ShaderNodeVolumePrincipled')
    volume_node.inputs['Density'].default_value = 0.0004
    tree.links.new(volume_node.outputs[0], output_node.inputs['Volume'])
    blwr_obj.assign_material(fog_box, m)

    return fog_box


def terrain_scatter(seed):
    """ Creates ready to render scene """
    terrain = make_initial_terrain(seed)
    grasses = blwr_examples.grass(seed, n_species=4, n_instances=2)
    color_grasses(seed, grasses)
    scatter_grass(seed, grasses, terrain)
    detail_terrain(seed, terrain)
    prepare_to_render()
    camera = setup_scene_and_camera(terrain)
    setup_lighting(seed, camera)
    make_fog(terrain)
    blwr_obj.hide(list(chain(*grasses)), hide_render=True)
