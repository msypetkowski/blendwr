import bpy
import math
import numpy as np
import random

from .. import edit as blwr_edit
from .. import object as blwr_obj
from .. import other as blwr_oth


def _lerp(factor, v1, v2):
    return v1 * (1 - factor) + v2 * factor


def _randomize_bevel_obj(rand):
    obj = blwr_obj.create_empty_mesh('GrassBevelObj')

    thickness = max(0.001, rand.gauss(0.015, 0.003))
    width = max(0.01, rand.gauss(0.04, 0.02))

    with blwr_edit.EditMesh(obj) as editor:
        editor.set_selection_mode(True, True, False)
        editor.add_vert((width, 0, 0))
        editor.select_all()
        editor.extrude((-width, thickness, 0))
        editor.extrude((-width, -thickness, 0))

    blwr_obj.focus(obj)
    return blwr_obj.convert_to_curve(obj)


def _randomize_single_spline(rand, radius):
    curve_obj = blwr_obj.create_empty_curve()
    blwr_obj.clear_transform(curve_obj)

    x1, y1, z1 = (rand.gauss(0.5, 0.2), 0, rand.gauss(0.6, 0.3))
    x2, y2, z2 = (rand.gauss(x1 * 1.5, 0.2), rand.gauss(y1, 0.2), rand.gauss(z1, 0.5))

    bp1 = blwr_edit.BezierPoint(co=(0, 0, 0), handle_right=(x1, y1, z1), radius=radius)
    bp2 = blwr_edit.BezierPoint(co=(x2, y2, z2), handle_left=(x1, y1, z1), radius=0)

    with blwr_edit.EditCurve(curve_obj) as editor:
        editor.add_bezier_spline([bp1, bp2])

    curve_obj.data.resolution_u = 6
    bevel_obj = _randomize_bevel_obj(rand)
    curve_obj.data.bevel_mode = 'OBJECT'
    curve_obj.data.bevel_object = bevel_obj
    blwr_obj.convert_to_mesh(curve_obj)
    blwr_obj.remove(bevel_obj)
    return curve_obj


def _randomize_grass_splines(spec_rand, instance_rand, is_dense_grass):
    base_radius_mean = _lerp(is_dense_grass, spec_rand.uniform(0.8, 7), spec_rand.uniform(0.5, 2))
    base_radius_std = spec_rand.uniform(base_radius_mean * 0.05, base_radius_mean * 0.4)
    splines = []
    for _ in range(spec_rand.choice([1, 2, 3, 4, 10, 10, 30])):
        splines.append(_randomize_single_spline(instance_rand, instance_rand.gauss(base_radius_mean, base_radius_std)))
    return splines


def generate_single_asset(spec_rand, instance_rand):
    is_dense_grass = spec_rand.betavariate(0.3, 0.3)
    grass_splines = _randomize_grass_splines(spec_rand, instance_rand, is_dense_grass)

    placed_curves = []
    n = round(_lerp(is_dense_grass, spec_rand.randint(5, 10), spec_rand.randint(40, 100)))
    base_radius = _lerp(is_dense_grass, spec_rand.uniform(0.0, 0.2), spec_rand.uniform(0.1, 0.3))
    base_std = spec_rand.uniform(0.03, 0.2)
    location_std = max(_lerp(is_dense_grass, spec_rand.uniform(0.4, 0.5), spec_rand.uniform(4.5, 1.0)), 0)
    for angle in np.linspace(0, 2 * math.pi, n + 1)[:-1]:
        radius = instance_rand.gauss(base_radius, base_std)
        location = np.array([math.cos(angle), math.sin(angle), 0]) * radius
        location[0] = instance_rand.gauss(location[0], base_radius * location_std)
        location[1] = instance_rand.gauss(location[1], base_radius * location_std)
        rotation = [instance_rand.gauss(0, 0.1), instance_rand.gauss(0, 0.1), angle]
        transform = blwr_oth.loc_rot_scale_to_matrix(location, rotation, [1, 1, 1])
        placed_curves.append(blwr_obj.duplicate(spec_rand.choice(grass_splines)))
        placed_curves[-1].matrix_world = transform
    blwr_obj.remove(grass_splines)
    ret = blwr_obj.join(*placed_curves)
    blwr_obj.apply_transform(ret)
    return ret


def generate_single_species(seed, spec_id, n_instances, scale=(1.0, 1.0, 0.5)):
    grasses = []
    for instance_id in range(n_instances):
        rand = random.Random(seed)
        spec_rand = random.Random(rand.randint(0, 2 ** 32) + spec_id)
        instance_rand = random.Random(spec_rand.randint(0, 2 ** 32) + instance_id)

        grass_obj = generate_single_asset(spec_rand, instance_rand)
        blwr_obj.resize(grass_obj, scale)
        blwr_obj.apply_transform(grass_obj)
        grass_obj.name = f'Grass_{spec_id}_{instance_id}'
        grasses.append(grass_obj)
    return grasses


def grass(seed, n_species=8, n_instances=5):
    species = []
    for spec_id in range(n_species):
        grasses = generate_single_species(seed, spec_id, n_instances)
        species.append(grasses)
        for inst_id, grass in enumerate(grasses):
            grass.location[0] = inst_id * 3
            grass.location[1] = spec_id * 3
    return species
