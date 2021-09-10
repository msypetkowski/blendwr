from .. import edit as blwr_edit
from .. import object as blwr_obj
from .. import other as blwr_oth


def displace_with_image(resolution=10, image_path='blendwr/images/example_hm.png'):
    obj = blwr_obj.create_empty_mesh()
    with blwr_edit.EditMesh(obj) as editor:
        editor.bm.faces.new([editor.bm.verts.new(co + [0]) for co in [[0, 0], [0, 1], [1, 1], [1, 0]]])
        editor.select_all()
        editor.uv_unwrap()
    blwr_obj.subdivide(obj, resolution, subdivision_type='SIMPLE')
    img = blwr_oth.make_blender_image(blwr_oth.read_image(image_path), img_save_path='/tmp/hm.png')
    blwr_obj.displace(obj, img, strength=0.02)
    blwr_obj.shade_smooth(obj)
    return obj
