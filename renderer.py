# filepath: /home/bjoern/Documents/Python/vtube/renderer.py
import bpy
import os
import time

def render_scene():
    if os.path.exists('/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render_done'):
        return
    bpy.context.scene.render.filepath = '/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render.jpg'
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.ops.render.render(write_still=True)
    with open('/home/bee_tron/admin:///media/bee_tron/STEAM/tmp/render_done', 'w') as f:
        f.write('done')

def render_timer():
    render_scene()
    return 0.01  # Check every 1 second

if __name__ == "__main__":
    bpy.app.timers.register(render_timer)