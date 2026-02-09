import open3d
import numpy

mesh = open3d.io.read_triangle_mesh("MODEL_DIRECTORY_HERE")
vertices = numpy.asarray(mesh.vertices)

z_threshold = numpy.percentile(vertices[:, 2], 95)
top_head = numpy.array([vert for vert in vertices if vert[2] > z_threshold])

top_head_mesh = open3d.geometry.PointCloud()
top_head_mesh.points = open3d.utility.Vector3dVector(top_head)

x_midline = (numpy.max(top_head[:, 0]) + numpy.min(top_head[:, 0]))/2
y_midline = (numpy.max(top_head[:, 1]) + numpy.min(top_head[:, 1]))/2

close_x = numpy.array([vert for vert in top_head if (x_midline - 3) < vert[0] < (x_midline + 3)])
close_x = close_x[numpy.argsort(close_x[:, 1])]

close_y = numpy.array([vert for vert in top_head if (y_midline - 3) < vert[1] < (y_midline + 3)])
close_y = close_y[numpy.argsort(close_y[:, 0])]

x_top = close_x[0]
y_top = close_y[0]
x_bottom = close_x[-1]
y_bottom = close_y[-1]

pcd = mesh.sample_points_uniformly(number_of_points=1024)
sampled_points = numpy.asarray(pcd.points)

labeled_points = numpy.array([x_top, x_bottom, y_top, y_bottom])
all_points = numpy.vstack([sampled_points, labeled_points])

colors = numpy.ones((1028, 3)) * 0.5
colors[1024] = [1, 0, 0]
colors[1025] = [0, 1, 0]
colors[1026] = [0, 0, 1]
colors[1027] = [1, 1, 0]

final_pcd = open3d.geometry.PointCloud()
final_pcd.points = open3d.utility.Vector3dVector(all_points)
final_pcd.colors = open3d.utility.Vector3dVector(colors)

vis = open3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(final_pcd)

render_option = vis.get_render_option()
render_option.point_color_option = open3d.visualization.PointColorOption.Color

vis.run()