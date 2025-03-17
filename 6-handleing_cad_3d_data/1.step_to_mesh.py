import cadquery as cq
import open3d as o3d
import numpy as np
import os
import tempfile

# import a step file
step_file = "data/turbine_wing.stp"
script_dir = os.path.dirname(os.path.realpath(__file__))
step_file = os.path.join(script_dir, step_file)
stp = cq.importers.importStep(step_file)

# 임시 파일 생성 (Windows에서는 delete=False를 사용하고, 직접 삭제할 수 있음)
with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
    temp_filename = tmp.name
    cq.exporters.export(stp, temp_filename)

# convert to STL
stl_path = os.path.join(script_dir, "data/turbine_wing.stl")
cq.exporters.export(stp, stl_path)

print("Imported and exported the step file")

# Load the STL file using Open3D
mesh = o3d.io.read_triangle_mesh(stl_path)  # Load the mesh
mesh.compute_vertex_normals()  # Compute the normals

# ✅ Edge 검출
edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

# ✅ Edge를 빨간색으로 칠하기
edges.paint_uniform_color([1, 0, 0])  # 빨간색 엣지

# ✅ Mesh를 Point Cloud로 변환
pcd = mesh.sample_points_uniformly(number_of_points=10000)

# ✅ RANSAC을 이용한 평면 검출
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

# ✅ 평면과 나머지 점군 분리
plane_cloud = pcd.select_by_index(inliers)
remaining_cloud = pcd.select_by_index(inliers, invert=True)

# ✅ 시각화 (빨간색: 평면, 초록색: 나머지 점)
o3d.visualization.draw_geometries([plane_cloud.paint_uniform_color([1, 0, 0]),
                                   remaining_cloud.paint_uniform_color([0, 1, 0])])

# # ✅ Open3D 시각화
# o3d.visualization.draw_geometries([mesh, edges])
