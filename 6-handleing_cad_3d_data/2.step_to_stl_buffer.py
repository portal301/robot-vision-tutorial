import cadquery as cq
import open3d as o3d
import numpy as np
import os
import io
import trimesh

# import a step file
step_file = "data/turbine_wing.stp"
script_dir = os.path.dirname(os.path.realpath(__file__))
step_file = os.path.join(script_dir, step_file)
stp = cq.importers.importStep(step_file)
print("success: import the step file")

# convert to STL
stl_buffer = io.BytesIO()
cq.exporters.export(stp, cq.exporters.ExportTypes.STL_ASCII)


print("success: export to the step file")

# ✅ Trimesh를 사용하여 메모리에서 STL 읽기
# mesh_trimesh = trimesh.load_mesh(stl_buffer, file_type="stl")
mesh_trimesh = trimesh.load_mesh(stl_buffer, file_type="stl", maintain_order=True, force='mesh')

# ✅ Face가 없을 경우, Trimesh의 `fix_normals()`로 복구 시도
if len(mesh_trimesh.faces) == 0:
    print("⚠ Warning: STL 파일에 삼각형(Face)이 없음! 자동 복구를 시도합니다.")
    mesh_trimesh.fix_normals()  # 삼각형 면이 사라졌을 경우 복구 시도


# ✅ Trimesh 데이터를 Open3D TriangleMesh로 변환
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
mesh.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
mesh.compute_vertex_normals()  # 법선 계산
mesh.compute_triangle_normals()  # 삼각형 법선 계산


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
