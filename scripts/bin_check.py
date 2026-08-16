from pathlib import Path
import numpy as np
from mayavi import mlab

# 脚本所在目录；项目根目录由它的上一级目录确定。
script_dir = Path(__file__).resolve().parent

# 项目根目录。
project_dir = script_dir.parent

# 要检查的 LiDAR 二进制点云文件。
lidar_path = (
    project_dir
    / "dataset"
    / "scene_20260814_205000"
    / "lidar"
    / "000008.bin"
)

print("读取文件：", lidar_path)
print("文件是否存在：", lidar_path.exists())

pointcloud = np.fromfile(
    lidar_path,
    dtype=np.float32
).reshape(-1, 4)

print("点云 shape:", pointcloud.shape)

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]
r = pointcloud[:, 3]

d = np.sqrt(x ** 2 + y ** 2)

fig = mlab.figure(
    bgcolor=(0, 0, 0),
    size=(640, 500)
)

mlab.points3d(
    x, y, z,
    z,
    mode="point",
    colormap="spectral",
    figure=fig
)

mlab.show()
