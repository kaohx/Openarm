import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
from typing import List, Tuple, Optional, Dict, Any


def calcu_dis(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两点之间的欧几里得距离"""
    return np.linalg.norm(p1 - p2)


class RRTTree:
    """RRT树数据结构"""

    def __init__(self):
        self.nodes = []  # 节点坐标列表
        self.parents = []  # 父节点索引列表

    def add_node(self, node: np.ndarray, parent_idx: int):
        """向树中添加节点"""
        self.nodes.append(node)
        self.parents.append(parent_idx)

    def get_node(self, idx: int) -> np.ndarray:
        """获取指定索引的节点"""
        return self.nodes[idx]

    def size(self) -> int:
        """返回树中节点的数量"""
        return len(self.nodes)


def sample_point(axis_start: np.ndarray, axis_lwh: np.ndarray,
                 goal_point: np.ndarray, goal_bias: float = 0.1) -> np.ndarray:
    """
    在空间中随机采样点

    参数:
        axis_start: 空间起始点 [x_min, y_min, z_min]
        axis_lwh: 空间尺寸 [length, width, height]
        goal_point: 目标点
        goal_bias: 采样到目标点的概率

    返回:
        采样点的坐标
    """
    # 有一定概率直接返回目标点（偏向性采样）
    if random.random() < goal_bias:
        return goal_point.copy()

    # 在空间范围内随机采样
    x = axis_start[0] + random.random() * axis_lwh[0]
    y = axis_start[1] + random.random() * axis_lwh[1]
    z = axis_start[2] + random.random() * axis_lwh[2]

    return np.array([x, y, z])


def find_near_point(rand_point: np.ndarray, tree: RRTTree) -> Tuple[np.ndarray, int]:
    """
    在树上寻找距离随机点最近的节点

    参数:
        rand_point: 随机点坐标
        tree: RRT树

    返回:
        (最近节点坐标, 最近节点索引)
    """
    min_dist = float('inf')
    nearest_idx = 0

    for i, node in enumerate(tree.nodes):
        dist = calcu_dis(rand_point, node)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    return tree.get_node(nearest_idx), nearest_idx


def expand_point(near_point: np.ndarray, rand_point: np.ndarray, step: float) -> np.ndarray:
    """
    从最近点向随机点方向扩展指定步长（严格限制步长）

    参数:
        near_point: 最近点坐标
        rand_point: 随机点坐标
        step: 扩展步长

    返回:
        新扩展点的坐标
    """
    # 计算方向向量
    direction = rand_point - near_point
    dist = calcu_dis(near_point, rand_point)

    # 严格限制步长，即使距离小于步长也按比例缩放（确保步长不超限）
    if dist < 1e-6:  # 避免除以0
        return near_point.copy()

    # 归一化并扩展（强制步长不超过设定值）
    direction = direction / dist
    new_point = near_point + direction * min(step, dist)

    return new_point


def plot_cube3d(ax, center, size, color='gray', alpha=0.5):
    """
    Draw cube 3 dimension
    :param center:
    :param size:
    :param color:
    :param alpha:
    """
    x = center[0]
    y = center[1]
    z = center[2]
    l = size[0]
    w = size[1]
    h = size[2]

    vertices = [
        [x - l / 2, y - w / 2, z - h / 2],
        [x + l / 2, y - w / 2, z - h / 2],
        [x + l / 2, y + w / 2, z - h / 2],
        [x - l / 2, y + w / 2, z - h / 2],
        [x - l / 2, y - w / 2, z + h / 2],
        [x + l / 2, y - w / 2, z + h / 2],
        [x + l / 2, y + w / 2, z + h / 2],
        [x - l / 2, y + w / 2, z + h / 2],
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
    ]

    # 绘制立方体
    cube = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(cube)
    return cube


def plot_cube(ax, center, size, color='gray', alpha=0.5):
    """绘制立方体"""
    # 计算立方体的8个顶点
    x = center[0]
    y = center[1]
    z = center[2]
    s = size / 2

    vertices = [
        [x - s, y - s, z - s],
        [x + s, y - s, z - s],
        [x + s, y + s, z - s],
        [x - s, y + s, z - s],
        [x - s, y - s, z + s],
        [x + s, y - s, z + s],
        [x + s, y + s, z + s],
        [x - s, y + s, z + s]
    ]

    # 定义立方体的6个面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
    ]

    # 绘制立方体
    cube = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(cube)
    return cube


def plot_cylinder(ax, center, radius, height, color='blue', alpha=0.5, num_points=50):
    """绘制圆柱体"""
    z_center = center[2]
    x_center = center[0]
    y_center = center[1]

    # 生成圆柱体的底面和顶面
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_bottom = x_center + radius * np.cos(theta)
    y_bottom = y_center + radius * np.sin(theta)
    z_bottom = np.full_like(x_bottom, z_center - height / 2)

    x_top = x_center + radius * np.cos(theta)
    y_top = y_center + radius * np.sin(theta)
    z_top = np.full_like(x_top, z_center + height / 2)

    # 绘制底面
    ax.plot_trisurf(x_bottom, y_bottom, z_bottom, color=color, alpha=alpha * 0.8)

    # 绘制顶面
    ax.plot_trisurf(x_top, y_top, z_top, color=color, alpha=alpha * 0.8)

    # 绘制侧面
    for i in range(num_points - 1):
        x_side = [x_bottom[i], x_bottom[i + 1], x_top[i + 1], x_top[i]]
        y_side = [y_bottom[i], y_bottom[i + 1], y_top[i + 1], y_top[i]]
        z_side = [z_bottom[i], z_bottom[i + 1], z_top[i + 1], z_top[i]]

        verts = [list(zip(x_side, y_side, z_side))]
        side = Poly3DCollection(verts, color=color, alpha=alpha * 0.6)
        ax.add_collection3d(side)

    return None


def plot_sphere(ax, center, radius, color='red', alpha=0.5, num_points=50):
    """绘制球体"""
    # 生成球面的点
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    # 绘制球体
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)
    return None


def is_point_in_cube(point, cube_center, cube_size):
    """检查点是否在立方体内"""
    half_size = cube_size / 2
    return (abs(point[0] - cube_center[0]) <= half_size and
            abs(point[1] - cube_center[1]) <= half_size and
            abs(point[2] - cube_center[2]) <= half_size)


def is_point_in_cube3d(point, cube_center, cube_size):
    """Check point is in cube"""
    half_size = [i / 2 for i in cube_size]
    return (abs(point[0] - cube_center[0]) <= half_size[0] and
            abs(point[1] - cube_center[1]) <= half_size[1] and
            abs(point[2] - cube_center[2]) <= half_size[2])


def is_cube_collision(cube_info: Dict[str, Any], near_point: np.ndarray,
                      new_point: np.ndarray, step: float) -> bool:
    """
    检测与立方体障碍物的碰撞（适配小步长）

    参数:
        cube_info: 立方体信息字典
        near_point: 起点
        new_point: 终点
        step: 步长（用于采样检测）

    返回:
        True: 发生碰撞, False: 无碰撞
    """
    if cube_info is None:
        return False

    # 支持多个立方体
    cubes = cube_info if isinstance(cube_info, list) else [cube_info]

    # 适配小步长的采样数计算：确保至少采样10个点，或按步长密度采样
    segment_length = calcu_dis(near_point, new_point)
    num_samples = max(10, int(segment_length / (step / 2)))  # 小步长时增加采样数
    num_samples = min(num_samples, 100)  # 限制最大采样数避免性能问题

    for t in np.linspace(0, 1, num_samples):
        point = near_point + (new_point - near_point) * t

    # 检查点是否在任何一个立方体内
    for cube in cubes:
        center = cube.get('center', np.zeros(3))
    size = cube.get('size', 1.0)
    if is_point_in_cube3d(point, center, size):
        return True

    return False


def is_point_in_cylinder(point, center, radius, height):
    """检查点是否在圆柱体内"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    radial_dist = np.sqrt(dx * dx + dy * dy)
    vertical_dist = abs(point[2] - center[2])

    return radial_dist <= radius and vertical_dist <= height / 2


def is_cylinder_collision(cylinder_info: Dict[str, Any], near_point: np.ndarray,
                          new_point: np.ndarray, step: float) -> bool:
    """
    检测与圆柱体障碍物的碰撞（适配小步长）

    参数:
        cylinder_info: 圆柱体信息字典
        near_point: 起点
        new_point: 终点
        step: 步长

    返回:
        True: 发生碰撞, False: 无碰撞
    """
    if cylinder_info is None:
        return False

    # 支持多个圆柱体
    cylinders = cylinder_info if isinstance(cylinder_info, list) else [cylinder_info]

    # 适配小步长的采样数计算
    segment_length = calcu_dis(near_point, new_point)
    num_samples = max(10, int(segment_length / (step / 2)))
    num_samples = min(num_samples, 100)

    for t in np.linspace(0, 1, num_samples):
        point = near_point + (new_point - near_point) * t

        # 检查点是否在任何一个圆柱体内
        for cylinder in cylinders:
            center = cylinder.get('center', np.zeros(3))
            radius = cylinder.get('radius', 1.0)
            height = cylinder.get('height', 2.0)
            if is_point_in_cylinder(point, center, radius, height):
                return True

    return False


def is_point_in_sphere(point, center, radius):
    """检查点是否在球体内"""
    return calcu_dis(point, center) <= radius


def is_sphere_collision(sphere_info: Dict[str, Any], near_point: np.ndarray,
                        new_point: np.ndarray, step: float) -> bool:
    """
    检测与球体障碍物的碰撞（适配小步长）

    参数:
        sphere_info: 球体信息字典
        near_point: 起点
        new_point: 终点
        step: 步长

    返回:
        True: 发生碰撞, False: 无碰撞
    """
    if sphere_info is None:
        return False

    # 支持多个球体
    spheres = sphere_info if isinstance(sphere_info, list) else [sphere_info]

    # 适配小步长的采样数计算
    segment_length = calcu_dis(near_point, new_point)
    num_samples = max(10, int(segment_length / (step / 2)))
    num_samples = min(num_samples, 100)

    for t in np.linspace(0, 1, num_samples):
        point = near_point + (new_point - near_point) * t

        # 检查点是否在任何一个球体内
        for sphere in spheres:
            center = sphere.get('center', np.zeros(3))
            radius = sphere.get('radius', 1.0)
            if is_point_in_sphere(point, center, radius):
                return True

    return False


def RRT(start_point: np.ndarray, axis_start: np.ndarray, axis_lwh: np.ndarray,
        goal_point: np.ndarray, cube_info: Optional[Dict] = None,
        cylinder_info: Optional[Dict] = None, sphere_info: Optional[Dict] = None,
        visualize: bool = True, step: float = 5.0, max_iter: int = 5000) -> Optional[np.ndarray]:
    """
    RRT路径规划算法主函数（适配小步长）

    参数:
        start_point: 起点坐标
        axis_start: 空间起始坐标
        axis_lwh: 空间尺寸 [长, 宽, 高]
        goal_point: 目标点坐标
        cube_info: 立方体障碍物信息
        cylinder_info: 圆柱体障碍物信息
        sphere_info: 球体障碍物信息
        visualize: 是否可视化
        step: 步长（严格限制每个新点的扩展距离）
        max_iter: 最大迭代次数

    返回:
        路径点数组 (N×3)，如果规划失败则返回None
    """
    # 参数初始化：根据步长调整目标阈值（确保最后一段距离也不超过步长）
    iter_max = max_iter
    thr = step * 2  # 目标点判定阈值设为步长的2倍，确保最后一步能到达

    # 初始化树
    tree = RRTTree()
    tree.add_node(start_point, -1)  # 根节点的父节点索引为-1

    # 创建可视化图形
    if visualize:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_xlim([axis_start[0], axis_start[0] + axis_lwh[0]])
        ax.set_ylim([axis_start[1], axis_start[1] + axis_lwh[1]])
        ax.set_zlim([axis_start[2], axis_start[2] + axis_lwh[2]])
        ax.set_title('RRT 3D Path Planning with Obstacles (Step=0.01)', fontsize=14)

        # 绘制起点和终点
        ax.scatter(start_point[0], start_point[1], start_point[2],
                   c='green', s=200, marker='o', label='Start Point', edgecolors='black', linewidth=2)
        ax.scatter(goal_point[0], goal_point[1], goal_point[2],
                   c='red', s=200, marker='s', label='Goal Point', edgecolors='black', linewidth=2)

        # 绘制障碍物
        obstacles_added = False

        # 绘制立方体障碍物
        if cube_info is not None:
            cubes = cube_info if isinstance(cube_info, list) else [cube_info]
            for i, cube in enumerate(cubes):
                center = cube.get('center', np.zeros(3))
                size = cube.get('size', np.zeros(3))
                plot_cube3d(ax, center, size, color='gray', alpha=0.6)
                obstacles_added = True

        # 绘制圆柱体障碍物
        if cylinder_info is not None:
            cylinders = cylinder_info if isinstance(cylinder_info, list) else [cylinder_info]
            for i, cylinder in enumerate(cylinders):
                center = cylinder.get('center', np.zeros(3))
                radius = cylinder.get('radius', 1.0)
                height = cylinder.get('height', 2.0)
                plot_cylinder(ax, center, radius, height, color='blue', alpha=0.5)
                obstacles_added = True

        # 绘制球体障碍物
        if sphere_info is not None:
            spheres = sphere_info if isinstance(sphere_info, list) else [sphere_info]
            for i, sphere in enumerate(spheres):
                center = sphere.get('center', np.zeros(3))
                radius = sphere.get('radius', 1.0)
                plot_sphere(ax, center, radius, color='red', alpha=0.5)
                obstacles_added = True

        if obstacles_added:
            ax.text2D(0.05, 0.95, "Obstacles: Gray=Cube, Blue=Cylinder, Red=Sphere",
                      transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # 主循环：调整目标偏向策略适配小步长
    path_found = False
    for iter_count in range(iter_max):
        # 动态调整目标偏向概率（小步长时提高偏向概率加快收敛）
        goal_bias = 0.2 if iter_count < iter_max // 2 else 0.4

        # 随机采样
        rand_coor = sample_point(axis_start, axis_lwh, goal_point, goal_bias)

        # 寻找最近点
        near_coor, pre_index = find_near_point(rand_coor, tree)

        # 扩展新点（严格限制步长）
        new_coor = expand_point(near_coor, rand_coor, step)

        # 碰撞检测
        cube_flag = is_cube_collision(cube_info, near_coor, new_coor, step)
        cylinder_flag = is_cylinder_collision(cylinder_info, near_coor, new_coor, step)
        sphere_flag = is_sphere_collision(sphere_info, near_coor, new_coor, step)

        if cube_flag or cylinder_flag or sphere_flag:
            continue

        # 将新点加入树中
        tree.add_node(new_coor, pre_index)

        # 可视化（小步长时降低更新频率避免卡顿）
        if visualize and iter_count % 200 == 0:
            ax.plot([near_coor[0], new_coor[0]],
                    [near_coor[1], new_coor[1]],
                    [near_coor[2], new_coor[2]],
                    'b-', alpha=0.3, linewidth=0.8)
            plt.pause(0.001)  # 减少暂停时间加快绘制

        # 检查是否到达目标点附近
        if calcu_dis(new_coor, goal_point) < thr:
            print(f"✓ 找到路径！迭代次数: {iter_count}")
            path_found = True
            break

    if not path_found:
        print("✗ 路径规划失败：达到最大迭代次数")
        if visualize:
            plt.show()
        return None

    # 提取路径
    path = []
    current_idx = tree.size() - 1  # 最后一个节点的索引

    while current_idx != -1:
        path.append(tree.get_node(current_idx))
        current_idx = tree.parents[current_idx]

    # 反转路径（从起点到终点）
    path = path[::-1]

    # 补全最后一段到目标点（确保最后一步也不超过步长）
    final_points = []
    last_point = path[-1]
    remaining_dist = calcu_dis(last_point, goal_point)

    if remaining_dist > step:
        # 按步长拆分最后一段
        num_steps = int(np.ceil(remaining_dist / step))
        direction = (goal_point - last_point) / remaining_dist

        for i in range(1, num_steps):
            intermediate_point = last_point + direction * step * i
            final_points.append(intermediate_point)

    final_points.append(goal_point)
    path.extend(final_points)

    # 验证路径点间距（输出调试信息）
    path_array = np.array(path)
    distances = [calcu_dis(path_array[i], path_array[i + 1]) for i in range(len(path_array) - 1)]
    print(f"路径点数量: {len(path_array)}")
    print(f"最大步长: {max(distances):.6f} (目标步长: {step})")
    print(f"最小步长: {min(distances):.6f}")

    # 绘制最终路径
    if visualize:
        # 绘制所有树节点（小步长时简化绘制）
        if tree.size() < 5000:  # 节点数较少时才绘制全部树
            for i in range(tree.size()):
                parent_idx = tree.parents[i]
                if parent_idx != -1:
                    node = tree.get_node(i)
                    parent = tree.get_node(parent_idx)
                    ax.plot([parent[0], node[0]],
                            [parent[1], node[1]],
                            [parent[2], node[2]],
                            'b-', alpha=0.1, linewidth=0.5)

        # 绘制最终路径
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'lime', linewidth=4, label='Final Path', zorder=10)
        ax.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                   c='yellow', s=10, zorder=11)  # 缩小散点避免遮挡

        # 添加图例
        ax.legend(loc='upper left')

        # 添加信息文本
        info_text = f"Iterations: {iter_count}\nPath Length: {len(path)} points\nMax Step: {max(distances):.6f}"
        ax.text2D(0.05, 0.85, info_text, transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        plt.show()

    return np.array(path)

def test_rrt_simple_obstacles(start_point, axis_start, axis_lwh, goal_point, cube_info):
    """
    测试简单障碍物场景（步长限制为0.01）
    start_point: np.array([x, y, z]) # start point
    axis_start: np.array([x, y, z]) # fig axis start (lower left corner)
    axis_lwh: np.array([l, w, h]) # fig axis length, width, height
    goal_point: np.array([x, y, z]) # goal point
    cube_info: {'center': np.array([x, y, z]), 'size': size}
    """
    print("\n" + "=" * 60)
    print("测试简单障碍物场景（步长=0.01）")
    print("=" * 60)

    # 小步长需要增加最大迭代次数
    max_iter = 20000  # 增加迭代次数确保找到路径
    print("正在规划路径...（小步长可能需要稍长时间）")

    path = RRT(start_point, axis_start, axis_lwh, goal_point,
               cube_info=cube_info, visualize=True, step=0.01, max_iter=max_iter)

    return path

if __name__ == "__main__":
    # 测试简单场景（步长限制为0.01）
    start_point = np.array([0., 0.6161, 0.1225])
    axis_start = np.array([-1, -1, -1])
    axis_lwh = np.array([2, 2, 2])
    goal_point = np.array([0.35, 0, 0.35])

    # 简单障碍物设置
    cube_info = {
        'center': np.array([0.35, 0, 0.15]),
        'size': np.array([0.5, 0.6, 0.3]),
    }

    # 测试简单障碍物场景（步长=0.01，增加迭代次数）
    path2 = test_rrt_simple_obstacles(start_point, axis_start, axis_lwh, goal_point, cube_info)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)