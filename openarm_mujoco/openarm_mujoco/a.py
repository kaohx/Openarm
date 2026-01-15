import time
import ikpy.chain
import numpy as np
import mujoco.viewer
import transforms3d as tf

from scipy.spatial.transform import Rotation as R

from RRT import test_rrt_simple_obstacles

def main():
    model = mujoco.MjModel.from_xml_path('model/scene.xml')
    data = mujoco.MjData(model)
    my_chain_left = ikpy.chain.Chain.from_urdf_file("model/openarm_bimanual_control.urdf",
                                                    base_elements=["openarm_body_link0"],
                                                    active_links_mask=[False]*2 + [True]*7 + [False]*2)
    """
    links:
        00 -> Base link bounds = (-inf, inf)
        01 -> openarm_left_openarm_body_link0_joint type: revolute bounds = (-inf, inf)
        02 -> openarm_left_joint1 type: revolute bounds = (-3.490659, 1.3962629999999998)
        03 -> openarm_left_joint2 type: revolute bounds = (-3.3161253267948965, 0.17453267320510335)
        04 -> openarm_left_joint3 type: revolute bounds = (-1.570796, 1.570796)
        05 -> openarm_left_joint4 type: revolute bounds = (0.0, 2.443461)
        06 -> openarm_left_joint5 type: revolute bounds = (-1.570796, 1.570796)
        07 -> openarm_left_joint6 type: revolute bounds = (-0.785398, 0.785398)
        08 -> openarm_left_joint7 type: revolute bounds = (-1.570796, 1.570796)
        09 -> left_openarm_hand_joint type: fixed bounds = (-inf, inf)
        10 -> openarm_left_hand_tcp_joint type: fixed bounds = (-inf, inf)
    """
    # [0, 0.15349774, 0.08189955] end effector position
    tmp_joint_angle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tf_mat = my_chain_left.forward_kinematics(tmp_joint_angle)
    ee_pos = tf_mat[:3, 3]
    rotation = R.from_matrix(tf_mat[:3, :3])
    ee_euler_tmp = rotation.as_euler('xyz', degrees=True)
    ee_euler = [i * (np.pi / 180) for i in ee_euler_tmp]

    # my_chain_right = ikpy.chain.Chain.from_urdf_file("model/openarm_bimanual_control.urdf",
    #                                                  base_elements=["openarm_right_link0"])
    axis_start = np.array([-1, -1, -1])
    axis_lwh = np.array([2, 2, 2])
    target_pos = np.array([0.35, 0, 0.225])
    target_euler = np.array([0, 0, 1.57])
    cube_info = [
    {
        'center': np.array([0.35, 0, 0.15]),
        'size': np.array([0.25, 0.3, 0.15]),
    }
    ]
    path = test_rrt_simple_obstacles(ee_pos, axis_start, axis_lwh, target_pos, cube_info)
    path_len = len(path)
    dx, dy, dz = (ee_euler - target_euler) / path_len

    joint_angle = [tmp_joint_angle]

    i = 0
    for p in path:
        tmp_pos = p
        tmp_euler = [ee_euler[0] + dx, ee_euler[1] + dy, ee_euler[2] + dz]
        joint_angle.append(my_chain_left.inverse_kinematics(tmp_pos, tf.euler.euler2mat(*tmp_euler), "all", initial_position=joint_angle[i]))
        i += 1

    # data_len = len(path)
    # data[ data_len - 2][-2], data[ data_len - 2][-1] = 1, 1
    # data[ data_len - 1][-2], data[ data_len - 1][-1] = 0, 0

    # ee_pos = [-0.1, 0.1, 0.1] # end effector position
    # ee_euler = [0, 0, 0] # end effector euler angle

    # ee_orientation = tf.euler.euler2mat(*ee_euler) # end effector orientation matrix
    # #
    # joint_angles = my_chain_left.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    # # ctrl = joint_angles[1:-1]
    # # data.ctrl[:6] = ctrl
    print(path)
    i = 0
    data.ctrl[0:7] = tmp_joint_angle[2:9]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # # ========== 第一步：执行初始的仿真步和视图同步 ==========
        # # 执行一次初始仿真步
        # mujoco.mj_step(model, data)
        # # 强制同步视图（确保初始状态更新）
        # viewer.sync()
        #
        # # 可选：添加短暂等待，确保视图完全刷新（解决异步更新问题）
        # time.sleep(3)  # 等待0.1秒，足够视图完成刷新

        # ========== 第二步：确认初始步完成后，再执行后续循环 ==========
        # 遍历关节角度数组，控制机器人运动
        for i in range(len(joint_angle)):
            # 检查Viewer是否还在运行
            if not viewer.is_running():
                break

            # 设置关节控制指令（提取第2到第8列的关节角度）
            ctrl = joint_angle[i][2:9]
            data.ctrl[0:7] = ctrl

            # 如果需要同时控制夹爪，取消下面的注释
            # data.ctrl[14] = 0.02  # 左夹爪1
            # data.ctrl[15] = 0.02  # 左夹爪2

            # 执行一次仿真步
            mujoco.mj_step(model, data)
            # 同步视图
            viewer.sync()
            # 控制仿真频率（500Hz，可根据需要调整）
            time.sleep(0.002)

        # ========== 第三步：保持最终位置 ==========
        # 循环保持最终状态，直到关闭Viewer
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)  # 降低保持阶段的刷新频率（100Hz）


if __name__ == '__main__':
    main()