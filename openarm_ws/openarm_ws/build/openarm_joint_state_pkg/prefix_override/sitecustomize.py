import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/shiver/Desktop/openarm_ws/install/openarm_joint_state_pkg'
