import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/regmed/dregmed/vis_to_nav/install/visualizer'
