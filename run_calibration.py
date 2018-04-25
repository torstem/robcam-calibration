import cPickle as pickle
import numpy
import chessboard
import park_martin
import yaml
numpy.set_printoptions(linewidth=300)
from scipy.linalg import expm3, inv
from numpy import dot, eye

img_list = pickle.load(open('image_list.dump'))
rob_pose_list = pickle.load(open('pose_list.dump'))
corner_list = []
obj_pose_list = []

camera_matrix, dist_coeffs = chessboard.calibrate_lens(img_list)

def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm3([[   0, -r[2],  r[1]],
                           [r[2],     0, -r[0]],
                           [-r[1],  r[0],    0]])
    res[0:3, -1] = t
    return res

for i, img in enumerate(img_list):
    found, corners = chessboard.find_corners(img)
    corner_list.append(corners)
    if not found:
        raise Exception("Failed to find corners in img # %d" % i)
    rvec, tvec = chessboard.get_object_pose(chessboard.pattern_points, corners, camera_matrix, dist_coeffs)
    object_pose = tf_mat(rvec, tvec)
    obj_pose_list.append(object_pose)

A, B = [], []
for i in range(1,len(img_list)):
    p = rob_pose_list[i-1], obj_pose_list[i-1]
    n = rob_pose_list[i], obj_pose_list[i]
    A.append(dot(inv(p[0]), n[0]))
    B.append(dot(inv(p[1]), n[1]))


# Transformation to chessboard in robot gripper
cie = eye(4)
Rx, tx = park_martin.calibrate(A, B)
cie[0:3, 0:3] = Rx
cie[0:3, -1] = tx

# Compute transformations to camera.
# All the transformations should be quite similar
for i in range(len(img_list)):
    rob = rob_pose_list[i]
    obj = obj_pose_list[i]
    tmp = dot(rob, dot(cie, inv(obj)))
    print(tmp)

# Here I just pick one, but maybe some average can be used instead
rob = rob_pose_list[0]
obj = obj_pose_list[0]
cam_pose = dot(dot(rob, cie), inv(obj))

cam = {'rotation' : cam_pose[0:3, 0:3].tolist(),
       'translation' : cam_pose[0:3, -1].tolist(),
       'camera_matrix' : camera_matrix.tolist(),
       'dist_coeffs' : dist_coeffs.tolist()}

fp = open('camera.yaml', 'w')
fp.write(yaml.dump(cam))
fp.close()
