import copy

import numpy as np

_USING_open3d = True
try:
    import open3d as o3d
    from scipy import stats
except ImportError:
    o3d = None
    stats = None
    _USING_open3d = False

_USING_cv2 = True
try:
    import cv2
except ImportError:
    cv2 = None
    _USING_cv2 = False

_USING_quaternion = True
try:
    import quaternion
except ImportError:
    quaternion = None
    _USING_quaternion = False


def depth_to_points(depth, cx, cy, fx, fy, to_m=True, return_uv=False):
    """
    :param depth:
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :param to_m:
    :return:
    """
    if not _USING_cv2:
        raise ImportError('Please run [pip install opencv-python] to install opencv.')

    vs, us = np.where(depth > 0)
    zs = depth[vs, us]

    Zcs = zs
    if to_m:
        Zcs = Zcs / 1000.0
    Xcs = (us - cx) * Zcs / fx
    Ycs = (vs - cy) * Zcs / fy

    Xcs = np.reshape(Xcs, (-1, 1))
    Ycs = np.reshape(Ycs, (-1, 1))
    Zcs = np.reshape(Zcs, (-1, 1))

    # np.concatenate(axis=1) 是按列合并，列数可以不同，但是行数得相同。
    # 行数肯定相同，因为xyz都是一个像素对应的，之前变成列，是因为一个point的坐标是[x,y,z],符合习惯
    points = np.concatenate([Xcs, Ycs, Zcs], axis=1)
    if return_uv:
        return points, vs, us
    return points


def transform_ndarray_to_pcd(points, color=None):
    """
    :param points: [N, 3] -> np.float (Input points)
    :return: pcd -> o3d format
    """
    if not _USING_open3d:
        raise ImportError('Please run [pip install open3d] to convert np.array to o3d.geometry.PointCloud')

    assert len(points.shape) == 2, 'Input points must in shape (N, 3)'
    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    if color is not None:
        pcd_points.paint_uniform_color(color)

    return pcd_points


def show_points(points, show_frame=False, frame_size=0.01):
    """
    :param points: np.array / o3d / list of o3d
    :param show_frame: bool
    :param frame_size: float
    :return:
    """
    if not _USING_open3d:
        raise ImportError('Please run [pip install open3d] to show pointcloud')
    if isinstance(points, o3d.geometry.PointCloud):
        points_list = [points]
    elif isinstance(points, np.ndarray):
        points_list = [transform_ndarray_to_pcd(points)]
    elif isinstance(points, list):
        for point in points:
            assert isinstance(point, o3d.geometry.PointCloud)
        points_list = points
    else:
        raise TypeError('Input only support <o3d>, <np.array>, '
                        '<a list of o3d>, <a list of np.array> got {}'.format(type(points)))
    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        points_list.append(frame)
    o3d.visualization.draw_geometries(points_list, window_name='Visualization')
    return


def show_points_and_frame(points, transformation_falan, transformation_predict, show_frame=False, frame_size=0.01
                          , gripper_offset=[0, 0, 0]):
    """
    :param points: np.array / o3d / list of o3d
    :param show_frame: bool
    :param frame_size: float
    :return:
    """
    if not _USING_open3d:
        raise ImportError('Please run [pip install open3d] to show pointcloud')
    if isinstance(points, o3d.geometry.PointCloud):
        points_list = [points]
    elif isinstance(points, np.ndarray):
        points_list = [transform_ndarray_to_pcd(points)]
    elif isinstance(points, list):
        for point in points:
            assert isinstance(point, o3d.geometry.PointCloud)
        points_list = points
    else:
        raise TypeError('Input only support <o3d>, <np.array>, '
                        '<a list of o3d>, <a list of np.array> got {}'.format(type(points)))

    transformation_gripper_offset = np.eye(4)
    transformation_gripper_offset[:, 3][:3] = gripper_offset

    transformation_gt = transformation_falan @ transformation_gripper_offset
    if show_frame:
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame_gt = copy.deepcopy(robot_frame).transform(transformation_gt)
        frame_predict = copy.deepcopy(robot_frame).transform(transformation_gripper_offset).transform(
            transformation_predict)
        # frame_gt.paint_uniform_color([1, 0, 0])  # red color for gt frame
        # frame_predict.paint_uniform_color([0, 1, 0])
        points_list.append(frame_gt)
        # points_list.append(frame_predict)
        points_list.append(robot_frame)
    o3d.visualization.draw_geometries(points_list, window_name='Visualization')
    return


# depth-> pcd -> o3d_pcd -> show

# show_points([o3d_pcd, falan_xyz_o3d], show_frame=True,frame_size=0.1)


def generate_transformation(rotation, translation=None):
    """
    :param rotation: (N, 3, 3) or (3, 3) -> np.float
    :param translation: (N, 3) or (3,) or None -> np.float
    :return: (N, 4, 4) or (4, 4) -> np.float (return homogeneous transformation matrix)
    """

    transformation = np.identity(4)
    if len(rotation.shape) == 3:
        N = rotation.shape[0]
        transformation = np.expand_dims(transformation, 0)
        transformation = np.tile(transformation, (N, 1, 1))
        transformation[:, :-1, :-1] = rotation
        if translation is not None:
            transformation[:, :-1, -1] = translation
    elif len(rotation.shape) == 2:
        transformation[:-1, :-1] = rotation
        if translation is not None:
            transformation[:-1, -1] = translation
    else:
        raise IOError('Rotation matrix must be in the shape of (N, 3, 3) or (3, 3), got {}'.format(rotation.shape))
    return transformation


def convert_xyzquat_to_transformation(xyzquat):
    """
    Change the xyzquat to the transformation format
    :param xyzquat: (N, 7) or (4,) -> np.float
    :return: (N, 4, 4) or (4, 4) -> np.float (return homogeneous transformation matrix)
    """
    if not _USING_quaternion:
        raise ImportError('Please run [pip install numpy-quaternion] to install quaternion.')

    if isinstance(xyzquat, list):
        xyzquat = np.array(xyzquat)
        if len(xyzquat) != 7:
            raise IOError('XYZ quaternions must be a list with a length of 7, '
                          'got list and length is {}'.format(len(xyzquat)))

    if len(xyzquat.shape) == 1:
        rotation = quaternion.from_float_array(xyzquat[3:])
        rotation = quaternion.as_rotation_matrix(rotation)
        transformation = generate_transformation(rotation, xyzquat[:3])
    elif len(xyzquat.shape) == 2:
        rotation = quaternion.from_float_array(xyzquat[:, 3:])
        rotation = quaternion.as_rotation_matrix(rotation)
        transformation = generate_transformation(rotation, xyzquat[:, :3])
    else:
        raise IOError('XYZ quaternions must be in the shape of (N, 7) or (N,), got {}'.format(xyzquat.shape))

    return transformation


def pose_to_4x4mat(pose):
    '''

    Args:
        pose: a 1x7 array x,y,z,ox,oy,oz,ow

    Returns:

    '''
    pos_x, pos_y, pos_z = pose[:3]
    orien_x, orien_y, orien_z, orien_w = pose[3:]
    translation = np.array([[1, 0, 0, pos_x],
                            [0, 1, 0, pos_y],
                            [0, 0, 1, pos_z],
                            [0, 0, 0, 1]])
    quaternion = np.array([[1 - 2 * (orien_y ** 2 + orien_z ** 2),
                            2 * (orien_x * orien_y - orien_z * orien_w),
                            2 * (orien_x * orien_z + orien_y * orien_w), 0],
                           [2 * (orien_x * orien_y + orien_z * orien_w),
                            1 - 2 * (orien_x ** 2 + orien_z ** 2),
                            2 * (orien_y * orien_z - orien_x * orien_w), 0],
                           [2 * (orien_x * orien_z - orien_y * orien_w),
                            2 * (orien_y * orien_z + orien_x * orien_w),
                            1 - 2 * (orien_x ** 2 + orien_y ** 2), 0],
                           [0, 0, 0, 1]])
    return np.matmul(translation, quaternion)


def visualize_point_clouds(point_cloud, predict_pose,
                                intri=[966.7210274621965, 626.0787916680408, 2734.497171257693, 2734.2139842813203],
                                extri=np.array([
                                    0.10431104317559005,
                                    -0.45378789156378785,
                                    0.6809180737974473,

                                    0.9977884597380278,
                                    -0.010699004055162784,
                                    -0.06559005498610207,
                                    0.0012905862086242488,
                                ]), gripper_offset=[0, 0, 0]):
    '''
    This function will call open3d to visualize the predicted_pose on point clouds
    Args:
        gt_pose: the ground truth position for next move [7,]
        predict_pose: the predicted position for next move [7,]
        intri: intri parameters of camera
        extri: extri parameters of camera

    Returns:
        None
    '''

    # poses to transformation matrixs
    predict_pose = pose_to_4x4mat(predict_pose)

    # scene point clouds
    o3d_pcd = transform_ndarray_to_pcd(point_cloud, color=[0.5, 0.5, 0.5])
    show_points_and_frame(o3d_pcd, predict_pose, predict_pose, show_frame=True, frame_size=0.1,
                          gripper_offset=gripper_offset)

def visualize_from_prediction(pcd, predict_transformation=[0,0,-0.1,0,0,0,0]):
    # check prediction and gt
    '''

    visualize the point cloud and prediction pose
    pcd: 3d 2d numpy (n, 3)
    predict_transformation: 7d numpy array, [x,y,z,ox,oy,oz,ow]
    Returns:
        None

    '''

    visualize_point_clouds(pcd, predict_transformation)

def save_points_and_frame(points, transformation_predict, show_frame=False, frame_size=0.01
                          , save_path=None):
    """
    :param points: np.array / o3d / list of o3d
    :param show_frame: bool
    :param frame_size: float
    :return:
    """

    assert '.ply' in save_path
    if not _USING_open3d:
        raise ImportError('Please run [pip install open3d] to show pointcloud')
    if isinstance(points, o3d.geometry.PointCloud):
        points_list = [points]
    elif isinstance(points, np.ndarray):
        points_list = [transform_ndarray_to_pcd(points)]
    elif isinstance(points, list):
        for point in points:
            assert isinstance(point, o3d.geometry.PointCloud)
        points_list = points
    else:
        raise TypeError('Input only support <o3d>, <np.array>, '
                        '<a list of o3d>, <a list of np.array> got {}'.format(type(points)))

    if show_frame:
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame_predict = copy.deepcopy(robot_frame).transform(
            transformation_predict)
        points_list.append(frame_predict)
        points_list.append(robot_frame)
    o3d.io.write_image("robot_frame.png", robot_frame)
    return

def save_pcd_prediction(pcd, predict_transformation=[0,0,-0.1,0,0,0,0]):
    '''

       save the point cloud and prediction pose

       predict_transformation: 7d numpy array, [x,y,z,ox,oy,oz,ow]
       Returns:
           None

       '''

    pcd = pcd[0, 0].transpose(1, 2, 0).reshape((-1, 3))

    # poses to transformation matrixs
    predict_pose = pose_to_4x4mat(predict_transformation)

    # scene point clouds
    o3d_pcd = transform_ndarray_to_pcd(pcd, color=[0.5, 0.5, 0.5])
    save_points_and_frame(o3d_pcd, predict_pose,show_frame=True, frame_size=0.1, save_path='a.ply' )

if __name__ == '__main__':


    resized_intri = [64.4480685, 66.78173778, 182.29981142, 291.64949166]
    pcd = np.load('a.npy')

    pcd = pcd.reshape(-1, 3)
    visualize_from_prediction(pcd)


