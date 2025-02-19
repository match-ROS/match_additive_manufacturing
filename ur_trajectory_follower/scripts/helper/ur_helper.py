import numpy as np

from tf import transformations
from geometry_msgs.msg import Twist, Vector3, Pose

Vector3.__mul__ = lambda self, other: Vector3(x=self.x*other, y=self.y*other, z=self.z*other)
Vector3.__rmul__ = lambda self, other: Vector3(x=self.x*other, y=self.y*other, z=self.z*other)
Vector3.__neg__ = lambda self: Vector3(x=-self.x, y=-self.y, z=-self.z)

def rotateVector(vec=(0.0, 0.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), transpose=False):
    if transpose:
        rot_conj = rot
        rot = transformations.quaternion_conjugate(rot_conj)
    else:
        rot_conj = transformations.quaternion_conjugate(rot)
    trans = transformations.quaternion_multiply(transformations.quaternion_multiply(rot_conj, vec), rot)[:3]
    return trans

def negateTwist(twist: Twist):
    return Twist(linear=-twist.linear, angular=-twist.angular)

def pose_to_matrix(pose):
    """Convert Pose to transformation matrix"""
    trans = [pose.position.x, pose.position.y, pose.position.z]
    rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    matrix = transformations.quaternion_matrix(rot)
    matrix[0:3, 3] = trans
    return matrix

def transform_pose_by_pose(world_T_base: Pose, world_T_ee: Pose, inverse=(True, False)) -> Pose:
    base_matrix = pose_to_matrix(world_T_base)
    ee_matrix = pose_to_matrix(world_T_ee)

    if inverse[0]:
        # Compute inverse of base_matrix
        base_matrix = transformations.inverse_matrix(base_matrix)
    if inverse[1]:
        # Compute inverse of ee_matrix
        ee_matrix = transformations.inverse_matrix(ee_matrix)
    # Get ee in base frame
    base_T_ee_matrix = transformations.concatenate_matrices(base_matrix, ee_matrix)

    # Convert back to Pose
    pose = Pose()
    translation = base_T_ee_matrix[0:3, 3]
    rotation = transformations.quaternion_from_matrix(base_T_ee_matrix)

    pose.position.x, pose.position.y, pose.position.z = translation
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rotation

    return pose

if __name__ == "__main__":
        twist = Twist()
        twist.linear.x = 1.0
        twist.angular.z = 1.0
        print(twist)
        print(negateTwist(twist))
        print(twist)
