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

def pose_to_matrix(pose:Pose):
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
def rotate_twist_by_pose(world_T_base: Pose, twist: Twist, inverse=True) -> Twist:
    base_matrix = pose_to_matrix(world_T_base)
    if inverse:
        # Compute inverse of base_matrix
        base_matrix = transformations.inverse_matrix(base_matrix)
    R = base_matrix[0:3, 0:3]
    v = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
    omega = np.array([twist.angular.x, twist.angular.y, twist.angular.z])

    # Transform the angular component: ω' = R * ω.
    omega_new = R.dot(omega)
    # Only rotate the twist for the linear component: v' = R * v. The translation is not rotated.
    v_new = R.dot(v)

    return Twist(linear=Vector3(x=v_new[0], y=v_new[1], z=v_new[2]), angular=Vector3(x=omega_new[0], y=omega_new[1], z=omega_new[2]))


def transform_twist_by_pose(world_T_base: Pose, twist: Twist, inverse=True) -> Twist:
    base_matrix = pose_to_matrix(world_T_base)
    if inverse:
        # Compute inverse of base_matrix
        base_matrix = transformations.inverse_matrix(base_matrix)
    
    # Extract rotation R and translation p from the transformation matrix.
    R = base_matrix[0:3, 0:3]
    p = base_matrix[0:3, 3]
    v = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
    omega = np.array([twist.angular.x, twist.angular.y, twist.angular.z])

    # Transform the angular component: ω' = R * ω.
    omega_new = R.dot(omega)
    # Transform the linear component: v' = R * v + (p x (R * ω)).
    v_new = R.dot(v) + np.cross(p, R.dot(omega))

    # Create a new Twist message with the transformed components.
    new_twist = Twist()
    new_twist.linear = Vector3(x=v_new[0], y=v_new[1], z=v_new[2])
    new_twist.angular = Vector3(x=omega_new[0], y=omega_new[1], z=omega_new[2])
    
    return new_twist

if __name__ == "__main__":
        twist = Twist()
        twist.linear.x = 1.0
        twist.angular.z = 1.0
        print(twist)
        print(negateTwist(twist))
        print(twist)
