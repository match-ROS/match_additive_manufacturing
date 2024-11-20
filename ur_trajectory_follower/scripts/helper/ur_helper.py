import numpy as np

from tf import transformations
from geometry_msgs.msg import Twist, Vector3

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


if __name__ == "__main__":
        twist = Twist()
        twist.linear.x = 1.0
        twist.angular.z = 1.0
        print(twist)
        print(negateTwist(twist))
        print(twist)
