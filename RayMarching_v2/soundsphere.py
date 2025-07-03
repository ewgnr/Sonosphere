import numpy as np
import transforms3d as t3d
import math


class SoundSphere():
    
    def __init__(self, object_count, radius):
        
        self.object_count = object_count
        self.radius = radius
        
        self.objectPositions = np.random.rand(self.object_count, 3)
        self.objectRotations = np.random.rand(self.object_count, 4)
        self.objectTransforms = np.zeros((self.object_count, 4, 4))
        
        self.spherePosition = np.array([0.0, 0.0, 0.0])
        self.sphereRotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.sphereTransform = np.eye(4)
        
        self.updateSphere()
        
    @staticmethod
    def evenly_spaced_sphere_points(n, radius=1.0):
        """Generate n evenly spaced points using the Golden Spiral method."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        indices = np.arange(n)
        z = (1 - 2 * indices / (n - 1)) * radius  # Linearly spaced from radius to -radius
        r = np.sqrt(radius**2 - z**2)  # Radius at height z
        theta = 2 * np.pi * indices / phi  # Golden angle increment
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y, z))
    
    @staticmethod
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r != 0 else 0  # handle r = 0 case
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    @staticmethod
    def quaternion_from_vectors(v1, v2):
        """Return quaternion to rotate v1 to v2."""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        cos_theta = np.dot(v1, v2)
        # Handle parallel or anti-parallel vectors
        if np.isclose(cos_theta, 1.0):
            return np.array([0., 0., 0., 1.])  # Identity quaternion
        elif np.isclose(cos_theta, -1.0):
            return np.array([1., 0., 0., 0.])  # 180° around x-axis (could be any axis, but this is a simple choice)
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(cos_theta)
        sin_half = np.sin(angle/2)
        return np.array([axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half, np.cos(angle/2)])
    
    @staticmethod
    def rotation_towards_center(position):
        """
        Calculate quaternion to rotate object's forward (z-axis) to point towards the sphere's center.
        
        Args:
            position (np.ndarray): 3D position on sphere's surface
        
        Returns:
            np.ndarray: [x, y, z, w] quaternion
        """
        forward = -position / np.linalg.norm(position)  # Direction toward center
        default_forward = np.array([0, 0, 1])           # Object's default forward (z-axis)
        return SoundSphere.quaternion_from_vectors(default_forward, forward)
    
    def updateSphere(self):
        
        self.objectPositions = self.evenly_spaced_sphere_points(self.object_count, self.radius)
        
        """
        # for the moment, fixed object rotations
        for oI in range(self.object_count):
            self.objectRotations[oI] = t3d.euler.euler2quat(0.0, 0.0, 0.0, axes='sxyz')
        """
            
        for oI in range(self.object_count):
            
            tmp_quat = self.rotation_towards_center(self.objectPositions[oI])
            
            # convert quaternion from w x y z to x y z w
            self.objectRotations[oI][0] = tmp_quat[1]
            self.objectRotations[oI][1] = tmp_quat[2]
            self.objectRotations[oI][2] = tmp_quat[3]
            self.objectRotations[oI][3] = tmp_quat[0]

            
    
        
        self.updateTransforms()
        
    def setSpherePosition(self, position):
        self.spherePosition = position    
        
        self.updateSphereTransform()
 
    def setSphereRotation(self, rotation):
        self.sphereRotation = rotation    
        
        self.updateSphereTransform()       
 
    def updateSphereTransform(self):
        
        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))
        
        sphereRotMat = t3d.quaternions.quat2mat(self.sphereRotation)
        sphereTransMat = t3d.affines.compose(self.spherePosition, defaultRotMat, defaultScale)
        sphereRotMat = t3d.affines.compose(defaultPos, sphereRotMat, defaultScale)

        self.sphereTransform = np.transpose(np.matmul(sphereRotMat, sphereTransMat))   
        
        self.updateTransforms()
        
        
    def setRadius(self, radius):
        
        self.radius = radius
        
        self.updateSphere()
        
    def updateTransforms(self):
        
        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))

        for oI in range(self.object_count):
            
            objectPosition = self.objectPositions[oI]
            objectRotation = self.objectRotations[oI] 

            objectRotMat = t3d.quaternions.quat2mat(objectRotation)
            objectRotMat = t3d.affines.compose(defaultPos, objectRotMat, defaultScale)
   
            objectTransMat = t3d.affines.compose(objectPosition, defaultRotMat, defaultScale)

            #self.objectTransforms[oI] = np.transpose(np.matmul(objectRotMat, objectTransMat))
            
            objectTransform = np.matmul(objectRotMat, objectTransMat)
            
            self.objectTransforms[oI] = np.transpose(np.matmul(objectTransform, self.sphereTransform))

    def getObjectCount(self):
        return self.object_count
    
    def getObjectPositions(self):
        return self.objectPositions
    
    def getObjectRotations(self):
        return self.objectRotations
    
    def getObjectTransforms(self):
        return self.objectTransforms
    
    
