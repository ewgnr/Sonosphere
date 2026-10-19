import numpy as np
import transforms3d as t3d
import math

class SoundSphere():
    
    def __init__(self, stack_count, slice_count, radius):
        
        self.stack_count = stack_count
        self.slice_count = slice_count
        self.radius = radius
        self.object_count = self.stack_count  * self.slice_count + 2
        
        self.objectPositions = np.random.rand(self.object_count, 3)
        self.objectRotations = np.random.rand(self.object_count, 4)
        self.objectTransforms = np.zeros((self.object_count, 4, 4))
        
        self.spherePosition = np.array([0.0, 0.0, 0.0])
        self.sphereRotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.sphereTransform = np.eye(4)
        
        self.updateSphere()
        
    def updateSphere(self):
        # add top object
        object_pos = np.array([0.0, 1.0, 0.0], dtype=np.float32) * self.radius
        object_rot = t3d.euler.euler2quat(0.0, 0.0, 0.0, axes='sxyz')
        
        self.objectPositions[0] = object_pos
        self.objectRotations[0] = object_rot
        
        # add objects per stack / slice
        oI = 1
        
        for i in range(self.stack_count):
            
            phi = (i + 1) * np.pi / (self.stack_count + 1)
            
            #phi = np.pi * i / self.stack_count
            
            for j in range(self.slice_count):
                
                theta = 2.0 * np.pi * j / self.slice_count
                
                object_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                """
                object_pos[0] = math.sin(phi) * math.cos(theta)
                object_pos[1] = math.cos(phi)
                object_pos[2] = math.sin(phi) * math.sin(theta)
                """
    
                
                object_pos[0] = math.sin(phi) * math.cos(theta)
                object_pos[2] = math.sin(phi) * math.sin(theta)
                object_pos[1] = math.cos(phi)
                
                """
                object_pos[0] = math.sin(theta) * math.cos(phi)
                object_pos[1] = math.cos(theta)
                object_pos[2] = math.sin(theta) * math.sin(phi)
                """
                

                #object_rot = t3d.euler.euler2quat(phi, theta, 0.0, axes='sxyz')
                object_rot = t3d.euler.euler2quat(0.0, theta, phi, axes='sxyz')
                
                object_pos = object_pos * self.radius
                
                self.objectPositions[oI] = object_pos
                self.objectRotations[oI] = object_rot
                
                print("i ", i, " j ", j, " oI ", oI, " phi ", phi, " theta ", theta, " pos ", object_pos[0], " ", object_pos[1], " ", object_pos[2])
                
                oI += 1
                
        print("self.object_count ", self.object_count)

        
        # add bottom object
        object_pos = np.array([0.0, -1.0, 0.0], dtype=np.float32) * self.radius
        object_rot = t3d.euler.euler2quat(0.0, 0.0, 0.0, axes='sxyz')
        
        self.objectPositions[oI] = object_pos
        self.objectRotations[oI] = object_rot
        
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
    
    
