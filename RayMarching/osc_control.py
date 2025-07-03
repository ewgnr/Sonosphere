import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server

from skeleton import SkeletonMode

class OscControl():
    
    def __init__(self, skeleton, visualization, address, port):
        
        self.skeleton = skeleton
        self.visualization = visualization
        self.address = address 
        self.port = port
         
        self.dispatcher = dispatcher.Dispatcher()
        self.mocap_running = True
        
        self.dispatcher.map("/mocap/initskelconfig", self.initSkeletonConfig)
        self.dispatcher.map("/mocap/start", self.startMocap)
        self.dispatcher.map("/mocap/stop", self.stopMocap)
        
        self.dispatcher.map("/mocap/updatesmoothing", self.setMocapUpdateSmoothing)
        
        self.dispatcher.map("/mocap/skelposworld", self.setMocapSkeletonPosition)
        self.dispatcher.map("/mocap/rootzero", self.setMocapRootZero)
        self.dispatcher.map("/mocap/joint/pos_world", self.setMocapJointPositions)
        self.dispatcher.map("/mocap/joint/rot_world", self.setMocapJointRotations)
        
        self.dispatcher.map("/vis/camposition", self.setVisCamPosition)
        self.dispatcher.map("/vis/camangle", self.setVisCamAngle)
        
        self.dispatcher.map("/vis/sceneposition", self.setVisScenePosition)
        self.dispatcher.map("/vis/scenerotation", self.setVisSceneRotation)
        
        self.dispatcher.map("/vis/lightposition", self.setVisLightPosition)
        self.dispatcher.map("/vis/shadowsmooth", self.setVisShadowSmooth)
        self.dispatcher.map("/vis/shadowstrength", self.setVisShadowStrength)
        
        self.dispatcher.map("/vis/bgcolor", self.setVisBGColor)
        self.dispatcher.map("/vis/bgocclusioncolor", self.setVisBGOcclusionColor)
        
        self.dispatcher.map("/vis/fogmindist", self.setVisFogMinDist)
        self.dispatcher.map("/vis/fogmaxdist", self.setVisFogMaxDist)

        self.dispatcher.map("/vis/rayrotation", self.setVisRayRot)
        self.dispatcher.map("/vis/raywiggle", self.setVisRayWig)

        
        self.dispatcher.map("/vis/jointcolor", self.setVisJointColor)
        self.dispatcher.map("/vis/jointambientscale", self.setVisJointAmbientScale)
        self.dispatcher.map("/vis/jointdiffusescale", self.setVisJointDiffuseScale)
        self.dispatcher.map("/vis/jointspecularscale", self.setVisJointSpecularScale)
        self.dispatcher.map("/vis/jointspecularpow", self.setVisJointSpecularPow)
        self.dispatcher.map("/vis/jointocclusionscale", self.setVisJointOcclusionScale)
        self.dispatcher.map("/vis/jointocclusionrange", self.setVisJointOcclusionRange)
        self.dispatcher.map("/vis/jointocclusionresolution", self.setVisJointOcclusionResolution)
        self.dispatcher.map("/vis/jointocclusioncolor", self.setVisJointOcclusionColor)
        self.dispatcher.map("/vis/jointprimitive", self.setVisJointPrimitive)
        self.dispatcher.map("/vis/jointsize", self.setVisJointSize)
        self.dispatcher.map("/vis/jointround", self.setVisJointRounding)
        self.dispatcher.map("/vis/jointsmooth", self.setVisJointSmoothing)
        
        self.dispatcher.map("/vis/edgecolor", self.setVisEdgeColor)
        self.dispatcher.map("/vis/edgeambientscale", self.setVisEdgeAmbientScale)
        self.dispatcher.map("/vis/edgediffusescale", self.setVisEdgeDiffuseScale)
        self.dispatcher.map("/vis/edgespecularscale", self.setVisEdgeSpecularScale)
        self.dispatcher.map("/vis/edgespecularpow", self.setVisEdgeSpecularPow)
        self.dispatcher.map("/vis/edgeocclusionscale", self.setVisEdgeOcclusionScale)
        self.dispatcher.map("/vis/edgeocclusionrange", self.setVisEdgeOcclusionRange)
        self.dispatcher.map("/vis/edgeocclusionresolution", self.setVisEdgeOcclusionResolution)
        self.dispatcher.map("/vis/edgeocclusioncolor", self.setVisEdgeOcclusionColor)
        self.dispatcher.map("/vis/edgeprimitive", self.setVisEdgePrimitive)
        self.dispatcher.map("/vis/edgesize", self.setVisEdgeSize)
        self.dispatcher.map("/vis/edgeround", self.setVisEdgeRounding)
        self.dispatcher.map("/vis/edgesmooth", self.setVisEdgeSmoothing)
        
        self.dispatcher.map("/vis/objectcolor", self.setVisObjectColor)
        self.dispatcher.map("/vis/objectambientscale", self.setVisObjectAmbientScale)
        self.dispatcher.map("/vis/objectdiffusescale", self.setVisObjectDiffuseScale)
        self.dispatcher.map("/vis/objectspecularscale", self.setVisObjectSpecularScale)
        self.dispatcher.map("/vis/objectspecularpow", self.setVisObjectSpecularPow)
        self.dispatcher.map("/vis/objectocclusionscale", self.setVisObjectOcclusionScale)
        self.dispatcher.map("/vis/objectocclusionrange", self.setVisObjectOcclusionRange)
        self.dispatcher.map("/vis/objectocclusionresolution", self.setVisObjectOcclusionResolution)
        self.dispatcher.map("/vis/objectocclusioncolor", self.setVisObjectOcclusionColor)
        self.dispatcher.map("/vis/objectprimitive", self.setVisObjectPrimitive)
        self.dispatcher.map("/vis/objectfrequency", self.setVisObjectFrequency)
        self.dispatcher.map("/vis/objectamplitude", self.setVisObjectAmplitude)
        self.dispatcher.map("/vis/objectphase", self.setVisObjectPhase)
        self.dispatcher.map("/vis/objectsize", self.setVisObjectSize)
        self.dispatcher.map("/vis/objectround", self.setVisObjectRounding)
        self.dispatcher.map("/vis/objectsmooth", self.setVisObjectSmoothing)
        self.dispatcher.map("/vis/objectposition", self.setVisObjectPosition)
        self.dispatcher.map("/vis/objectrotation", self.setVisObjectRotation)
 
        self.dispatcher.map("/vis/jointedgesmooth", self.setVisJointEdgeSmoothing)
        self.dispatcher.map("/vis/skelobjectsmooth", self.setVisSkelObjectSmoothing)
        
        self.server = osc_server.ThreadingOSCUDPServer((self.address, self.port), self.dispatcher)
        
        self.fix_root = False
        
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def initSkeletonConfig(self, address, *args):

        self.skeleton.initConfig(args[0])
        self.visualization.resetSkeleton(self.skeleton)
        
    def startMocap(self, address, *args):
        
        print("startMocap")
        
        self.visualization.startVis()
        
        self.mocap_running = True

        
    def stopMocap(self, address, *args):
        
        print("stopMocap")
        
        self.visualization.stopVis()
        
        self.mocap_running = False
    
    def setMocapUpdateSmoothing(self, address, *args):
        
        self.skeleton.setUpdateSmoothing(args[0])
        
    def setMocapRootZero(self, address, *args):
        
        self.fix_root = bool(args[0])
        
    def setMocapSkeletonPosition(self, address, *args):
        
        position = np.array(args)
        
        self.skeleton.setPosition(position)
        
    def setMocapJointPositions(self, address, *args):

        if self.mocap_running == False:
            return

        argCount = len(args)
        posCount = argCount // 3
        
        positions = np.array(args)
        positions = np.reshape(positions, (-1, 3))
        
        #print("positions ", positions)
        
        if self.fix_root == True:
            # set root positions to zero
            positions -= positions[0]
        
        if self.skeleton.skeletonMode == SkeletonMode.Avatar or self.skeleton.skeletonMode == SkeletonMode.SnakeAvatar or self.skeleton.skeletonMode == SkeletonMode.Hand :
            # right handed to left handed
            tmp = np.copy(positions)

            """
            positions[:, 0] = tmp[:, 0] / 100.0
            positions[:, 1] = tmp[:, 1] / 100.0
            positions[:, 2] = tmp[:, 2] / 100.0
            """
            
            positions[:, 0] = tmp[:, 1] / 100.0
            positions[:, 1] = tmp[:, 0] / 100.0
            positions[:, 2] = tmp[:, 2] / 100.0

            """
            positions[:, 0] = tmp[:, 1] / 100.0
            positions[:, 1] = tmp[:, 0] / 100.0
            positions[:, 2] = tmp[:, 2] / 100.0
            """

        
        #print("positions ", positions)
        
        self.skeleton.setJointPositions(positions)

    def setMocapJointRotations(self, address, *args):
        
        if self.mocap_running == False:
            return

        argCount = len(args)
        rotCount = argCount // 4
        
        rotations = np.array(args)
        rotations = np.reshape(rotations, (-1, 4))

        if self.skeleton.skeletonMode == SkeletonMode.Avatar or self.skeleton.skeletonMode == SkeletonMode.SnakeAvatar or self.skeleton.skeletonMode == SkeletonMode.Hand:
            tmp = np.copy(rotations)
            
            #print("rotations 0 ", rotations[0])

            """
            rotations[:, 0] = tmp[:, 0]
            rotations[:, 1] = tmp[:, 1]
            rotations[:, 2] = tmp[:, 2]
            rotations[:, 3] = tmp[:, 3]
            """

            rotations[:, 0] = tmp[:, 0]
            rotations[:, 1] = tmp[:, 2]
            rotations[:, 2] = tmp[:, 1]
            rotations[:, 3] = tmp[:, 3]
        
        self.skeleton.setJointRotations(rotations)
        
    def setVisCamPosition(self, address, *args):
        
        position = np.array(args)

        self.visualization.setCamPosition(position)

    def setVisCamAngle(self, address, *args):
        
        angle = np.array(args)

        self.visualization.setCamAngle(angle)
        
    def setVisScenePosition(self, address, *args):
        
        position = np.array(args)

        self.visualization.setScenePosition(position)

    def setVisSceneRotation(self, address, *args):
        
        rotation = np.array(args)

        self.visualization.setSceneRotation(rotation)
        
    def setVisLightPosition(self, address, *args):
        
        position = np.array(args)
        
        self.visualization.setLightPosition(position)
        
    def setVisShadowSmooth(self, address, *args):
        
        smooth = args[0]

        self.visualization.setShadowSmooth(smooth)
        
    def setVisShadowStrength(self, address, *args):
        
        strength = args[0]

        self.visualization.setShadowStrength(strength)

    def setVisBGColor(self, address, *args):
        
        color = np.array(args)
        
        self.visualization.setBGColor(color)
        
    def setVisBGOcclusionColor(self, address, *args):

        color = np.array(args)
        
        self.visualization.setBGOcclusionColor(color)

    def setVisFogMinDist(self, address, *args):
        
        dist = args[0]

        self.visualization.setFogMinDist(dist)
        
        #print("setVisFogMinDist ", dist)
        
    def setVisFogMaxDist(self, address, *args):
        
        dist = args[0]

        self.visualization.setFogMaxDist(dist)
        
        #print("setVisFogMaxDist ", dist)


    def setVisRayRot(self, address, *args):

        rot = args[0]

        self.visualization.setRayRotation(rot)

    def setVisRayWig(self, address, *args):

        wig = args[0]

        self.visualization.setRayWiggle(wig)


    def setVisJointColor(self, address, *args):

        color = np.array(args)
        
        self.visualization.setJointColor(color)        

    def setVisJointAmbientScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setJointAmbientScale(scale)

    def setVisJointDiffuseScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setJointDiffuseScale(scale)

    def setVisJointSpecularScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setJointSpecularScale(scale)

    def setVisJointSpecularPow(self, address, *args):
        
        pow_ = args[0]
        
        self.visualization.setJointSpecularPow(pow_)

    def setVisJointOcclusionScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setJointOcclusionScale(scale)

    def setVisJointOcclusionRange(self, address, *args):
        
        range_ = args[0]

        self.visualization.setJointOcclusionRange(range_)

    def setVisJointOcclusionResolution(self, address, *args):
        
        resolution = args[0]

        self.visualization.setJointOcclusionResolution(resolution)
        
    def setVisJointOcclusionColor(self, address, *args):

        color = np.array(args)
        
        self.visualization.setJointOcclusionColor(color)

    def setVisJointPrimitive(self, address, *args):

        if len(args) == 1:
            
            primitive = args[0]
            
            self.visualization.setJointPrimitives(primitive)
            
        elif len(args) == 2:
            
            index = args[0]
            primitive = args[1]
            
            self.visualization.setJointPrimitive(index, primitive)

    def setVisJointSize(self, address, *args):

        if len(args) == 3:
            
            size = np.array(args)
            
            self.visualization.setJointSizes(size)
            
        elif len(args) == 4:
            
            index = args[0]
            size = np.array(args[1:])
            
            self.visualization.setJointSize(index, size)

    def setVisJointRounding(self, address, *args):
        
        if len(args) == 1:
            
            round = args[0]
            
            self.visualization.setJointRoundings(round)
            
        elif len(args) == 2:
            
            index = args[0]
            round = args[1]
            
            self.visualization.setJointRounding(index, round)

    def setVisJointSmoothing(self, address, *args):
        
        if len(args) == 1:
            
            smooth = args[0]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setJointSmoothings(smooth)
            
        elif len(args) == 2:
            
            index = args[0]
            smooth = args[1]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setJointSmoothing(index, smooth)

    def setVisEdgeColor(self, address, *args):

        color = np.array(args)
        
        self.visualization.setEdgeColor(color)        

    def setVisEdgeAmbientScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setEdgeAmbientScale(scale)

    def setVisEdgeDiffuseScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setEdgeDiffuseScale(scale)

    def setVisEdgeSpecularScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setEdgeSpecularScale(scale)

    def setVisEdgeSpecularPow(self, address, *args):
        
        pow_ = args[0]
        
        self.visualization.setEdgeSpecularPow(pow_)

    def setVisEdgeOcclusionScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setEdgeOcclusionScale(scale)

    def setVisEdgeOcclusionRange(self, address, *args):
        
        range_ = args[0]

        self.visualization.setEdgeOcclusionRange(range_)

    def setVisEdgeOcclusionResolution(self, address, *args):
        
        resolution = args[0]

        self.visualization.setEdgeOcclusionResolution(resolution)
        
    def setVisEdgeOcclusionColor(self, address, *args):

        color = np.array(args)
        
        self.visualization.setEdgeOcclusionColor(color)

    def setVisEdgePrimitive(self, address, *args):

        if len(args) == 1:
            
            primitive = args[0]
            
            self.visualization.setEdgePrimitives(primitive)
            
        elif len(args) == 2:
            
            index = args[0]
            primitive = args[1]
            
            self.visualization.setEdgePrimitive(index, primitive)

    def setVisEdgeSize(self, address, *args):

        if len(args) == 3:
            
            size = np.array(args)
            
            self.visualization.setEdgeSizes(size)
            
        elif len(args) == 4:
            
            index = args[0]
            size = np.array(args[1:])
            
            self.visualization.setEdgeSize(index, size)

    def setVisEdgeRounding(self, address, *args):
        
        if len(args) == 1:
            
            round = args[0]
            
            self.visualization.setEdgeRoundings(round)
            
        elif len(args) == 2:
            
            index = args[0]
            round = args[1]
            
            self.visualization.setEdgeRounding(index, round)

    def setVisEdgeSmoothing(self, address, *args):
        
        if len(args) == 1:
            
            smooth = args[0]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setEdgeSmoothings(smooth)
            
        elif len(args) == 2:
            
            index = args[0]
            smooth = args[1]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setEdgeSmoothing(index, smooth)

    def setVisObjectColor(self, address, *args):

        if len(args) == 3:
            
            color = np.array(args)
            
            self.visualization.setObjectColors(color)
            
        elif len(args) == 4:
            
            index = args[0]
            color = np.array(args[1:])
            
            self.visualization.setObjectColor(index, color)

    def setVisObjectAmbientScale(self, address, *args):

        if len(args) == 1:
            
            scale = args[0]
            
            self.visualization.setObjectAmbientScales(scale)
            
        elif len(args) == 2:
            
            index = args[0]
            scale = args[1]
            
            self.visualization.setObjectAmbientScale(index, scale)        

    def setVisObjectDiffuseScale(self, address, *args):

        if len(args) == 1:
            
            scale = args[0]
            
            self.visualization.setObjectDiffuseScales(scale)
            
        elif len(args) == 2:
            
            index = args[0]
            scale = args[1]
            
            self.visualization.setObjectDiffuseScale(index, scale)    

    def setVisObjectSpecularScale(self, address, *args):

        if len(args) == 1:
            
            scale = args[0]
            
            self.visualization.setObjectSpecularScales(scale)
            
        elif len(args) == 2:
            
            index = args[0]
            scale = args[1]
            
            self.visualization.setObjectSpecularScale(index, scale)   

    def setVisObjectSpecularPow(self, address, *args):

        if len(args) == 1:
            
            pow_ = args[0]
            
            self.visualization.setObjectSpecularPows(pow_)
            
        elif len(args) == 2:
            
            index = args[0]
            pow_ = args[1]
            
            self.visualization.setObjectSpecularPow(index, pow_)   
            
    def setVisObjectOcclusionScale(self, address, *args):

        if len(args) == 1:
            
            scale = args[0]
            
            self.visualization.setObjectOcclusionScales(scale)
            
        elif len(args) == 2:
            
            index = args[0]
            scale = args[1]
            
            self.visualization.setObjectOcclusionScale(index, scale)   

    def setVisObjectOcclusionRange(self, address, *args):

        if len(args) == 1:
            
            range_ = args[0]
            
            self.visualization.setObjectOcclusionRanges(range_)
            
        elif len(args) == 2:
            
            index = args[0]
            range_ = args[1]
            
            self.visualization.setObjectOcclusionRange(index, range_)   

    def setVisObjectOcclusionResolution(self, address, *args):

        if len(args) == 1:
            
            resolution = args[0]
            resolution = max(resolution, 0.001)
            
            self.visualization.setObjectOcclusionResolutions(resolution)
            
        elif len(args) == 2:
            
            index = args[0]
            resolution = args[1]
            resolution = max(resolution, 0.001)
            
            self.visualization.setObjectOcclusionResolution(index, resolution)   

    def setVisObjectOcclusionColor(self, address, *args):

        if len(args) == 3:
            
            color = np.array(args)
            
            self.visualization.setObjectOcclusionColors(color)
            
        elif len(args) == 4:
            
            index = args[0]
            color = np.array(args[1:])
            
            self.visualization.setObjectOcclusionColor(index, color)
            
    def setVisObjectFrequency(self, address, *args):
        
        if len(args) == 3:
            
            frequency = np.array(args)
            
            self.visualization.setObjectFrequencies(frequency)
            
        elif len(args) == 4:
            
            index = args[0]
            frequency = np.array(args[1:])
            
            self.visualization.setObjectFrequency(index, frequency)

    def setVisObjectAmplitude(self, address, *args):
        
        if len(args) == 3:
            
            amplitude = np.array(args)
            
            self.visualization.setObjectAmplitudes(amplitude)
            
        elif len(args) == 4:
            
            index = args[0]
            amplitude = np.array(args[1:])
            
            self.visualization.setObjectAmplitude(index, amplitude)                    

    def setVisObjectPhase(self, address, *args):
        
        if len(args) == 3:
            
            phase = np.array(args)
            
            self.visualization.setObjectPhases(phase)
            
        elif len(args) == 4:
            
            index = args[0]
            phase = np.array(args[1:])
            
            self.visualization.setObjectPhase(index, phase)   

    def setVisObjectPrimitive(self, address, *args):

        if len(args) == 1:
            
            primitive = args[0]
            
            self.visualization.setObjectPrimitives(primitive)
            
        elif len(args) == 2:
            
            index = args[0]
            primitive = args[1]
            
            self.visualization.setObjectPrimitive(index, primitive)

    def setVisObjectSize(self, address, *args):

        if len(args) == 3:
            
            size = np.array(args)
            
            self.visualization.setObjectSizes(size)
            
        elif len(args) == 4:
            
            index = args[0]
            size = np.array(args[1:])
            
            self.visualization.setObjectSize(index, size)

    def setVisObjectRounding(self, address, *args):
        
        if len(args) == 1:
            
            round = args[0]
            
            self.visualization.setObjectRoundings(round)
            
        elif len(args) == 2:
            
            index = args[0]
            round = args[1]
            
            self.visualization.setObjectRounding(index, round)

    def setVisObjectSmoothing(self, address, *args):
        
        if len(args) == 1:
            
            smooth = args[0]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setObjectSmoothings(smooth)
            
        elif len(args) == 2:
            
            index = args[0]
            smooth = args[1]
            smooth = max(smooth, 0.0001)
            
            self.visualization.setObjectSmoothing(index, smooth)

    def setVisObjectPosition(self, address, *args):
        
        index = args[0]
        position = np.array(args[1:])

        self.visualization.setObjectPosition(index, position)

    def setVisObjectRotation(self, address, *args):
        
        index = args[0]
        rotation = np.array(args[1:])

        self.visualization.setObjectRotation(index, rotation)

    def setVisJointEdgeSmoothing(self, address, *args):
        
        smooth = args[0]
        smooth = max(smooth, 0.0001)

        self.visualization.setJointEdgeSmoothing(smooth)
        
    def setVisSkelObjectSmoothing(self, address, *args):
    
        smooth = args[0]
        smooth = max(smooth, 0.0001)
        
        self.visualization.setSkelObjectSmoothing(smooth)