import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import google.protobuf.descriptor
if not hasattr(google.protobuf.descriptor.FieldDescriptor, 'label'):
    google.protobuf.descriptor.FieldDescriptor.label = property(lambda self: getattr(self, '_label', 1))
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
print("Mediapipe Pose loaded successfully!")
