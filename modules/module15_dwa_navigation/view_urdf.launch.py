<?xml version="1.0"?>
<robot name="servoAdapterV1">

  <!-- ============================= -->
  <!-- ROOT LINK (required for TF)   -->
  <!-- ============================= -->
  <link name="world"/>

  <!-- ============================= -->
  <!-- MAIN LINK (your adapter)      -->
  <!-- ============================= -->
  <link name="base_link">

    <!-- Visual -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="/home/harsh/Downloads/meshes/servoAdapterV1.stl"
              scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.82 0.82 0.82 1.0"/>
      </material>
    </visual>

    <!-- Collision -->
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="/home/harsh/Downloads/meshes/servoAdapterV1.stl"
              scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>

    <!-- Inertial -->
    <inertial>
      <origin xyz="0.042175 0.015245 0.002742" rpy="0 0 0"/>
      <mass value="0.008746"/>
      <inertia
        ixx="6.99e-7" ixy="0.0" ixz="0.0"
        iyy="5.21e-6" iyz="0.0"
        izz="5.86e-6"/>
    </inertial>

  </link>

  <!-- ============================= -->
  <!-- JOINT (CRITICAL FOR TF)       -->
  <!-- ============================= -->
  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

</robot>
