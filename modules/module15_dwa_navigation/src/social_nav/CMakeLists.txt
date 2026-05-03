cmake_minimum_required(VERSION 3.8)
project(social_nav)

find_package(ament_cmake REQUIRED)

# Install Python scripts
install(
  PROGRAMS
    scripts/human_tracker_node.py
    scripts/social_costmap_node.py
    scripts/social_override_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install world files (includes walk.dae animation mesh)
install(
  DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# Install walk.dae from package root if present
install(
  FILES walk.dae
  DESTINATION share/${PROJECT_NAME}/worlds
  OPTIONAL
)
# Install map files
install(
  DIRECTORY maps/
  DESTINATION share/${PROJECT_NAME}/maps
)

# Install rviz config
install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
