cmake_minimum_required(VERSION 3.16.3)
# Version 3.16.3 is required to fix these issues on Windows: https://gitlab.kitware.com/cmake/cmake/issues/19382

# Find Matlab package:
find_package(Matlab REQUIRED COMPONENTS MAT_LIBRARY)

# Add hwsim mex targets:
matlab_add_mex(NAME mexRoad MODULE SRC mexRoad.cpp LINK_TO hwsim Clothoids ${Matlab_MAT_LIBRARY} R2018a)
matlab_add_mex(NAME mexSim MODULE SRC mexSim.cpp LINK_TO hwsim Clothoids ${Matlab_MAT_LIBRARY} R2018a)

# Install Matlab library:
install(TARGETS mexRoad mexSim DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/+hwsim)
install(DIRECTORY +hwsim/ DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/+hwsim)