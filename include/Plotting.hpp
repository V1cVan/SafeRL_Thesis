#ifndef SIM_PLOTTING
#define SIM_PLOTTING

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace Plotting{
    struct Utils{
        
        /* Transforms the given points (3xP) by scaling them by S, rotating them by the
         * angles supplied in A (yaw, pitch and roll ; following the Tait-Bryan convention)
         * and translating them by C.
         */
        static inline void transformPoints(const Eigen::Ref<const Eigen::Matrix3Xd>& points, Eigen::Ref<Eigen::Matrix3Xd> out, const Eigen::Vector3d& C = {0,0,0}, const Eigen::Vector3d& S = {1,1,1}, const Eigen::Vector3d& A = {0,0,0}){
            Eigen::Affine3d t(
                Eigen::Translation3d(C) *                           // Last: Translation
                Eigen::AngleAxisd(A[0], Eigen::Vector3d::UnitZ()) *  // Fourth: yaw rotation
                Eigen::AngleAxisd(A[1], Eigen::Vector3d::UnitY()) *  // Third: pitch rotation
                Eigen::AngleAxisd(A[2], Eigen::Vector3d::UnitX()) *  // Second: roll rotation
                Eigen::Scaling(S)                                   // First: Scaling
            );
            out = t * points;
        }

    };
}

#endif