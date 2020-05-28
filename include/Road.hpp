#ifndef SIM_ROAD
#define SIM_ROAD

#ifndef NOMINMAX
#define NOMINMAX // Prevent interference of windows min & max macros with std::min and std::max functions
#endif

#include "Utils.hpp"
#include "ClothoidList.hh"
#include "Property.hpp"
#include <array>
#include <vector>
#include <set>
#include <utility>
#include <algorithm>
#include <cmath>

class Road{
    private:
        static constexpr double EPS = 1e-6;

    public:
        using id_t = unsigned int;

        enum class Side{
            RIGHT = -1,
            LEFT  = 1
        };

        struct Lane{
            enum class Direction{
                NEG = -1, // Vehicles drive in direction of decreasing s
                POS = 1 // Vehicles drive in direction of increasing s
            };

            Direction direction; // Direction of travel on this lane
            std::array<double,2> validity; // Validity bounds of this lane (s coordinates from and to) ; always defined in positive road direction (i.e. from<to)
            Property offsetProp; // Lateral offset of this lane's center w.r.t. the road's outline
            Property widthProp; // Width of this lane, centered around the lane's center
            Property heightProp; // Height of this lane
            Property superelevationProp; // Superelevation of the lane (related to angle the lane makes around its longitudinal axis)
            Property speedProp; // Maximum speed limit on this lane
            Property leftProp; // Binary property, a value of 1 denotes the possibility to cross towards the neighbouring lane towards the LEFT/RIGHT SIDE OF THE ROAD (i.e. the right/left side for a lane
            Property rightProp; // in negative direction), 0 otherwise. Note that this only denotes a possibility, the laneBoundary function will determine if it is also physically possible.
            std::optional<std::pair<id_t,id_t>> from;// Only set if there is a connection from another lane. (Road id, lane id)
            std::optional<std::pair<id_t,id_t>> to;// Only set if there is a connection to another lane. (Road id, lane id)
            std::optional<id_t> merge;// Only set if this lane merges with another lane. (Lane id)

            inline bool isValid(const double s) const{
                return validity[0]<=s && s<=validity[1];
            }

            inline void offset(const double s, double& c, double& dc) const{
                if(isValid(s)){
                    offsetProp.evaluate(s,c,dc);
                }else{
                    c = std::nan("");
                    dc = std::nan("");
                }
            }

            inline double offset(const double s) const{
                double c, dc;
                offset(s,c,dc);
                return c;
            }

            inline void width(const double s, double& w, double& dw) const{
                if(isValid(s)){
                    widthProp.evaluate(s,w,dw);
                }else{
                    w = std::nan("");
                    dw = std::nan("");
                }
            }

            inline double width(const double s) const{
                double w, dw;
                width(s,w,dw);
                return w;
            }

            inline void height(const double s, double& h, double& dh) const{
                if(isValid(s)){
                    heightProp.evaluate(s,h,dh);
                }else{
                    h = std::nan("");
                    dh = std::nan("");
                }
            }

            inline double height(const double s) const{
                double h, dh;
                height(s,h,dh);
                return h;
            }

            inline void superElevation(const double s, double& e, double& de) const{
                if(isValid(s)){
                    superelevationProp.evaluate(s,e,de);
                }else{
                    e = std::nan("");
                    de = std::nan("");
                }
            }

            inline double superElevation(const double s) const{
                double e, de;
                superElevation(s,e,de);
                return e;
            }

            inline void speed(const double s, double& v, double& dv) const{
                if(isValid(s)){
                    speedProp.evaluate(s,v,dv);
                }else{
                    v = std::nan("");
                    dv = std::nan("");
                }
            }

            inline double speed(const double s) const{
                double v, dv;
                speed(s,v,dv);
                return v;
            }

            inline void left(const double s, double& l, double& dl) const{
                if(isValid(s)){
                    leftProp.evaluate(s,l,dl);
                }else{
                    l = std::nan("");
                    dl = std::nan("");
                }
            }

            inline void right(const double s, double& r, double& dr) const{
                if(isValid(s)){
                    rightProp.evaluate(s,r,dr);
                }else{
                    r = std::nan("");
                    dr = std::nan("");
                }
            }

            inline void availability(const double s, const Side d, double& a, double& da) const{
                if(d==Side::LEFT){
                    left(s,a,da);
                }else{
                    right(s,a,da);
                }
            }

            inline double availability(const double s, const Side d) const{
                double a,da;
                availability(s,d,a,da);
                return a;
            }

            inline double start() const{
                int i = (1-static_cast<int>(direction))/2;
                return validity[i];
            }

            inline double end() const{
                int i = (1+static_cast<int>(direction))/2;
                return validity[i];
            }
        };

        enum class BoundaryCrossability{
            NONE = 0, // Boundary cannot be crossed at all
            FROM = 1, // Boundary can only be crossed from the current lane towards the neighbouring lane
            TO   = 2, // Boundary can only be crossed from the neighbouring lane towards the current lane
            BOTH = 3  // Boundary can be crossed from the current lane towards the neighbouring lane and vice versa
        };

        G2lib::ClothoidList outline;
        std::vector<Lane> lanes;
        double length;

        Road(const G2lib::ClothoidList& outline_, const std::vector<Lane>& lanes_)
        : outline(outline_), lanes(lanes_), length(outline_.length()){
            #ifndef NDEBUG
            // std::cout << "Created road with outline through points: [" << std::endl;
            // int N = outline.numSegment();
            // std::vector<double> X(N+1),Y(N+1);
            // outline.getXY(X.data(),Y.data());
            // for(int i=0;i<=N;i++){
            //     std::cout << "[" << X[i] << "," << Y[i] << "]" << std::endl;
            // }
            // std::cout << "]" << std::endl;
            // std::cout << "and lane offset properties: [" << std::endl;
            // for(const Lane& lane : lanes){
            //     std::cout << "[";
            //     lane.offsetProp.dump();
            //     std::cout << "]" << std::endl;
            // }
            #endif
        }

        inline std::optional<id_t> laneId(const double s, const double l, const bool exact = true) const{
            // Possibly returns std::nullopt in case there is no exact lane match and exact==true.
            // Otherwise always returns the closest matching lane.
            double c, w;
            double opt_cost = -1, cost;
            int id = -1;
            for(id_t L=0;L<lanes.size();L++){
                c = lanes[L].offset(s);// nan if lane is invalid at s
                w = lanes[L].width(s);
                cost = std::abs(c-l);// Might add a cost to prefer lanes that are longer valid
                if(lanes[L].isValid(s) && (std::abs(c-l)<=w/2 || !exact) && (cost<opt_cost || opt_cost<0)){
                    opt_cost = cost;
                    id = L;
                }
            }
            if(id>=0){
                return static_cast<id_t>(id);
            }else{
                return std::nullopt;
            }
        }

        inline double curvature(const double s, const double l) const{
            double kappa = outline.kappa(s);
            return kappa/(1-l*kappa);
        }

        inline double heading(const double s) const{
            return outline.theta(s);
        }

        inline double heading(const double s, const double l) const{
            double psi = outline.theta(s);
            const std::optional<id_t> L = laneId(s,l);
            if(L){
                double c, dc;
                lanes[*L].offset(s,c,dc);
                return psi+std::atan2(dc,1-l*curvature(s,0))+(1-static_cast<int>(lanes[*L].direction))*Utils::PI/2;
            }else{
                return std::nan("");
            }
        }

        inline std::optional<id_t> laneNeighbour(const double s, const id_t L, const Side d) const{
            // Get the first valid lane towards the left/right of the given lane. This method
            // takes the lane direction into account. At a point s' where the neighbour changes,
            // the NEW neighbour is returned (in the direction of the lane).
            if(!lanes[L].isValid(s)){
                return std::nullopt;
            }
            bool hasNeighbour = false;
            const int dir = static_cast<int>(lanes[L].direction);
            const int delta = static_cast<int>(d)*dir;
            int N = L+delta;
            while(!hasNeighbour && N>=0 && N<lanes.size()){
                hasNeighbour = lanes[N].isValid(s) && s!=lanes[N].validity[(1+dir)/2];// TODO: maybe include lane end if it coincides with road end?
                N += delta;
            }
            if(hasNeighbour){
                return static_cast<id_t>(N-delta);
            }else{
                return std::nullopt;
            }
        }

        inline std::pair<std::optional<BoundaryCrossability>,std::optional<id_t>> laneBoundary(const double s, const id_t L, const Side d) const{
            // Get the boundary crossability with the first valid lane towards the left/right
            // of the given lane. This method takes the lane direction into account. At a point
            // s' where the crossability changes, the NEW crossability is returned (in the
            // direction of the lane).
            if(!lanes[L].isValid(s)){
                return {std::nullopt,std::nullopt};// Lane is invalid for the given s => no boundary
            }
            std::optional<id_t> neighbour = laneNeighbour(s,L,d);
            if(!neighbour){
                return {BoundaryCrossability::NONE,std::nullopt};// No neighbour, so uncrossable
            }
            const id_t N = *neighbour;
            if(lanes[L].direction!=lanes[N].direction){
                return {BoundaryCrossability::NONE,N};// Traffic in opposite direction on neighbouring lane
            }
            const int dir = static_cast<int>(lanes[L].direction);
            const int delta = static_cast<int>(d)*dir;
            double cN,cL,dcN,dcL,wN,wL,dwN,dwL,hN,hL,dhN,dhL;
            lanes[N].offset(s,cN,dcN); lanes[N].width(s,wN,dwN); lanes[N].height(s,hN,dhN);
            lanes[L].offset(s,cL,dcL); lanes[L].width(s,wL,dwL); lanes[L].height(s,hL,dhL);
            // Calculate total (squared) gap as the sum of the squared lateral gap and the squared height gap
            const double latGap = std::max(0.0,(delta*cN-wN/2)-(delta*cL+wL/2));
            const double vertGap = hN-hL;
            const double gap = latGap*latGap + vertGap*vertGap;
            const double dgap = 2*latGap*(delta*dcN-dwN/2-delta*dcL-dwL/2) + 2*vertGap*(dhN-dhL);
            if(gap>EPS*EPS || dgap*dir>EPS){// There is a gap OR the gap is increasing in forward direction of the lane
                return {BoundaryCrossability::NONE,N};// Physically impossible to cross towards the neighbouring lane
            }
            double aN,aL,daN,daL;
            lanes[L].availability(s,static_cast<Side>(delta),aL,daL);
            lanes[N].availability(s,static_cast<Side>(-delta),aN,daN);
            if(daN!=0){// If we are at a position where the availability changes,
                aN = (dir*daN>0) ? 1 : 0;// put it equal to the value right after the change (in direction of the lane)
            }
            if(daL!=0){// If we are at a position where the availability changes,
                aL = (dir*daL>0) ? 1 : 0;// put it equal to the value right after the change (in direction of the lane)
            }
            int cross = static_cast<int>(aL) + 2*static_cast<int>(aN);// Determine crossability based on left and right availability properties
            return {static_cast<BoundaryCrossability>(cross),N};
        }

        inline std::optional<id_t> roadBoundary(const double s, const id_t L, const Side d) const{
            // Get the first non-crossable road boundary from the given lane and in the given
            // direction. This method loops over the lane's neighbours until a boundary with
            // crossability NONE or TO is encountered. This method takes the lane direction
            // into account.
            id_t Lb = L;
            std::pair<std::optional<BoundaryCrossability>,std::optional<id_t>> boundary = laneBoundary(s,Lb,d);
            if(!boundary.first){
                return std::nullopt;// Lane is invalid for the given s => no boundary
            }
            while(boundary.second && (*boundary.first!=BoundaryCrossability::NONE && *boundary.first!=BoundaryCrossability::TO)){
                Lb = *boundary.second;
                boundary = laneBoundary(s,Lb,d);
            }
            return Lb;
        }

        inline void globalPose(const std::array<double,3>& roadPose, std::array<double,3>& pos, std::array<double,3>& ang) const{
            // roadPose = (s,l,gamma) ; pos = (x,y,z) ; ang = (psi,theta,phi)
            double psi,kappa;
            outline.evaluate(roadPose[0],psi,kappa,pos[0],pos[1]);
            pos[0] += -roadPose[1]*std::sin(psi);
            pos[1] += roadPose[1]*std::cos(psi);
            ang[0] = heading(roadPose[0],roadPose[1])+roadPose[2];
            id_t L = *laneId(roadPose[0],roadPose[1],false);
            double h,dh,e;
            lanes[L].height(roadPose[0],h,dh);
            e = lanes[L].superElevation(roadPose[0]);
            pos[2] = h+(roadPose[1]-lanes[L].offset(roadPose[0]))*e;
            // TODO: below are approximations, not exact when psi_{\rho,L} significantly differs from psi_\rho
            // TODO: also not taking gamma into account
            ang[1] = std::atan(dh);
            ang[2] = std::atan(e);
        }

        inline std::tuple<Property,Property,double> linearLaneMapping(const double d0) const{
            // Returns two properties ML and Ms defining a linear mapping from a variable
            // d to a lane id and a curvilinear abscissa s. The variable d ranges from 0 to
            // D (which can be retrieved as the last tuple element). Outside this range, ML
            // evaluates to -1 (invalid lane id) and Ms evaluates to 0 (start of this road).
            // Within this range each value of d maps to a unique position on this road.
            std::vector<Transition> ML = std::vector<Transition>();
            std::vector<Transition> Ms = std::vector<Transition>();
            ML.reserve(1+lanes.size());
            Ms.reserve(1+2*lanes.size());
            double prev_s = 0, d = d0;
            std::array<double,2> val;
            for(int L=0;L<lanes.size();L++){
                val = lanes[L].validity;
                if(lanes[L].width(val[0])==0){
                    // If lane's width is zero at s=val[0], increase val[0] to the point where the lane is fully
                    // inserted (assuming there is a single transition responsible for the insertion)
                    double newVal = val[0];
                    for(const Transition& T : lanes[L].widthProp.transitions){
                        if(T.from==val[0]){
                            newVal = std::max(newVal,T.to);
                        }
                    }
                    val[0] = newVal;
                }
                if(lanes[L].width(val[1])==0){
                    // If lane's width is zero at s=val[1], decrease val[1] to the point where the lane is fully
                    // inserted (assuming there is a single transition responsible for the insertion)
                    double newVal = val[1];
                    for(const Transition& T : lanes[L].widthProp.transitions){
                        if(T.to==val[1]){
                            newVal = std::min(newVal,T.from);
                        }
                    }
                    val[1] = newVal;
                }
                if(lanes[L].merge){
                    // If lane merges with another lane, decrease val[1] (or increase val[0]) to the point where
                    // the lane is not yet being merged (assuming there is a single transition responsible for
                    // the merge)
                    double newVal = lanes[L].end();
                    int dir = static_cast<int>(lanes[L].direction);
                    for(const Transition& T : lanes[L].offsetProp.transitions){
                        if(dir>0 && T.to==lanes[L].end()){
                            newVal = std::min(newVal,T.from);
                        }else if(dir<0 && T.from==lanes[L].end()){
                            newVal = std::max(newVal,T.to);
                        }
                    }
                    if(dir>0){
                        val[1] = std::min(newVal,val[1]);
                    }else{
                        val[0] = std::max(newVal,val[0]);
                    }
                }
                // Add extra validity offset (to account for vehicle dimensions and prevent
                // collisions at lane connection points)
                val[0] += 5;
                val[1] -= 5;
                ML.push_back(Transition(0,d,d,0,1));// Step to next lane number
                if(val[1]>val[0]){
                    Ms.push_back(Transition(0,d,d,0,val[0]-prev_s));// Step to valid from s-value
                    Ms.push_back(Transition(1,d,d+val[1]-val[0],0,val[1]-val[0]));// Linearly go to valid to s-value
                    prev_s = val[1];
                    d += val[1]-val[0];
                }
            }
            ML.push_back(Transition(0,d,d,0,-static_cast<double>(lanes.size())));// Step back to zero (-1)
            Ms.push_back(Transition(0,d,d,0,-prev_s));// Step back to zero
            return std::tuple<Property,Property,double>(Property(ML,-1),Property(Ms),d);
        }

        inline std::set<double> principalCA() const{
            // Get the principal curvilinear abscissa of this road. These are the abscissa
            // for which the road layout might change. I.e. the begin and end of lanes and
            // the begin and end of layout property transitions.
            std::set<double> pa;
            for(const Lane& lane : lanes){
                pa.insert(lane.validity[0]);
                pa.insert(lane.validity[1]);
                auto insertProperty = [&pa,vF=lane.validity[0],vT=lane.validity[1]](const Property& prop){
                    for(const Transition& trans : prop.transitions){
                        if(trans.from>vF && trans.from<vT){
                            pa.insert(trans.from);
                        }
                        if(trans.to>vF && trans.to<vT){
                            pa.insert(trans.to);
                        }
                    }
                };
                insertProperty(lane.offsetProp);
                insertProperty(lane.widthProp);
                insertProperty(lane.heightProp);
                insertProperty(lane.leftProp);
                insertProperty(lane.rightProp);
                insertProperty(lane.superelevationProp);
            }
            return pa;
        }

};
#endif