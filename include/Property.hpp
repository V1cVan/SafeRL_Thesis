#ifndef SIM_PROPERTY
#define SIM_PROPERTY

#include "Utils.hpp"
#include "Transition.hpp"
#include <vector>
#include <utility>
#ifndef NDEBUG
#include <iostream>
#endif

class Property{
    private:
        static constexpr double DEFAULT_XRES = 1e-4;
        static constexpr double DEFAULT_VRES = 1e-4;
        using config_t = std::pair<std::vector<Transition>,double>;

    public:
        std::vector<Transition> transitions;
        double C;

        Property(const std::vector<Transition>& transitions_, const double C_ = 0)
        : transitions(transitions_), C(C_){}

        Property(const int N,double P[])
        : transitions(), C(0){
            // Construct from mat-file array
            transitions.reserve(N);
            for(int i=0;i<N;i++){
                transitions.push_back(Transition(static_cast<int>(P[i+2*N]),P[i],P[i+N],P[i+3*N],P[i+4*N]));
            }
        }

        Property(const std::vector<Property>& properties, const std::vector<double>& offsets, const std::vector<double>& weights, const double C_ = 0)
        : transitions(), C(C_){
            // Combination constructor:
            //  The resulting property P satisfies P(x)=sum_i{w_i*P_i(x-d_i)} for all x
            auto itProperties = properties.begin();
            auto itOffsets = offsets.begin();
            auto itWeights = weights.begin();
            for(;itProperties!=properties.end();++itProperties,++itOffsets,++itWeights){
                const Property P = *itProperties;
                transitions.reserve(transitions.size()+P.transitions.size());// TODO: move outside loop and reserve enough space beforehand
                C += P.C;
                for(const Transition& T : P.transitions){
                    transitions.push_back(Transition(T.getType(),T.from+*itOffsets,T.to+*itOffsets,T.before**itWeights,T.after**itWeights));
                }
            }
        }

        inline void evaluate(const double x, double& v, double& dv) const{
            // Inputs:
            //  size of all arrays
            //  x:  Values for which the transition should be evaluated
            // Outputs:
            //  v:  Transition values for the given x
            //  dv: Derivative of the transition values for the given x
            v = C;
            dv = 0;
            for(int t=0;t<transitions.size();t++){
                transitions[t].evaluate(x,v,dv);
            }
        }

        inline void simplify(const double xRes = DEFAULT_XRES, const double vRes = DEFAULT_VRES){
            // First merge the fully overlapping transitions
            for(auto it=transitions.begin();it!=transitions.end();++it){
                const Transition T = *it;
                for(auto refIt=transitions.begin();refIt!=it;++refIt){
                    Transition Tref = *refIt;
                    if(Tref.getType()==T.getType() && abs(Tref.from-T.from)<xRes && abs(Tref.to-T.to)<xRes){
                        Tref.before += T.before;
                        Tref.after += T.after;
                        it = transitions.erase(it);
                        break;
                    }
                }
            }
            // Next remove all constant transitions:
            for(auto it=transitions.begin();it!=transitions.end();++it){
                const Transition T = *it;
                if(abs(T.before-T.after)<vRes){
                    C += T.before;
                    it = transitions.erase(it);
                }
            }
        }

        #ifndef NDEBUG
        inline void dump() const{
            std::cout << "Property with constant " << C << " and transitions [" << std::endl;
            for(const Transition& trans : transitions){
                std::cout << "[type: " << trans.getType() << " ; x_range: " << trans.from << " -> " << trans.to;
                std::cout << " ; v_range: " << trans.before << " -> " << trans.after << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }
        #endif
};
#endif