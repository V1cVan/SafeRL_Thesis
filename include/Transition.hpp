#ifndef SIM_TRANSITION
#define SIM_TRANSITION

#include "Utils.hpp"
#include <functional>
#include <math.h>

class Transition {
    private:
        std::function<void(double&,double&)> smoother;
        int type;

    public:
        double from, to, before, after;

        Transition(int type_, double from_, double to_, double before_, double after_)
        : smoother(getSmoother(type_)), type(type_), from(from_<=to_ ? from_ : to_), to(from_<=to_ ? to_ : from_), before(from_<=to_ ? before_ : after_), after(from_<=to_ ? after_ : before_){}

        // Transition(const Transition& T, double dx, double w)
        // : Transition(T.type,T.from+dx,T.to+dx,T.before*w,T.after*w){
        //     // Modify constructor:
        //     //  The new transition will be translated (from_ and to_) over dx and multiplied
        //     //  (before_ and after_) by w
        // }

        // Transition(const Transition& T)
        // : type(T.type), from(T.from), to(T.to), before(T.before), after(T.after), smoother(getSmoother(T.type)){
        //     // Copy constructor
        // }

        inline void evaluate(const double x, double& v, double& dv) const{
            // Inputs:
            //  x:  Value for which the transition should be evaluated
            // Outputs:
            //  v:  Transition value for the given x
            //  dv: Derivative of the transition value for the given x
            if (x<=from){
                v += before;
            }else if(x>to){
                v += after;
            }else{
                double u = (x-from)/(to-from);// x_rel
                double du = 1/(to-from);// dx_rel
                smoother(u,du);
                v += before + u*(after-before);
                dv += du*(after-before);
            }
        }

        inline int getType() const{
            return type;
        }

        inline void changeType(const int newType){
            type = newType;
            smoother = getSmoother(newType);
        }

    private:
        static inline std::function<void(double&,double&)> getSmoother(const int type){
            if(type==0){
                return Transition::smooth_heaviside;
            }else if(type==1){
                return Transition::smooth_linear;
            }else if(type==2){
                return Transition::smooth_quadratic;
            }else if(type==3){
                return Transition::smooth_cosine;
            }else{
                return Transition::smooth_linear;// Default to linear smoothing
            }
        }

        static inline void smooth_heaviside(double& u, double& du){
            du = 0;
            u = (u<=0.5) ? 0 : 1;
        }

        static inline void smooth_linear(double& u,double& du){}

        static inline void smooth_quadratic(double& u, double& du){
            if(u<=0.5){
                du *= 4*u;
                u = 2*u*u;
            }else{
                du *= 4*(1-u);
                u = 1-2*(1-u)*(1-u);
            }
        }

        static inline void smooth_cosine(double& u, double& du){
            du *= Utils::PI*sin(u*Utils::PI)/2;
            u = (1-cos(u*Utils::PI))/2;
        }
};
#endif