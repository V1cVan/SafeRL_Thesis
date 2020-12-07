#ifndef SIM_CONTROLLERS
#define SIM_CONTROLLERS

class PID{
    public:
        const double Kp, Ki, Kd;// Proportional, integral and derivative gains
        double ep, ei;// Previous error, integrated error

        PID(const double Kp_ = 0, const double Ki_ = 0, const double Kd_ = 0)
        : Kp(Kp_), Ki(Ki_), Kd(Kd_), ep(0), ei(0){}

        inline double step(const double dt, const double e){
            ei += (e+ep)*dt/2;
            double y = Kp*e + Ki*ei + Kd*(e-ep)/dt;
            ep = e;
            return y;
        }

};

#endif