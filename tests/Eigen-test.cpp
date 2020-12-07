#include "Eigen/Core"
#include <iostream>

struct State : public Eigen::Matrix<double,12,1>{
    static constexpr unsigned int SIZE = 12;
    using data_t = Eigen::Matrix<double,SIZE,1>;

    Eigen::Ref<Eigen::Vector3d> pos;// Same problem using Eigen::VectorBlock<data_t,3>
    Eigen::Ref<Eigen::Vector3d> ang;
    Eigen::Ref<Eigen::Vector3d> vel;
    Eigen::Ref<Eigen::Vector3d> ang_vel;

    State(const Eigen::Vector3d& pos, const Eigen::Vector3d& ang, const Eigen::Vector3d& vel, const Eigen::Vector3d& ang_vel)
    : State(){
        this->pos = pos;
        this->ang = ang;
        this->vel = vel;
        this->ang_vel = ang_vel;
    }

    State() : State(data_t::Zero()){}

    State(const State&) = delete;
    // State& operator=(const State&) = delete;
    State(State&& other) : data_t(std::move(other)), pos(this->segment<3>(0)), ang(this->segment<3>(3)), vel(this->segment<3>(6)), ang_vel(this->segment<3>(9)){
        // Default move constructor causes dangling references in debug mode.
    }
    // State& operator=(State&&) = delete;

    // This constructor allows you to construct State from Eigen expressions
    template<typename OtherDerived>
    State(const Eigen::MatrixBase<OtherDerived>& other)
    : data_t(other), pos(this->segment<3>(0)), ang(this->segment<3>(3)), vel(this->segment<3>(6)), ang_vel(this->segment<3>(9)){}

    // This method allows you to assign Eigen expressions to State
    template<typename OtherDerived>
    State& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
        this->data_t::operator=(other);
        return *this;
    }
};

State createState(){
    State x;
    x.pos = Eigen::Vector3d(1.0,1.0,1.0);
    return x;
}

struct BaseA{
    int a;

    template<int O>
    int off() const{
        return a+O;
    }
};

struct BaseB{
    int b;
};

template<typename T>
struct Derived : public T, public BaseB{
    Derived() : BaseA{5}, BaseB{this->template off<5>()}{} // template keyword is required for GCC
};

int main(){
    State x = createState();
    std::cout << x.pos << std::endl;
    State dx;
    dx.pos = x.pos/2;
    std::cout << dx.pos << std::endl;

    Derived<BaseA> d;
    std::cout << d.a << " ; " << d.b << std::endl;
    return 0;
}