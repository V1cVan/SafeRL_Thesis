#ifndef SIM_UTILS
#define SIM_UTILS

#include <random>
#include <array>
#include <vector>
#include <cassert>
#include <algorithm>

#ifndef COMPAT
#include <optional>
#define STATIC_INLINE static inline
#else
#include <experimental/optional>
#define optional experimental::optional
#define nullopt experimental::nullopt
#define STATIC_INLINE static
#endif

struct Utils{
    #ifdef COMPAT
    static std::mt19937 rng;
    #else
    static inline std::mt19937 rng{std::random_device{}()};
    #endif
    static constexpr double PI = 3.14159265358979323846;

    template<class Scalar>
    static inline int sign(Scalar d){
        return std::signbit(d) ? -1 : 1;
    }

    template<class State, class Sys>
    static inline State integrateRK4(const Sys& sys, const State& x, const double dt){
        // Integrate the given system from the given state for a time step dt.
        State k1 = dt*sys(x);
        State k2 = dt*sys(x+k1/2);
        State k3 = dt*sys(x+k2/2);
        State k4 = dt*sys(x+k3);
        return x+(k1+2*k2+2*k3+k4)/6;
    }

    template <class Op, class ItOut, class ... ItInputs>
    static inline void transform(Op f, ItOut outStart, ItOut outEnd, ItInputs... inputs)
    {
        while(outStart != outEnd){
            *outStart++ = f(*inputs++...);
        }
    }

    template<class C1, class C2, class Op>
    static inline C1 ebop(const C1& lhs, const C2& rhs, const Op& op){// Elementwise binary operation on values of two containers
        assert(rhs.size()==lhs.size());
        C1 result = C1();
        std::transform(lhs.begin(),lhs.end(),rhs.begin(),result.begin(),op);
        return result;
    }

    template<class C, class Op>
    static inline C euop(const C& container, const Op& op){// Elementwise unary operation on values of a container
        C result = C();
        std::transform(container.begin(),container.end(),result.begin(),op);
        return result;
    }
};

#ifdef COMPAT
std::mt19937 Utils::rng = std::mt19937(std::random_device{}());
#endif

#endif