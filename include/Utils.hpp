#ifndef SIM_UTILS
#define SIM_UTILS

#include <random>
#include <array>
#include <vector>
#include <cassert>
#include <algorithm>
#include <map>
#include <functional>
#include <memory>
#include <exception>
#include <sstream>
#include <cstddef>

#if __has_include(<optional>)
#include <optional>
#elif __has_include(<experimental/optional>)
#include <experimental/optional>
namespace std{
    // Expose std::optional and std::nullopt to the std namespace:
    template<typename T>
    using optional = experimental::optional<T>;
    constexpr experimental::nullopt_t nullopt = experimental::nullopt;
}
#else
#error The hwsim library requires the optional type, either from <optional> or from <experimental/optional>.
#endif

#ifndef COMPAT
#define STATIC_INLINE static inline
#else
namespace std{
    // Define byte
    using byte = unsigned char;
    //enum class byte : unsigned char{};
    // Add clamp function (taken from cppreference.com):
    template<class T>
    constexpr const T& clamp( const T& v, const T& lo, const T& hi )
    {
        assert( !(hi < lo) );
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }
    // Define as_const (taken from recent GCC source):
    template<typename _Tp>
    constexpr add_const_t<_Tp>& as_const(_Tp& __t) noexcept { return __t; }

    template<typename _Tp>
    void as_const(const _Tp&&) = delete;
}
#define STATIC_INLINE static
#endif

template<class R>
struct RNG : public R{
    // Wrapper class for a random number generator which stores the latest seed value.
    public:
        using rng_t = R;
    private:
        typename R::result_type lastSeed;

    public:
        RNG(typename R::result_type seed) : R(seed), lastSeed(seed){}

        RNG() : RNG(std::random_device{}()){}

        inline typename R::result_type getSeed(){
            return lastSeed;
        }

        inline void seed(typename R::result_type value = R::default_seed){
            lastSeed = value;
            R::seed(value);
        }
};

struct Utils{
    #ifdef COMPAT
    static RNG<std::mt19937> rng;
    #else
    static inline RNG<std::mt19937> rng{};
    #endif
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double EPS = 1e-6;
    using sdata_t = std::vector<std::byte>;

    template<class Scalar>
    static inline int sign(Scalar d){
        if(std::abs(d)<EPS){
            return 0;
        }else{
            return std::signbit(d) ? -1 : 1;
        }
    }

    static inline double wrapAngle(double alpha){
        // Wraps any given angle (in radians) to the interval [-PI,PI]
        return std::remainder(alpha,2*PI);
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

    // Default serializer (to existing data vector)
    template<class C>
    static inline void serialize(const C& obj, sdata_t& data){
        const std::byte* bytes = reinterpret_cast<const std::byte*>(&obj);
        data.insert(data.end(),bytes,bytes+sizeof(C));
    }

    // Default serializer (to new data vector)
    template<class C>
    static inline sdata_t serialize(const C& obj){
        sdata_t data;
        serialize(obj, data);
        return data;
    }

    // Default deserializer
    template<class C>
    static inline C deserialize(const sdata_t& data){
        const std::byte* bytes = data.data();
        const C* obj = reinterpret_cast<const C*>(bytes);
        return C(*obj);// Create copy
    }

    // template<class To>
    // struct cast_from_byte{
    //     To* operator()(std::byte* data){ return reinterpret_cast<To*>(data); }
    // };

    // template<class From>
    // struct cast_to_byte{
    //     std::byte* operator()(From* data){ return reinterpret_cast<std::byte*>(data); }
    // };
};

#ifdef COMPAT
RNG<std::mt19937> Utils::rng = RNG<std::mt19937>();
#endif

namespace hwsim{
    struct invalid_state : public std::runtime_error{
        invalid_state(const std::string& what_arg) : std::runtime_error(what_arg){}
    };
}
// struct NonCopyable{
//     NonCopyable(){}
//     // Disable copy constructor and assignment
//     NonCopyable(const NonCopyable&) = delete;
//     NonCopyable& operator=(const NonCopyable&) = delete;
//     // Set move consturctor and assignment to default
//     NonCopyable(NonCopyable&&) = default;
//     NonCopyable& operator=(NonCopyable&&) = default;
// };

// struct NonMovable{
//     NonMovable(){}
//     // Disable move constructor and assignment
//     NonMovable(NonMovable&&) = delete;
//     NonMovable& operator=(NonMovable&&) = delete;
// };

struct fixedBase{// Fixed base class
    fixedBase(){}
    // Disable copying and moving:
    fixedBase(const fixedBase&) = delete;
    fixedBase(fixedBase&&) = delete;
    fixedBase& operator=(const fixedBase&) = delete;
    fixedBase& operator=(fixedBase&&) = delete;
};

// The following classes implement a factory with self registering classes
// to easily serialize and recreate those classes (e.g. from a saved file).
// Factory design is based on https://www.bfilipek.com/2018/02/factory-selfregister.html
struct BaseFactory{
    using id_t = unsigned int;
    using data_t = Utils::sdata_t;

    // For now we use a vector of bytes to serialize objects.
    // This looks very promising but will have to wait for C++23:
    // https://github.com/Lyberta/cpp-io-impl

    struct BluePrint{
        unsigned int id;
        data_t args;
    };
};

template<class T>
class Factory : public BaseFactory{
    private:
        using CreateFunc = std::function<std::unique_ptr<T>(data_t)>;
        std::map<id_t,CreateFunc> createFunctions;
        std::map<id_t,size_t> serializedLengths;
        const std::string name;

    public:
        Factory(const std::string name = "FACTORY_NAME")
        : createFunctions(), serializedLengths(), name(name){}

        template<class S>
        inline id_t record(const id_t id, const size_t N){
            // This method registers a subclass S (of T) such that it can be created
            // through its id.
            assert(createFunctions.count(id)==0);
            createFunctions.insert({id,[](data_t args) -> std::unique_ptr<T>{return std::make_unique<S>(args);}});
            serializedLengths[id] = N;
            return id;
        }

        inline std::unique_ptr<T> create(const BluePrint& bp) const{
            verifyID(bp.id);
            if(bp.args.size()!=getSerializedLength(bp.id)){
                throw std::invalid_argument("The given blueprint args do not have the required size.");
            }else{
                return createFunctions.at(bp.id)(bp.args);
            }
        }

        inline size_t getSerializedLength(const id_t id) const{
            verifyID(id);
            return serializedLengths.at(id);
        }
    
    private:
        inline void verifyID(const id_t id) const{
            if(createFunctions.count(id)==0){
                std::ostringstream err;
                err << "Unknown " << name << " type: " << id << std::endl;
                err << "Allowed " << name << " types: ";
                for(auto it = createFunctions.begin(); it!=createFunctions.end(); ++it){
                    if(it!=createFunctions.begin()){
                        err << ",";
                    }
                    err << it->first;
                }
                throw std::invalid_argument(err.str());
            }
        }
};

struct ISerializable{
    virtual typename BaseFactory::BluePrint blueprint() const = 0;
    virtual void serialize(BaseFactory::data_t&) const = 0;
};

template<class T, Factory<T>& F, class S, unsigned int I, unsigned int N = 0>
struct Serializable : public T{// T inherits from ISerializable
    // F is the factory in which S should be recorded and from whose BluePrints a new
    // S can be constructed through an appropriate constructor in S (i.e. S(const data_t args)).
    // S is the Serializable class, which will automatically get registered in the factory with
    // the given ID. A default serialize function is also added which can be overriden.
    #ifdef COMPAT
    static const BaseFactory::id_t ID;
    #else
    static inline const BaseFactory::id_t ID = F.template record<S>(I,N);
    #endif
    using sid_t = BaseFactory::id_t;
    using sdata_t = BaseFactory::data_t;
    using Base = Serializable<T,F,S,I,N>;
    static constexpr size_t SERIALIZED_LENGTH = N;

    template<class... Args>
    Serializable(Args&&... args) : T(std::forward<Args>(args)...){
        // Forward constructor arguments to Base constructor (T)
        ID; // Prevent ID (and hence the class registration) from being removed by compiler optimizations
    }

    // Serializable(){
    //     ID; // Prevent ID (and hence the class registration) from being removed by compiler optimizations
    // }

    inline typename BaseFactory::BluePrint blueprint() const{
        sdata_t data;
        data.reserve(SERIALIZED_LENGTH);
        serialize(data);
        assert(data.size()==SERIALIZED_LENGTH);
        return {ID,data};
    }

    // Default serializer, leaving the data vector empty.
    virtual void serialize(sdata_t&) const{}
};

#ifdef COMPAT
template<class T, Factory<T>& F, class S, unsigned int I, unsigned int N>
const BaseFactory::id_t Serializable<T,F,S,I,N>::ID = F.template record<S>(I,N);
#endif

#endif