#ifndef SIM_EIGEN
#define SIM_EIGEN

#include "Eigen/Core"

/* Following code fragments are based on:
 * https://stackoverflow.com/questions/54512878/eigen-custom-classes-and-function-parameters
 * https://stackoverflow.com/questions/54523063/extending-eigen-ref-class
 * https://stackoverflow.com/questions/62829850/eigen-named-access-to-vector-segment
 * https://stackoverflow.com/questions/6635851/real-world-use-of-x-macros
 */

// Helper macros and structures:
#define UNPACK(...) __VA_ARGS__
#define COMMA ,

namespace Eigen{
    template<class T> struct remove_ref{using type=typename std::remove_const<T>::type;};
    template<class T> struct remove_ref<Ref<T>>{using type=typename std::remove_const<T>::type;};
    template<class _Scalar> using Scalar = Array<_Scalar,1,1>;
};


/*
----- EIGEN NAMED BASE ---
Macro that defines a base structure of a custom named matrix. This
common structure is shared between actual named matrices and Eigen
references to these named matrices. See also EIGEN NAMED MATRIX and
EIGEN NAMED REF below.
*/
// Helper macros:
#define EIGEN_NAMED_BASE_REF_DECL(Type, Name, RowOff, ColOff) Eigen::Ref<UNPACK Type> Name;
#define EIGEN_NAMED_BASE_REF_SIZE(Type, Name, RowOff, ColOff) UNPACK Type::SizeAtCompileTime
#define EIGEN_NAMED_BASE_REF_INIT(Type, Name, RowOff, ColOff) Name ## (this->template block<UNPACK Type::RowsAtCompileTime,UNPACK Type::ColsAtCompileTime>(RowOff,ColOff))

// Base declaration: Note that the Core type (C) should be passed with extra brackets:
#define EIGEN_NAMED_BASE_DECL(Name, C) template<typename B=UNPACK C> struct Name ## Base : public B

// Base implementation:
#define EIGEN_NAMED_BASE_IMPL_OP(Name,Refs,OP_DECL,OP_SIZE,OP_INIT)\
/* Base (B) is either Core (C) or Eigen::Ref<[const] Core> */\
using Core = typename Eigen::remove_ref<B>::type;\
using Base = B;\
static_assert(std::is_same<Base,Core>::value || std::is_same<Base,Eigen::Ref<Core>>::value || std::is_same<Base,Eigen::Ref<const Core>>::value);\
static constexpr size_t SIZE = Core::SizeAtCompileTime;\
/* Declare references */\
Refs(OP_DECL,)\
/* Define the default, copy & move constructor */\
template<typename Derived>\
Name ## Base(const Eigen::DenseBase<Derived>& expr)\
: Base(expr), Refs(OP_INIT,COMMA){}\
Name ## Base(const Name ## Base& other)\
: Base(other), Refs(OP_INIT,COMMA){\
    /* Default copy constructor causes dangling references in debug mode. */\
}\
Name ## Base(Name ## Base&& other)\
: Base(std::move(other)), Refs(OP_INIT,COMMA){\
    /* Default move constructor causes dangling references in debug mode. */\
}

// Default base class (no custom members or methods):
#define EIGEN_NAMED_BASE_OP(Name,C,Refs,OP_DECL,OP_SIZE,OP_INIT)\
EIGEN_NAMED_BASE_DECL(Name,C){\
    EIGEN_NAMED_BASE_IMPL_OP(Name,Refs,OP_DECL,OP_SIZE,OP_INIT)\
};

// Using default operations:
#define EIGEN_NAMED_BASE_IMPL(Name,Refs) EIGEN_NAMED_BASE_IMPL_OP(Name,Refs,EIGEN_NAMED_BASE_REF_DECL,EIGEN_NAMED_BASE_REF_SIZE,EIGEN_NAMED_BASE_REF_INIT)
#define EIGEN_NAMED_BASE(Name,C,Refs) EIGEN_NAMED_BASE_OP(Name,C,Refs,EIGEN_NAMED_BASE_REF_DECL,EIGEN_NAMED_BASE_REF_SIZE,EIGEN_NAMED_BASE_REF_INIT)


/*
----- EIGEN NAMED MATRIX ---
Macro that defines the actual custom named matrix and implements all
boilerplate constructors/assignment operators required to easily work
with it in Eigen. This requires a previously defined Base for this
named matrix (see above).
*/
// Helper macros:
// TODO: maybe pass Eigen::Ref<Type> to the constructors?
#define EIGEN_NAMED_MATRIX_REF_ARG(Type, Name, RowOff, ColOff) const UNPACK Type & Name
#define EIGEN_NAMED_MATRIX_REF_ASSIGN(Type, Name, RowOff, ColOff) this-> Name = Name;

// Matrix declaration:
#define EIGEN_NAMED_MATRIX_DECL(Name) struct Name : public Name ## Base<>

// Matrix implementation:
#define EIGEN_NAMED_MATRIX_IMPL_OP(Name,Refs,OP_ARG,OP_ASSIGN)\
Name() : Name(Base::Zero()){} /* Default constructor */\
\
/* Construction from reference values */\
Name(Refs(OP_ARG,COMMA)) : Name(){\
Refs(OP_ASSIGN,)\
}\
/* Redefine implicitly deleted copy and move constructors */\
Name(const Name& other) : Name ## Base(other){}\
Name(Name&& other) : Name ## Base(std::move(other)){}\
/* This constructor allows you to construct Name from Eigen expressions */\
template<typename OtherDerived>\
Name(const Eigen::MatrixBase<OtherDerived>& other)\
: Name ## Base(other){}\
\
/* This method allows you to assign Eigen expressions to Name */\
template<typename OtherDerived>\
Name& operator=(const Eigen::MatrixBase<OtherDerived>& other)\
{\
    this->Base::operator=(other);\
    return *this;\
}\
/* This method redefines the implicitly deleted assignment operator */\
EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Name)

// Default matrix class (no custom members or methods):
#define EIGEN_NAMED_MATRIX_OP(Name,Refs,OP_ARG,OP_ASSIGN)\
EIGEN_NAMED_MATRIX_DECL(Name){\
    EIGEN_NAMED_MATRIX_IMPL_OP(Name,Refs,OP_ARG,OP_ASSIGN)\
};

// Using default operations:
#define EIGEN_NAMED_MATRIX_IMPL(Name,Refs) EIGEN_NAMED_MATRIX_IMPL_OP(Name,Refs,EIGEN_NAMED_MATRIX_REF_ARG,EIGEN_NAMED_MATRIX_REF_ASSIGN)
#define EIGEN_NAMED_MATRIX(Name,Refs) EIGEN_NAMED_MATRIX_OP(Name,Refs,EIGEN_NAMED_MATRIX_REF_ARG,EIGEN_NAMED_MATRIX_REF_ASSIGN)


/*
----- EIGEN NAMED REFERENCE ---
Macro that defines an Eigen reference to a custom named matrix and
implements all boilerplate constructors/assignment operators required
to easily work with them in Eigen. This requires a previously defined
Base and Matrix for this named matrix (see above).
*/
// Helper macros:
#define EIGEN_NAMED_REF_BASE(Name) Name ## Base<Eigen::Ref<Name ## Base<>::Core>>
#define EIGEN_NAMED_CONSTREF_BASE(Name) Name ## Base<Eigen::Ref<const Name ## Base<>::Core>>

// Default Ref class:
#define EIGEN_NAMED_REF(Name)\
namespace Eigen{\
    /* Specialization of Eigen::Ref for Name */\
    template<>\
    struct Ref<Name> : public EIGEN_NAMED_REF_BASE(Name){\
        template<typename Derived>\
        Ref(DenseBase<Derived>& expr)\
        : EIGEN_NAMED_REF_BASE(Name)(expr){}\
\
        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Ref<Name>)\
    };\
    /* Specialization of Eigen::Ref for const Name */\
    template<>\
    struct Ref<const Name> : public EIGEN_NAMED_CONSTREF_BASE(Name){\
        template<typename Derived>\
        Ref(DenseBase<Derived>& expr)\
        : EIGEN_NAMED_CONSTREF_BASE(Name)(expr){}\
    };\
};


/* Macro's that simplify the creation of a named Eigen vector: */
// Helper macros (overriding the defaults with 4 parameters)
#define EIGEN_NAMED_VEC_REF_HELP(Type, Name) uint8_t Name[EIGEN_NAMED_VEC_REF_SIZE(Type,Name)];
#define EIGEN_NAMED_VEC_REF_DECL(Type, Name) EIGEN_NAMED_BASE_REF_DECL(Type,Name,offsetof(Helper,Name),0)
#define EIGEN_NAMED_VEC_REF_SIZE(Type, Name) EIGEN_NAMED_BASE_REF_SIZE(Type,Name,offsetof(Helper,Name),0)
#define EIGEN_NAMED_VEC_REF_INIT(Type, Name) EIGEN_NAMED_BASE_REF_INIT(Type,Name,offsetof(Helper,Name),0)
#define EIGEN_NAMED_VEC_REF_ARG(Type, Name) EIGEN_NAMED_MATRIX_REF_ARG(Type,Name,offsetof(Helper,Name),0)
#define EIGEN_NAMED_VEC_REF_ASSIGN(Type, Name) EIGEN_NAMED_MATRIX_REF_ASSIGN(Type,Name,offsetof(Helper,Name),0)

#define EIGEN_NAMED_VEC(Name, Refs)\
struct Name ## Helper{\
    /* Helper structure, to get correct offsets from reference name using 'offsetof' */\
    Refs(EIGEN_NAMED_VEC_REF_HELP,)\
    static constexpr size_t SIZE = Refs(EIGEN_NAMED_VEC_REF_SIZE,+);\
};\
EIGEN_NAMED_BASE_DECL(Name,(Eigen::Matrix<double,Name ## Helper::SIZE,1>)){\
    using Helper = Name ## Helper;\
    EIGEN_NAMED_BASE_IMPL_OP(Name,Refs,EIGEN_NAMED_VEC_REF_DECL,EIGEN_NAMED_VEC_REF_SIZE,EIGEN_NAMED_VEC_REF_INIT)\
};\
EIGEN_NAMED_MATRIX_OP(Name,Refs,EIGEN_NAMED_VEC_REF_ARG,EIGEN_NAMED_VEC_REF_ASSIGN)
#endif