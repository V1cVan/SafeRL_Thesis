#ifndef SIM_EIGEN
#define SIM_EIGEN

#include "Eigen/Core"

/* Following code fragments are based on:
 * https://stackoverflow.com/questions/54512878/eigen-custom-classes-and-function-parameters
 * https://stackoverflow.com/questions/54523063/extending-eigen-ref-class
 * https://stackoverflow.com/questions/62829850/eigen-named-access-to-vector-segment
 */

/*
Macro that defines a base structure of a custom named matrix. This
requires the existence of a Name ## Interface structure outlining the
common structure between actual named matrices derived from this base
type and Eigen references to these named matrices. See also below two
macros.
*/
#define EIGEN_NAMED_BASE(Name,InterfaceInit)\
template<typename T>\
struct Name ## Base : T, Name ## Interface{\
    /* T is either Name ## Interface::Base or Eigen::Ref<[const] Name ## Interface::Base> */\
    static_assert(std::is_same<T,Name ## Interface::Base>::value || std::is_same<T,Eigen::Ref<Name ## Interface::Base>>::value || std::is_same<T,Eigen::Ref<const Name ## Interface::Base>>::value);\
    using Base = T;\
\
    template<typename Derived>\
    Name ## Base(const Eigen::DenseBase<Derived>& expr)\
    : Base(expr), Name ## Interface InterfaceInit{}\
\
    /* TODO: below three methods seem only to be necessary for MSVC in Debug mode (see https://stackoverflow.com/questions/62829850/eigen-named-access-to-vector-segment) */\
    /* TODO: Not sure if they are still necessary after the refactoring */\
    Name ## Base(const Name ## Base& other)\
    : Base(other), Name ## Interface InterfaceInit{\
        /* Default copy constructor causes dangling references in debug mode. */\
    }\
\
    Name ## Base(Name ## Base&& other)\
    : Base(std::move(other)), Name ## Interface InterfaceInit{\
        /* Default move constructor causes dangling references in debug mode. */\
    }\
};


#define EIGEN_NAMED_MATRIX_BASE(Name) Name ## Base<Name ## Interface::Base>
/*
Macro that introduces all boilerplate constructors/assignment operators
required to easily work with named matrices/vectors in Eigen. This requires
a previously defined Base for this named matrix and requires the given class
to inherit from Name ## Base<Name ## Interface::Base>.
*/
#define EIGEN_NAMED_MATRIX_IMPL(Name)\
Name() : Name(Base::Zero()){} /* Default constructor */\
\
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


#define EIGEN_NAMED_REF_BASE(Name) Name ## Base<Eigen::Ref<Name ## Interface::Base>>
#define EIGEN_NAMED_CONSTREF_BASE(Name) Name ## Base<Eigen::Ref<const Name ## Interface::Base>>
/* Macro that defines an Eigen::Ref implementation for a custom named matrix.
This requires a previously defined Base for this named matrix. */
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
        Ref<const Name>(DenseBase<Derived>& expr)\
        : EIGEN_NAMED_CONSTREF_BASE(Name)(expr){}\
    };\
};

#endif