#ifndef _UTIL_HH_
#define _UTIL_HH_
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <functional>
#include <numeric>
#include <complex>
#if defined(_WIN32) || defined(_WIN64)
#define USE_REGEX
#endif
#ifdef USE_REGEX
#include <regex>
#else
#include <sys/types.h>
#include <regex.h>
#endif

template<typename T>
bool find_elem(std::vector<T>& haystack, const T& needle)
{
  return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
}

template<typename T, typename A>
void scale(std::vector<T, A>& v, T f)
{
  std::transform(v.begin(), v.end(), v.begin(),
            std::bind2nd(std::multiplies<T>(), f));
}

template <typename T>
class make_vector {
public:
  typedef make_vector<T> my_type;
  my_type& operator<< (const T& val) {
    data_.push_back(val);
    return *this;
  }
  my_type& operator<< (const std::vector<T>& val) {
    data_.insert( data_.end(), val.begin(), val.end() );
    return *this;
  }
  operator std::vector<T>() const {
    return data_;
  }
private:
  std::vector<T> data_;
};

template<class A1, class A2>
std::ostream& operator<<(std::ostream& s, std::vector<A1, A2> const& vec)
{
  std::cout << "[ ";
  for(typename std::vector<A1, A2>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
    std::cout << (*it) << " ";
  }
  std::cout << "]";
  return s;
}

template<typename T, typename A>
T prod(std::vector<T, A> v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template<class T> 
std::complex<T> operator | (const std::complex<T>& lhs, const std::complex<T>& rhs)
{
  std::complex<T> out = lhs * std::conj(rhs);
  float t = std::norm(rhs);
  if (t > 0.0f) out /= t;
  return out;
}

template<class T> 
void operator |= (std::complex<T>& lhs, const std::complex<T>& rhs)
{
  lhs *= std::conj(rhs);
  float t = std::norm(rhs);
  if (t > 0.0f) lhs /= t;
}

template <typename T> 
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

inline void strrep(std::string& subject, const std::string& search,
            const std::string& replace) {
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

inline std::string
sanitizeString(const std::string& original,
                            const std::string& invalidChars,
                            const std::string& replacement) {
#ifdef USE_REGEX
    std::regex rexpr;
    rexpr = std::regex(invalidChars, std::regex::extended);
    std::stringstream result;
    std::regex_replace(std::ostream_iterator<char>(result),
        original.begin(), original.end(),
        rexpr, replacement);
    return (result.str());
#else
    // Compile the expression.
    regex_t rex;
    int ec = regcomp(&rex, invalidChars.c_str(), REG_EXTENDED);
    if (ec) {
        return original;
        //char errbuf[512] = "";
        //static_cast<void>(regerror(ec, &rex, errbuf, sizeof(errbuf)));
        //isc_throw(isc::BadValue, "invalid regex: '" << invalidChars
        //          << "', " << errbuf);
    }

    // Iterate over original string, match by match.
    const char* origStr = original.c_str();
    const char* startFrom = origStr;
    const char* endAt = origStr + strlen(origStr);
    regmatch_t matches[2];  // n matches + 1
    std::stringstream result;

    while (startFrom < endAt) {
        // Look for the next match
        if (regexec(&rex, startFrom, 1, matches, 0) == REG_NOMATCH) {
            // No matches, so add in the remainder
            result << startFrom;
            break;
        }

        // Shouldn't happen, but one never knows eh?
        //if (matches[0].rm_so == -1) {
        //    isc_throw(isc::Unexpected, "matched but so is -1?");
        //}

        // Add everything from starting point up to the current match
        const char* matchAt = startFrom + matches[0].rm_so;
        while (startFrom < matchAt) {
            result << *startFrom;
            ++startFrom;
        }

        // Add in the replacement
        result << replacement;

        // Move past the match.
        ++startFrom;
    }

    regfree(&rex);
    return (result.str());
#endif
}

//template <typename T> // T models Any
//struct static_cast_func
//{
//  template <typename T1> // T1 models type statically convertible to T
//  T operator()(const T1& x) const { return std::static_cast<T>(x); }
//};

//template<typename T1, typename T2>
//void array_cast(T1* src, T2 *dest, size_t n)
//{
//  std::transform(src, src + n, dest, static_cast_func<dest>());
//}

inline float machineEpsilon()
{
    float eps=1.0f;
    while ((1.0f+eps) != 1.0f) { eps/=2.0f; }
    return eps;
}

#define VAR(V,init) __typeof(init) V=(init)
#define FOREACH(I,C) for(VAR(I,(C).begin());I<(C).end();I++)

#define PRINT_stringize(y) #y
#define PRINT(a) (std::cout << __FILE__ << ":" << __LINE__ <<" " << PRINT_stringize(a) << " = " << (a) << std::endl)
#define PRINTA(a,n) \
  (std::cout << __FILE__ << ":" << __LINE__ <<" " << PRINT_stringize(a) << " = " << std::vector<__typeof(*(a))>((a),(a)+n) << std::endl)


#if defined(__CYGWIN__) 
#define Isnan(x) isnan(x)
#define Isinf(x) isinf(x)
#elif defined(__APPLE__)
#define Isnan(x) std::isnan(x)
#define Isinf(x) std::isinf(x)
#else
#define Isnan(x) ::isnan(x)
#define Isinf(x) ::isinf(x)
#endif


//#define MKL_FREE(a) do { if (NULL != (a)) { printf("%s line %d\n", __FILE__, __LINE__); fflush(stdout); mkl_free(a); (a)=NULL;} } while (0)
#define MKL_FREE(a) do { if (NULL != (a)) { mkl_free(a); (a)=NULL;} } while (0)
#define MKL_ALLOC(a,type,num) \
do { \
  (a) = (type*)mkl_malloc((num)*sizeof(type), 64); \
} while (0)

#define MKL_ALLOC_ZERO(a,type,num) \
do { \
  (a) = (type*)mkl_malloc((num)*sizeof(type), 64); \
  memset((a), 0, sizeof(type)*(num));\
} while (0)

inline bool almostEqual(float f1, float f2) {
  float eps = 4.0f*std::numeric_limits<float>::epsilon();
  return fabs(f1-f2)<eps;
}
inline bool notEqual(float f1, float f2) {
  float eps = 4.0f*std::numeric_limits<float>::epsilon();
  return fabs(f1-f2)>=eps;
}

#endif
