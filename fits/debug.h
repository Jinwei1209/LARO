#ifndef _DEBUG_H_
#define _DEBUG_H_
#include <stdlib.h>
#include "stringmacro.h"

#define SHOW(a)  \
  do { \
    printf("%s :%d : %s\n",__FILE__, __LINE__, stringize(a)); \
    (a); \
  } while (0) 

#define exit(a) SHOW(exit(a))


#ifdef NO_LOC_DEBUG

#  define FPRINTSTR(fp, a, fmt) fprintf((fp), stringize(a = fmt\n), (a))

#  define FPRINTARRSTR(fp, a, fmt, num)                              \
     do {                                                            \
       int __i;                                                      \
       for (__i=0; __i<num;__i++) {                                  \
         fprintf((fp), stringize(a[%d] = fmt\n), (__i), (a[(__i)])); \
       }                                                             \
     } while(0)

#  define FPRINTMATSTR(fp, a, fmt, rows, cols) \
     do {                                      \
       int __i, __j;                           \
       for (__i=0; __i<rows;__i++) {           \
         fprintf((fp), stringize(a[%d][] = ),  \
                 (__i));                       \
           for (__j=0; __j<cols;__j++) {       \
             fprintf((fp), stringize(fmt),     \
                      (a[(__i)][__j]));        \
             fprintf((fp), " ");               \
           }                                   \
           fprintf((fp), "\n");                \
       }                                       \
     } while(0)

#else 

#  define FPRINTSTR(fp, a, fmt) fprintf((fp), stringize(%s line %d : a = fmt\n), __FILE__, __LINE__, (a))
#  define FPRINTARRSTR(fp, a, fmt, num)          \
     do {                                        \
       int __i;                                  \
       fprintf((fp), stringize(%s line %d :\n),  \
                  __FILE__, __LINE__);           \
       for (__i=0; __i<num;__i++) {              \
         fprintf((fp), stringize(a[%d] = fmt\n), \
                 (__i), (a[(__i)]));             \
       }                                         \
     } while(0)
#  define FPRINTMATSTR(fp, a, fmt, rows, cols)   \
     do {                                        \
       int __i, __j;                             \
       fprintf((fp), stringize(%s line %d :\n),  \
                  __FILE__, __LINE__);           \
       for (__i=0; __i<rows;__i++) {             \
         fprintf((fp), stringize(a[%d][] = ),    \
                 (__i));                         \
           for (__j=0; __j<cols;__j++) {         \
             fprintf((fp), stringize(fmt ),      \
                      (a[(__i)][__j]));          \
             fprintf((fp), " ");                 \
           }                                     \
           fprintf((fp), "\n");                  \
       }                                         \
     } while(0)

#endif

#define fshowi(fp, a) FPRINTSTR(fp,a,%d)
#define fshowu(fp, a) FPRINTSTR(fp,a,%u)
#define fshowf(fp, a) FPRINTSTR(fp,a,%f)
#define fshow6g(fp, a) FPRINTSTR(fp,a,%.6g)
#define fshowX(fp, a) FPRINTSTR(fp,a,%X)
#define fshows(fp, a) FPRINTSTR(fp,a,%s)
#define fshowc(fp, a) FPRINTSTR(fp,a,%c)
#define fshowl(fp, a) FPRINTSTR(fp,a,%ld)
#define showi(a) fshowi(stdout,a)
#define showu(a) fshowu(stdout,a)
#define showf(a) fshowf(stdout,a)
#define show6g(a) fshow6g(stdout,a)
#define showX(a) fshowX(stdout,a)
#define shows(a) fshows(stdout,a)
#define showc(a) fshowc(stdout,a)
#define showl(a) fshowl(stdout,a)

#define fshowarri(fp, a, n) FPRINTARRSTR(fp,a,%d,n)
#define fshowarru(fp, a, n) FPRINTARRSTR(fp,a,%u,n)
#define fshowarrf(fp, a, n) FPRINTARRSTR(fp,a,%f,n)
#define fshowarr6g(fp, a, n) FPRINTARRSTR(fp,a,%.6g,n)
#define fshowarrX(fp, a, n) FPRINTARRSTR(fp,a,%X,n)
#define fshowarrs(fp, a, n) FPRINTARRSTR(fp,a,%s,n)
#define fshowarrc(fp, a, n) FPRINTARRSTR(fp,a,%c,n)
#define fshowarrl(fp, a, n) FPRINTARRSTR(fp,a,%ld,n)
#define showarri(a, n) fshowarri(stdout,a,n)
#define showarru(a, n) fshowarru(stdout,a,n)
#define showarrf(a, n) fshowarrf(stdout,a,n)
#define showarr6g(a, n) fshowarr6g(stdout,a,n)
#define showarrX(a, n) fshowarrX(stdout,a,n)
#define showarrs(a, n) fshowarrs(stdout,a,n)
#define showarrc(a, n) fshowarrc(stdout,a,n)
#define showarrl(a, n) fshowarrl(stdout,a,n)

#define fshowmati(fp, a, n, m) FPRINTMATSTR(fp,a,%d,n,m)
#define fshowmatu(fp, a, n, m) FPRINTMATSTR(fp,a,%u,n,m)
#define fshowmatf(fp, a, n, m) FPRINTMATSTR(fp,a,%f,n,m)
#define fshowmatX(fp, a, n, m) FPRINTMATSTR(fp,a,%X,n,m)
#define fshowmats(fp, a, n, m) FPRINTMATSTR(fp,a,%s,n,m)
#define fshowmatc(fp, a, n, m) FPRINTMATSTR(fp,a,%c,n,m)
#define fshowmatl(fp, a, n, m) FPRINTMATSTR(fp,a,%ld,n,m)
#define showmati(a, n, m) fshowmati(stdout,a,n,m)
#define showmatu(a, n, m) fshowmatu(stdout,a,n,m)
#define showmatf(a, n, m) fshowmatf(stdout,a,n,m)
#define showmatX(a, n, m) fshowmatX(stdout,a,n,m)
#define showmats(a, n, m) fshowmats(stdout,a,n,m)
#define showmatc(a, n, m) fshowmatc(stdout,a,n,m)
#define showmatl(a, n, m) fshowmatl(stdout,a,n,m)

#endif
