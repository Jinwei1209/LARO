#ifndef _STRINGMACRO_H_
#define _STRINGMACRO_H_

#define stringize(y) #y

#define CONCAT0(a, b)          a ## b
#define CONCAT(a, b)           CONCAT0(a, b)
#define CONCAT3(a, b, c)       CONCAT(a, CONCAT(b,c))
#define CONCAT4(a, b, c ,d)    CONCAT(a, CONCAT3(b,c,d))
#define CONCAT5(a, b, c ,d, e) CONCAT(a, CONCAT4(b,c,d,e))

#endif
