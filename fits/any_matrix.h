#ifndef _ANY_MATRIX_H_
#define _ANY_MATRIX_H_

#ifdef __cplusplus
extern "C" {
#endif


#define ANY_ALIGN (16)

#define ANY_MARKER 12345678
#define any_matrix(dimension, el_size, ...) \
        _any_matrix(dimension, el_size, ## __VA_ARGS__, ANY_MARKER)

#define any_matrix_realloc(ptr, dimension, el_size, ...) \
        _any_matrix_realloc(ptr, dimension, el_size, ## __VA_ARGS__, ANY_MARKER)

#define any_matrix_wrap(ptr, dimension, el_size, ...) \
        _any_matrix_wrap(ptr, dimension, el_size, ## __VA_ARGS__, ANY_MARKER)

void * _any_matrix(int dimension, int el_size, ...);
void * _any_matrix_realloc(void *ptr, int dimension, int el_size, ...);
void * _any_matrix_wrap(void *ptr, int dimension, int el_size, ...);
unsigned long any_matrix_data_offset(void *ptr, int dimension, int el_size, int *dims);
void *any_matrix_2(int dimension, int el_size, int *dims);
void *any_matrix_2_realloc(void *ptr, int dimension, int el_size, int *dims);
void *any_matrix_2_wrap(void *ptr, int dimension, int el_size, int *dims);

#ifdef __cplusplus
}
#endif

#endif
