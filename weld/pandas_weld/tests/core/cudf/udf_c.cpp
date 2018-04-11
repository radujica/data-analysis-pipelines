#include <immintrin.h>

// the following typedefs and templates should be included from grizzly/common.h
typedef bool       i1;
typedef int8_t     i8;
typedef int16_t    i16;
typedef int32_t    i32;
typedef int64_t    i64;

template<typename T>
struct vec {
  T *ptr;
  i64 size;
};

template<typename T>
vec<T> make_vec(i64 size) {
  vec<T> t;
  t.ptr = (T *)malloc(size * sizeof(T));
  t.size = size;

  return t;
}

extern "C" void udf_add(vec<int64_t> *arr, int64_t *scalar, vec<int64_t> *result) {
  *result = make_vec<int64_t>(arr->size);

  for (int i = 0; i < result->size; i++) {
    result->ptr[i] = arr->ptr[i] + *scalar;
  }
}
