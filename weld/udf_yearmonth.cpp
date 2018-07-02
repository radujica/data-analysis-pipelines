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

extern "C" void udf_yearmonth(vec<vec<int8_t>> *arr, vec<vec<int8_t>> *result) {
  *result = make_vec<vec<int8_t>>(arr->size);
  int64_t res_size = 6;

  for (int i = 0; i < result->size; i++) {
    result->ptr[i].size = res_size;
    result->ptr[i].ptr = (int8_t *) malloc(6 * sizeof(int8_t));
    result->ptr[i].ptr[0] = arr->ptr[i].ptr[0];
    result->ptr[i].ptr[1] = arr->ptr[i].ptr[1];
    result->ptr[i].ptr[2] = arr->ptr[i].ptr[2];
    result->ptr[i].ptr[3] = arr->ptr[i].ptr[3];
    result->ptr[i].ptr[4] = arr->ptr[i].ptr[5];
    result->ptr[i].ptr[5] = arr->ptr[i].ptr[6];
  }
}
