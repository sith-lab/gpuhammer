#include <hammer_util.cuh>
#include <stdint.h>

#ifndef GPU_ROWHAMMER_HAMMER_KERNELS_CUH
#define GPU_ROWHAMMER_HAMMER_KERNELS_CUH

uint64_t start_simple_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                             uint64_t it);

uint64_t start_single_thread_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                                  uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period);

uint64_t start_multi_thread_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                                  uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period);

uint64_t start_warp_simple_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                             uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period);

uint64_t start_trh_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                          std::vector<uint64_t> &dum_vec,
                          uint64_t it, uint64_t n, uint64_t k, uint64_t len, 
                          uint64_t delay, uint64_t period,
                          uint64_t agg_period, uint64_t dum_period);
                          
#endif /* GPU_ROWHAMMER_HAMMER_KERNELS_CUH */