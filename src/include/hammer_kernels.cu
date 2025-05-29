#include <cuda_helpers.cuh>
#include <hammer_kernels.cuh>
#include <memory>
#include <algorithm>

static uint8_t **get_aggressor_device_addr(RowList &rows,
                                           std::vector<uint64_t> &agg_vec);

uint64_t start_simple_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                             uint64_t it)
{
  /* GPU memory to store aggressors */
  uint8_t **agg_device_arr = get_aggressor_device_addr(rows, agg_vec);

  /* Start hammering */
  auto dim = get_dim_from_size(agg_vec.size());
  int numBlock = std::get<0>(dim);
  int numThreads = std::get<1>(dim);

  /* Setup time measures */
  uint64_t *timeSpentDevice;
  uint64_t timeSpentHost;
  cudaMalloc(&timeSpentDevice, sizeof(uint64_t *));

  std::cout << CLI_PREFIX << "Iterating: " << it << " times\n";

  simple_hammer_kernel<<<numBlock, numThreads>>>(agg_device_arr, it,
                                                 timeSpentDevice);
  cudaDeviceSynchronize();
  cudaMemcpy(&timeSpentHost, timeSpentDevice, sizeof(uint64_t *),
             cudaMemcpyDeviceToHost);

  cudaFree(agg_device_arr);
  return toNS(timeSpentHost);
}

uint64_t start_single_thread_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                                  uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period)
{
  /* GPU memory to store aggressors */
  uint8_t **agg_device_arr = get_aggressor_device_addr(rows, agg_vec);

  uint64_t *timeSpentDevice;
  uint64_t timeSpentHost;
  cudaMalloc(&timeSpentDevice, sizeof(uint64_t *));

  std::cout << CLI_PREFIX << "Iterating: " << it << " times\n";
  std::cout << CLI_PREFIX << "Delay: " << delay << "\n";

  single_thread_hammer<<<1, 1>>>(agg_device_arr, it, len, timeSpentDevice);
  cudaDeviceSynchronize();

  cudaMemcpy(&timeSpentHost, timeSpentDevice, sizeof(uint64_t *),
             cudaMemcpyDeviceToHost);
  cudaFree(agg_device_arr);
  cudaFree(timeSpentDevice);
  return toNS(timeSpentHost);
}

uint64_t start_multi_thread_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                                  uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period)
{
  /* GPU memory to store aggressors */
  uint8_t **agg_device_arr = get_aggressor_device_addr(rows, agg_vec);

  uint64_t *timeSpentDevice;
  uint64_t timeSpentHost;
  cudaMalloc(&timeSpentDevice, sizeof(uint64_t *));

  std::cout << CLI_PREFIX << "Iterating: " << it << " times\n";
  std::cout << CLI_PREFIX << "Delay: " << delay << "\n";

  sync_hammer_kernel<<<1, len>>>(agg_device_arr, it, delay, period, timeSpentDevice);
  cudaDeviceSynchronize();

  cudaMemcpy(&timeSpentHost, timeSpentDevice, sizeof(uint64_t *),
             cudaMemcpyDeviceToHost);
  cudaFree(agg_device_arr);
  cudaFree(timeSpentDevice);
  return toNS(timeSpentHost);
}

uint64_t start_warp_simple_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                                  uint64_t it, uint64_t n, uint64_t k, uint64_t len, uint64_t delay, uint64_t period)
{
  /* GPU memory to store aggressors */
  uint8_t **agg_device_arr = get_aggressor_device_addr(rows, agg_vec);

  uint64_t *timeSpentDevice;
  uint64_t timeSpentHost;
  cudaMalloc(&timeSpentDevice, sizeof(uint64_t *));

  std::cout << CLI_PREFIX << "Iterating: " << it << " times\n";
  std::cout << CLI_PREFIX << "Delay: " << delay << "\n";

  warp_simple_hammer_kernel<<<1, 1024>>>(agg_device_arr, it, n, k, len, delay, period, timeSpentDevice);
  // single_thread_hammer<<<1, 1>>>(agg_device_arr, it, n, delay, timeSpentDevice);
  cudaDeviceSynchronize();

  cudaMemcpy(&timeSpentHost, timeSpentDevice, sizeof(uint64_t *),
             cudaMemcpyDeviceToHost);
  cudaFree(agg_device_arr);
  cudaFree(timeSpentDevice);
  return toNS(timeSpentHost);
}


uint64_t start_trh_hammer(RowList &rows, std::vector<uint64_t> &agg_vec,
                          std::vector<uint64_t> &dum_vec,
                          uint64_t it, uint64_t n, uint64_t k, uint64_t len, 
                          uint64_t delay, uint64_t period,
                          uint64_t agg_period, uint64_t dum_period)
{
  /* GPU memory to store aggressors */
  uint8_t **agg_device_arr = get_aggressor_device_addr(rows, agg_vec);
  uint8_t **dum_device_arr = get_aggressor_device_addr(rows, dum_vec);

  uint64_t *timeSpentDevice;
  uint64_t timeSpentHost;
  cudaMalloc(&timeSpentDevice, sizeof(uint64_t *));

  std::cout << CLI_PREFIX << "Iterating: " << it << " times\n";
  std::cout << CLI_PREFIX << "Delay: " << delay << "\n";

  rh_threshold_kernel<<<1, 1024>>>(agg_device_arr, dum_device_arr, it, n, k, len, 
                                    delay, period, timeSpentDevice, 
                                    agg_period, dum_period);
  cudaDeviceSynchronize();

  cudaMemcpy(&timeSpentHost, timeSpentDevice, sizeof(uint64_t *),
             cudaMemcpyDeviceToHost);
  cudaFree(agg_device_arr);
  cudaFree(dum_device_arr);
  cudaFree(timeSpentDevice);
  return toNS(timeSpentHost);
}

uint8_t **get_aggressor_device_addr(RowList &rows,
                                    std::vector<uint64_t> &agg_vec)
{
  uint8_t **agg_device_arr;
  cudaMalloc(&agg_device_arr, sizeof(uint8_t *) * agg_vec.size());

  /* Copy aggressors to GPU Memory */
  auto agg_host_arr = std::make_unique<uint8_t *[]>(agg_vec.size());
  for (auto i = 0; i < agg_vec.size(); i++)
    *(agg_host_arr.get() + i) = rows[agg_vec[i]][0];

  cudaMemcpy(agg_device_arr, agg_host_arr.get(),
             sizeof(uint8_t *) * agg_vec.size(), cudaMemcpyHostToDevice);

  return agg_device_arr;
}
