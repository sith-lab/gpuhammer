#include "cuda_helpers.cuh"
#include "hammer_util.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void set_row(Row &row, uint8_t pat, uint64_t b_count);
static void clear_L2cache_row(Row &row, uint64_t step);
static uint64_t find_diff(const std::vector<uint8_t> &vec1,
                          const std::vector<uint8_t> &vec2);
inline uint64_t CurrentTime_nanoseconds()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

/**
 * @brief Returns the row set in matrix form as seen in the file. Each newline
 * is a row and addresses are tab-seperated
 *
 * @param file
 * @param base_addr
 * @return RowList
 */
RowList read_row_from_file(std::ifstream &file, const uint8_t *base_addr)
{
  std::string buf;
  std::vector<std::vector<uint8_t *>> rows;
  while (std::getline(file, buf))
  {
    rows.emplace_back();
    std::stringstream ss;
    ss << buf;
    while (std::getline(ss, buf, '\t')){
      rows.back().push_back((uint8_t *)(base_addr + std::stoull(buf)));
      // std::cout << static_cast<const void*>(base_addr + std::stoull(buf)) << '\n';
    }
  }
  return rows;
}

/**
 * @brief Return a vector of v_count victims chosen randomly, where each victim
 * are 1 row apart from each other.
 *
 * @param rows
 * @param v_count
 * @return std::vector<uint64_t>
 */
std::vector<uint64_t> get_random_victims(RowList &rows, uint64_t v_count)
{

  /* 1. Ignore first and last row for two sided hammer, between 1 and size-1 */
  /* 2. choose only from non-neighboring rows, i.e. odd numbers [1, size-1] */

  uint64_t num_between = rows.size() - 2;
  uint64_t num_valid_victim = std::ceil(num_between / 2.0);
  std::vector<uint64_t> vic_vec(num_valid_victim);

  /* Initialize as odd numbers starting from 1*/
  std::generate(vic_vec.begin(), vic_vec.end(),
                [value = -1]() mutable -> uint64_t
                {
                  value += 2;
                  return value;
                });

  /* Shuffle the vector at random */
  std::mt19937 generator(std::time(0));
  std::shuffle(vic_vec.begin(), vic_vec.end(), std::move(generator));
  return {vic_vec.rbegin(), vic_vec.rbegin() + v_count};
}

std::vector<uint64_t> get_random_sequential_victims(RowList &rows,
                                                    uint64_t v_count)
{
  uint64_t latest_row = rows.size() - 2 * v_count;
  std::cout << latest_row << '\n';
  std::mt19937 generator(std::time(0));
  std::uniform_int_distribution<> random_range(1, latest_row);
  auto random_row_id = random_range(generator);

  std::vector<uint64_t> vic_vec(v_count);
  std::generate(vic_vec.begin(), vic_vec.end(),
                [value = random_row_id - 2]() mutable -> uint64_t
                {
                  value += 2;
                  return value;
                });
  return get_sequential_victims(rows, random_row_id, v_count);
}

std::vector<uint64_t> get_sequential_victims(RowList &rows, uint64_t row_id,
                                             uint64_t v_count)
{
  std::vector<uint64_t> vic_vec(v_count);
  std::generate(vic_vec.begin(), vic_vec.end(),
                [value = row_id - 2]() mutable -> uint64_t
                {
                  value += 2;
                  return value;
                });
  return vic_vec;
}

/**
 * @brief From victims, get the rows around it to be aggressors
 *
 * @param victims
 * @return std::vector<uint64_t>
 */
std::vector<uint64_t> get_aggressors(std::vector<uint64_t> &victims)
{
  std::set<uint64_t> agg;
  for (const auto &vic : victims)
  {
    agg.insert(vic + 1);
    agg.insert(vic - 1);
  }
  return {agg.begin(), agg.end()};
}

// with variable step size
std::vector<uint64_t> get_sequential_victims(RowList &rows, uint64_t row_id,
                                             uint64_t num_vic, uint64_t step)
{
  std::vector<uint64_t> vic_vec(num_vic);
  std::generate(vic_vec.begin(), vic_vec.end(),
                [value = row_id - step * 3 / 2, step = step]() mutable -> uint64_t
                {
                  value += step;
                  return value;
                });
  return vic_vec;
}

std::vector<uint64_t> get_aggressors(RowList &rows, uint64_t row_id,
                                     uint64_t num_agg, uint64_t step)
{
  std::vector<uint64_t> agg_vec(num_agg);
  std::generate(agg_vec.begin(), agg_vec.end(),
                [value = row_id - step, step = step]() mutable -> uint64_t
                {
                  value += step;
                  return value;
                });
  return agg_vec;
}

/**
 * @brief Helper function to set all the target rows to pat
 *
 * @param rows
 * @param target_rows
 * @param pat
 * @param b_count byte difference between each address
 */
void set_rows(RowList &rows, std::vector<uint64_t> &target_rows, uint8_t pat,
              uint64_t b_count)
{
  for (const auto v : target_rows)
    set_row(rows[v], pat, b_count);
}

void clear_L2cache_rows(RowList &rows, std::vector<uint64_t> &target_rows, uint64_t step)
{
  for (const auto v : target_rows)
    clear_L2cache_row(rows[v], step);
}

/**
 * @brief Return whether the victim has bitflip or not.
 *
 * @param rows
 * @param victims
 * @param aggressors
 * @param b_count
 * @return true
 * @return false
 */
bool verify_content(RowList &rows, std::vector<uint64_t> &victims,
                    std::vector<uint64_t> &aggressors, uint64_t b_count,
                    uint8_t pat)
{
  /* Expected constants */
  const std::vector<uint8_t> vic_exp(b_count, pat);

  uint8_t value[b_count];
  std::vector<uint64_t> total_diff;

  /* Count the number of bitflips per victim */
  for (const auto v : victims)
  {
    uint64_t total_vic_diff = 0;
    for (const auto addr : rows[v])
    {
      cudaMemcpy(&value, addr, b_count, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      std::vector<uint8_t> vic_mem{value, value + b_count};
      total_vic_diff += find_diff(vic_exp, vic_mem);
    }

    if (std::count(aggressors.begin(), aggressors.end(), v) == 0) {
      total_diff.push_back(total_vic_diff);
    } else {
      total_diff.push_back(0);
    }
    
  }

  /* Report to user */
  bool has_diff = false;
  for (auto i = 0; i < total_diff.size(); i++)
  {
    if (total_diff[i] != 0)
    {
      std::cout << CLI_PREFIX << "Victim " << victims[i] << " has a total bitflip of "
                << total_diff[i] << '\n';
      has_diff = true;
    }
  }
  return has_diff;
}

bool verify_all_content(RowList &rows, std::vector<uint64_t> &victims,
                        std::vector<uint64_t> &aggressors, 
                        const uint64_t b_count, const uint8_t pat)
{
  bool *diff_device;
  bool diff;
  uint8_t **addrs_device;
  int batchSize = 64;
  cudaMalloc(&diff_device, sizeof(bool *));
  cudaMemset(diff_device, 0, sizeof(bool *));
  cudaMalloc(&addrs_device, sizeof(uint8_t *) * 8 * batchSize);

  for (int i = 0; i < victims.size(); i += batchSize)
  {
    int amount = 0;
    for (int j = i; j < i + batchSize && j < victims.size(); j++)
    {
      auto& v = victims[j];
      if (std::count(aggressors.begin(), aggressors.end(), v) != 0) continue;

      int size = rows[v].size() <= 8 ? rows[v].size() : 8;
      cudaMemcpy(addrs_device + amount, rows[v].data(), size * sizeof(uint8_t *), cudaMemcpyHostToDevice);
      amount += size;
    }
    if (amount != 0)
      verify_result_kernel<<<amount, 256>>>(addrs_device, pat, b_count, diff_device);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(&diff, diff_device, sizeof(bool *), cudaMemcpyDeviceToHost);
  cudaFree(diff_device);
  return diff;
}

/**
 * @brief Sleeps the curent thread for time time_type amount.
 *
 * @param time
 * @param time_type {'s', 'm', 'h'}
 */
void sleep_for(uint64_t time, char time_type)
{
  std::cout << CLI_PREFIX << "Hammering for " << time << time_type << "\n";
  switch (time_type)
  {
  case 'h':
    std::this_thread::sleep_for(std::chrono::hours(time));
    break;
  case 'm':
    std::this_thread::sleep_for(std::chrono::minutes(time));
    break;
  case 's':
    std::this_thread::sleep_for(std::chrono::seconds(time));
    break;
  default:
    exit(-1);
    break;
  }
}

/**
 * @brief Attempts to Evicts the L2 cache to enforce write backs for rows we
 * just set patterns to.
 *
 * @param layout
 */
void evict_L2cache(uint8_t *layout)
{
  struct cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  // std::cerr << "L2 Cache Size: " << device_prop.l2CacheSize << std::endl;

  uint64_t size = device_prop.l2CacheSize * 8;
  static uint64_t maxThreads = []()
  {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    return device_prop.maxThreadsDim[2];
  }();
  
  evict_kernel<<<1, maxThreads>>>(layout, size / maxThreads);
  // ////
  // for (int i = 0; i <= device_prop.l2CacheSize * 8; i += 128)
  //   normal_address_access<<<1, 1>>>(layout + i, 0);

  // ////
  // uint64_t size = device_prop.l2CacheSize * 36 / 128;
  // std::cout << "l2cachesize: " << device_prop.l2CacheSize << '\n';

  // static int numBlock = std::get<0>(get_dim_from_size(size));
  // static int numThreads = std::get<1>(get_dim_from_size(size));
  // std::cout << "Block: " << numBlock << " Threads: " << numThreads << '\n';

  // normal_address_access<<<numBlock, numThreads>>>(layout, 128);
  // gpuErrchk(cudaPeekAtLastError());
}

void print_time(uint64_t time_ns)
{
  std::cout << CLI_PREFIX << "Took Approx: " << time_ns << "ns\n";
  std::cout << CLI_PREFIX << "Took Approx: " << (time_ns) / 1000.0 << "us\n";
  std::cout << CLI_PREFIX << "Took Approx: " << (time_ns) / 1000000.0 << "ms\n";
}

/**
 * @brief Set the b_count bytes of each address in row to pat. This uses L2
 * cache.
 *
 * @param row
 * @param pat
 * @param b_count
 */
void set_row(Row &row, uint8_t pat, uint64_t b_count)
{
  /* Constant for this function */
  static int numBlock = std::get<0>(get_dim_from_size(b_count));
  static int numThreads = std::get<1>(get_dim_from_size(b_count));

  for (const auto addr : row) {
    set_address_kernel<<<numBlock, numThreads>>>(addr, pat, b_count);
    gpuErrchk(cudaPeekAtLastError());
  }
}

void clear_L2cache_row(Row &row, uint64_t step)
{
  for (auto addr : row)
    clear_address_kernel<<<1, 1>>>(addr, step);
}

uint64_t find_diff(const std::vector<uint8_t> &vec1,
                   const std::vector<uint8_t> &vec2)
{
  uint64_t diff_count = 0;

  for (auto i = 0; i < vec1.size(); i++)
  {
    uint8_t diff = vec1[i] ^ vec2[i];
    for (int j = 0; j < 8; ++j)
    {
      if (diff & (1 << j))
      {
        ++diff_count;
      }
    }
  }
  return diff_count;
}

/* Set each addr in row to a random value. */
void initialize_rows(RowList &rows, uint64_t b_count) {
  static int numBlock = std::get<0>(get_dim_from_size(b_count));
  static int numThreads = std::get<1>(get_dim_from_size(b_count));
  std::cout << "Block: " << numBlock << " Threads: " << numThreads << '\n';

  for (auto &row : rows) {
    for (auto addr : row) {

      uint64_t value = rand();
      set_address_kernel<<<numBlock, numThreads>>>(addr, value, b_count);
      gpuErrchk(cudaPeekAtLastError());

    }
  }
}