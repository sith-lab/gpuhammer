#include <hammer_util.cuh>
#include <atomic>
#include <chrono>
#include <cuda_helpers.cuh>
#include <hammer_kernels.cuh>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <stdint.h>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm>

std::string CLI_PREFIX = "(rh-threshold): ";
int main(int argc, char *argv[])
{
  const uint64_t num_victim = std::stoull(argv[2]);
  const uint64_t step       = std::stoull(argv[3]);
  const uint64_t total_it   = std::stoull(argv[4]);
  const uint64_t min_rowId  = std::stoull(argv[5]);
  const uint64_t max_rowId  = std::stoull(argv[6]);
  const uint64_t dum_rowId  = std::stoull(argv[7]);
  const uint64_t row_step   = std::stoull(argv[8]);
  const uint64_t skip_step  = std::stoull(argv[9]);
  const uint64_t size       = std::stoull(argv[10]);
  const uint64_t n          = std::stoull(argv[11]);
  const uint64_t k          = std::stoull(argv[12]);
  const uint64_t delay      = std::stoull(argv[13]);
  const uint64_t period     = std::stoull(argv[14]);
  const uint64_t count_iter = std::stoull(argv[15]);
  const uint64_t num_rows   = std::stoull(argv[16]);
  const uint64_t vic_pat    = std::stoull(argv[17], nullptr, 16);
  const uint64_t agg_pat    = std::stoull(argv[18], nullptr, 16);
  const uint64_t agg_period = std::stoull(argv[19]);
  const uint64_t dum_period = std::stoull(argv[20]);
  std::ofstream bitflip_file(argv[21]);


  uint64_t it = total_it / (agg_period + dum_period);

  /* Read the row set */
  uint8_t *layout;
  cudaMalloc(&layout, size);
  
  std::ifstream row_set_file(argv[1]);
  RowList rows = read_row_from_file(row_set_file, layout);
  row_set_file.close();

  if ((int64_t)(rows.size() - 2 * num_victim - 1) < 0)
  {
    std::cout << CLI_PREFIX << "Error: "
              << "Not enough rows to generate the specified victims." << '\n';
    exit(-1);
  }

  std::cout << CLI_PREFIX << "Layout address: " << static_cast<void*>(layout) << '\n';
  std::cout << std::hex;
  std::cout << CLI_PREFIX << "Victim pattern: " << vic_pat << '\n';
  std::cout << CLI_PREFIX << "Aggressor pattern: " << agg_pat << '\n';
  std::cout << std::dec;

  /* Treat all rows as victim rows */
  std::vector<uint64_t> all_vics(num_rows);
  std::iota(all_vics.begin(), all_vics.end(), 0);
  set_rows(rows, all_vics, vic_pat, step);
  cudaDeviceSynchronize();

  /* Dummy hammer to keep timing consistent, due to device startup time */
  start_simple_hammer(rows, all_vics, 1);

  std::vector<int> bitflip_count(std::ceil((max_rowId - min_rowId) / skip_step), 0);

  /* Running */
  for (int i = min_rowId; i < max_rowId; i += skip_step) {

    /* Initialize indexes of victims and aggressors */
    std::vector<uint64_t> victims = get_sequential_victims(rows, i, num_victim + 2, row_step);
    std::vector<uint64_t> aggressors = get_aggressors(rows, i, num_victim + 1, row_step);
    std::vector<uint64_t> dummies = get_aggressors(rows, dum_rowId, num_victim + 1, row_step);
    // TODO: i -> dum_rowId
    
    std::cout << CLI_PREFIX << "Chosen Victims:" << vector_str(victims) << std::endl;
    std::cout << CLI_PREFIX << "Chosen Aggressors:" << vector_str(aggressors) << std::endl;
    std::cout << CLI_PREFIX << "Chosen Dummies:" << vector_str(dummies) << std::endl;
    std::cout << CLI_PREFIX << "==========================================================" << std::endl;

    for (int j = 0; j < count_iter; j++) {
      
      std::cout << CLI_PREFIX << "Aggressor Iteration: " << j << std::endl;
      auto start_loop = std::chrono::high_resolution_clock::now();

      std::vector<uint64_t> pat_vics(110);
      std::iota(pat_vics.begin(), pat_vics.end(), victims[0] - 4);

      /* Sets the row and evict cache to store it in the memory. */
      set_rows(rows, pat_vics, vic_pat, step);
      set_rows(rows, dummies, vic_pat, step);
      set_rows(rows, aggressors, agg_pat, step);
      cudaDeviceSynchronize();

      evict_L2cache(layout);
      cudaDeviceSynchronize();

      auto start_hammer = std::chrono::high_resolution_clock::now();

      /* Start the hammering and measure the time */
      uint64_t time = start_trh_hammer(rows, aggressors, dummies, it, n, k, 
                                      aggressors.size(), delay, period,
                                      agg_period, dum_period);
      print_time(time);
      std::cout << CLI_PREFIX << "Average time per round: " << time / total_it << std::endl;

      auto end_hammer = std::chrono::high_resolution_clock::now();

      /* Verify result */
      evict_L2cache(layout);

      // Comment out the first line and uncomment the following line to check 
      // for bit-flips in the nearby neighborhood to reduce hammering time.
      bool res = verify_all_content(rows, all_vics, aggressors, step, vic_pat);
      // bool res = verify_all_content(rows, pat_vics, aggressors, step, vic_pat);
      
      std::cout << CLI_PREFIX << "Bit-flip in victim rows: " 
                              << (res ? "Observed Bit-Flip" : "No Bit-Flip") << std::endl;
      if (res) bitflip_count[std::ceil((i - min_rowId) / skip_step)] ++;

      /* Clean up and prepare for next launch*/
      cudaDeviceSynchronize();
      auto end_loop = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration_evict = start_hammer - start_loop;
      std::chrono::duration<double, std::milli> duration_hammer = end_hammer - start_hammer;
      std::chrono::duration<double, std::milli> duration_verify = end_loop - end_hammer;
      std::chrono::duration<double, std::milli> duration_total = end_loop - start_loop;
      std::cout << CLI_PREFIX << "Evict time: " << duration_evict.count() << " ms" << std::endl;
      std::cout << CLI_PREFIX << "Hammer time: " << duration_hammer.count() << " ms" << std::endl;
      std::cout << CLI_PREFIX << "Verify time: " << duration_verify.count() << " ms" << std::endl;
      std::cout << CLI_PREFIX << "Total time: " << duration_total.count() << " ms" << std::endl;

      std::cout << CLI_PREFIX << "==========================================================" << std::endl;
    }
  }

  for (int i = 0; i < bitflip_count.size(); i++) {
    bitflip_file << bitflip_count[i] << std::endl;
  }
  bitflip_file.close();

  return 0;
}