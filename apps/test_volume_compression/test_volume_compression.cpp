// +++HDR+++
// ======================================================================
// 
//   This file is part of the TDM SPARSE CODING software library.
// 
//   Main authors: Enrico Gobbetti & Fabio Marton
// 
//   Copyright (C) 2021 by CRS4, Cagliari, Italy
// 
//   For more information, visit the CRS4 Visual Computing 
//   web pages at http://www.crs4.it/vic/ and the TDM project web
//   pages at  http://www.tdm-project.it
// 
//   The software is distributed under the CC BY-NC-ND 4.0 licence.
//   https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
//  
//   For use in commercial projects, kindly contact gobbetti@crs4.it.
//   If you use this software for research or in a publication, kindly
//   cite the original references. 
// 
//   CRS4 reserves all rights not expressly granted herein.
//   
//   This file is provided AS IS with NO WARRANTY OF ANY KIND, 
//   INCLUDING THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS 
//   FOR A PARTICULAR PURPOSE.
// 
// ======================================================================
// ---HDR---
#include "config.h"

// Input
#include "raw_volume.hpp"
#include "volume_tile_stream.hpp"

// Coreset
#include <vic/tdm_sparse_coding/streaming_coreset_builder_reservoir.hpp>
#include <vic/tdm_sparse_coding/streaming_coreset_builder_uniform.hpp>

// Training
#include <vic/tdm_sparse_coding/dictionary_coder_ksvd.hpp>

// Encoding
#include <vic/tdm_sparse_coding/dictionary_coding_budget_optimizer.hpp>
#include <vic/tdm_sparse_coding/dictionary_coding_budget_optimizer_greedy_grow.hpp>

// Error
#include <vic/tdm_sparse_coding/error_evaluator.hpp>

// Misc
#include <sl/clock.hpp>
#include <sl/keyed_heap.hpp>
#include <Eigen/Core>
#include <unistd.h> // For getopt
#include <cstdlib> // For random
#include <sstream>
#include <iostream>
#include <cassert>
#include <fenv.h> // For numerical debugging

#include <omp.h>

#include<queue>

// ===========================================================================
// Config
// ===========================================================================

const std::size_t DEFAULT_ROCKDVR_bits_per_block_offset = 0;
const std::size_t DEFAULT_ROCKDVR_bits_per_nonzero_count = 8;
const std::size_t DEFAULT_ROCKDVR_bits_per_gamma_range = 6;
const std::size_t DEFAULT_ROCKDVR_unquantized_gamma_bits = 24;
const std::size_t DEFAULT_ROCKDVR_bits_per_avg = 12;
const std::size_t DEFAULT_ROCKDVR_bits_per_index_gamma_pair = 16;
const std::size_t DEFAULT_ROCKDVR_bits_per_dictionary_element = 16; // for quantization of dict
const std::size_t DEFAULT_ROCKDVR_bits_per_block_header=32;
const std::size_t DEFAULT_ROCKDVR_bits_per_brick_header=4 * 32;

// ===========================================================================
// Helper
// ===========================================================================

template <class T>
static inline T cube(T x) { return x*x*x; }

// ===========================================================================
// Program arguments
// ===========================================================================

// Parameters

// Input & output volume
static std::string arg_input_file = "";
static std::string arg_output_file = "";
static std::string arg_output_delta_file = "";
static std::string arg_input_dictionary_file ="";
static std::string arg_output_dictionary_file ="";

static int         arg_block_size = 6;
const int	   default_apron_size = 0;

// Coreset
static std::size_t arg_max_coreset_scalar_count = 512*512*512;

// ... General params
const  bool  default_is_average_encoded_separately = true;
static int   arg_dictionary_element_count = 1024;
static int   arg_max_non_zeros = 6;
static float arg_tolerance = 0.0f;
static float arg_target_sparsity_with_variable_rate = 0.0f;
static int   arg_epoch_count = 100;

// Coding
const  std::size_t default_coding_budget_optimizer_group_size1d = 1;
static std::string arg_coding_budget_optimizer_method_string = "fixed";
static int         arg_brick_multiblock_count1d = 4; // How many blocks per brick (1d)
static int         arg_page_brick_count1d = 4;  // Home manu bricks per page (1d)

// Rockdvr stuff
static bool  arg_is_quantization_enabled = false;
static std::size_t arg_quantization_index_value_bits = 16;
static std::string arg_quantization_method = "q-mtv"; // ex rockdvr-cag

// ===========================================================================
// CODING
// ===========================================================================

bool is_fixed_sparsity() {
  return
    (arg_coding_budget_optimizer_method_string == "fixed") &&
    (arg_tolerance == 0.0f) &&
    (arg_target_sparsity_with_variable_rate == 0.0f);
}

static void minmax_in(float *xmin,
		      float *xmax,
		      const std::vector<float>& x,
		      std::size_t i_begin,
		      std::size_t i_end) {
  if (i_begin>=i_end) {
    *xmin=0.0f;
    *xmax=0.0f;
  } else {      
    *xmin=x[i_begin];
    *xmax=*xmin;
    for (std::size_t i=i_begin+1; i<i_end; ++i) {
      *xmin=std::min(*xmin, x[i]);
      *xmax=std::max(*xmax, x[i]);
    }
  }
}


inline std::size_t coded_index_bits(std::size_t /*bitcount*/, std::size_t dictionary_element_count) {
  std::size_t result=1;
  while ((std::size_t(1)<<result) < dictionary_element_count) ++result;
  return result;
}

inline std::size_t coded_gamma_bits(std::size_t bitcount, std::size_t dictionary_element_count) {
  return bitcount - coded_index_bits(bitcount, dictionary_element_count);
}

// ===========================================================================

static void show_usage(int /*argc*/, char** argv) {
  std::cerr
    << argv[0] << " Version " << TDM_SPARSE_CODING_VERSION_MAJOR << "."
    << TDM_SPARSE_CODING_VERSION_MINOR << "."
    << TDM_SPARSE_CODING_VERSION_PATCH << std::endl 
    << "Usage:" << std::endl
    << argv[0] << " [options] input-file" << std::endl  <<
    "General options:\n"
    "  -o output file              (output file for decoded volume)\n"
    "  -d delta  file              (output file for difference volume)\n"
    "  -b block_size               (default 6)\n"
    "  -C Mvalues                  (max coreset size - default 512^3)\n"
    "  -D dict size                (default 1024)\n"
    "  -K sparsity                 (default 6)\n"
    "  -T tolerance                (k variable if != 0.0 otherwise fixed - default 0.0)\n"
    "  -a sparsity                 (find tolerance that produce average sparsity, with variable approach - default 0.0)\n"
    "  -I training epochs          (default 100)\n"
    "  -e encoding method          (fixed|greedygrow|none - default fixed)\n"
    "  -B brick_multiblock_count1d (default 4)\n"
    "  -P page_brick_count1d       (default 4)\n"
    "  -l input-dictionary-name    load dictionary\n"          
    "  -s output-dictionary-name   save dictionary\n"
    "DS specific options\n"
    "Quantization\n"
    "  -q quantization method      (none | covra | q-esc | q-mtv - default q-mtv)\n"
    "  -Q nbits                    (0: no quantization; 1: use 24 bits for index-value pairs >1: use specified bits)\n";
}

int parse_options(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "ho:d:b:C:D:K:T:I:e:B:P:Q:q:a:l:s:")) != -1) {
    switch (opt) {
    case 'h': {
      show_usage(argc, argv);
      exit(0); // FIXME
    } break;
    case 'o': {
      arg_output_file = optarg;
      break;
    }
    case 'd': {
      arg_output_delta_file = optarg;
      break;
    }
    case 'l': {
      arg_input_dictionary_file = optarg;
      break;
    }
    case 's': {
      arg_output_dictionary_file = optarg;
      break;
    }
    case 'b': {
      arg_block_size  = atoi(optarg);
    } break;
    case 'C': {
      arg_max_coreset_scalar_count = std::size_t(atof(optarg)*1024*1024);
      break;
    }
    case 'D': {
      arg_dictionary_element_count = atoi(optarg);
    } break;
    case 'K': {
      arg_max_non_zeros = atoi(optarg);
    } break;
    case 'T': {
      arg_tolerance = atof(optarg);
      break;
    }
    case 'a': {
      arg_target_sparsity_with_variable_rate = atof(optarg);
    } break;
    case 'I': {
      arg_epoch_count = atoi(optarg);
    } break;
    case 'e': {
      arg_coding_budget_optimizer_method_string = optarg;
    } break;
    case 'B': {
      arg_brick_multiblock_count1d = atoi(optarg);
    } break;
    case 'P': {
      arg_page_brick_count1d = atoi(optarg);
    } break;
    case 'q': {
      arg_quantization_method = optarg;
      if (arg_quantization_method == "none") {
	arg_is_quantization_enabled = false;
	arg_quantization_index_value_bits = 0;
      } else if (arg_quantization_method == "covra") {
	arg_is_quantization_enabled = true;
	arg_quantization_index_value_bits = 24;
      } else if (arg_quantization_method == "q-esc") {
	arg_is_quantization_enabled = true;
	arg_quantization_index_value_bits = 16;
      } else if (arg_quantization_method == "q-mtv") {
	arg_is_quantization_enabled = true;
	arg_quantization_index_value_bits = 16;
      } else {	
	std::cerr << "Unknown quantization method: " << arg_quantization_method << std::endl;
	show_usage(argc, argv);
	return 1;
      }
    } break;
    case 'Q': { // AFTER -q ON COMMAND LINE to ovverride bits
      arg_quantization_index_value_bits = atoi(optarg);
      if (arg_quantization_index_value_bits==1) arg_quantization_index_value_bits=24;
      arg_is_quantization_enabled = (arg_quantization_index_value_bits > 0);
    } break;
    default:
      std::cerr << "Unknown option: -" << (char)opt << std::endl;
      show_usage(argc, argv);
      return 1;
    }
  }
  if (optind >= argc) {
    std::cerr << "Missing input filenames." << std::endl;
    show_usage(argc, argv);
    return 1;
  } else if (optind+1<argc) {
    std::cerr << "Too many input filenames." << std::endl;
    show_usage(argc, argv);
    return 1;
  }
  arg_input_file = std::string(argv[optind]);

  if (arg_quantization_method=="none") {
    // Use default bits for quantizing (24 default value)
    arg_is_quantization_enabled = false;

    std::size_t index_bits = coded_index_bits(0, arg_dictionary_element_count);
    arg_quantization_index_value_bits=((DEFAULT_ROCKDVR_unquantized_gamma_bits + index_bits + 15) / 16) * 16;
  } else if (!arg_is_quantization_enabled) {
    arg_quantization_method="none";
  }
  
  return 0;
}

// -- The following routines have the following characteristics
// --   Number of distinct values: L = 2^QBITS
// --   Stepsize = delta = (xmax-xmin)/L
// --   Max reconstruction error: delta/2
// --   Range in: (xmin... xmax)
// --   Range out: (xmin+delta/2 ... xmax-delta/2)

static inline sl::uint32_t quantize(float x,
				    float xmin,
				    float xmax,
				    std::size_t QBITS) {
  int L = (1<<QBITS); // Number of intervals
  double delta = (double(xmax)-double(xmin))/double(L);
  int I = delta ? sl::median(int(trunc((x-xmin)/delta)),0,L-1) : 0;
  return sl::uint32_t(I);
}

static inline float dequantize(sl::uint32_t qv,
			       float xmin,
			       float xmax,
			       std::size_t QBITS) {
  int L = (1<<QBITS); // Number of intervals
  double delta = (double(xmax)-double(xmin))/double(L);
  return xmin+(float(qv)+0.5f)*delta;
}

static inline float quantize_dequantize(float x,
					float xmin,
					float xmax,
					std::size_t QBITS) {
  return dequantize(quantize(x, xmin, xmax, QBITS), xmin, xmax, QBITS);
}

// -- The quantizer maps the input range (xmin-xmax) to (xmin+0.5*delta, xmax-0.5*delta)
// -- If one wants to ensure perfect reconstruction of xmin, xmax (at the expense of slightly
// -- higher errors of other values, it must extend the interval by a factor 0.5*(xmax-xmin)/(L-1)

static inline float quantize_two_side_expansion_delta(float xmin,
						      float xmax,
						      std::size_t QBITS) {
  int L = (1<<QBITS); // Number of intervals
  double delta = L<=1 ? 0.0 : (double(xmax)-double(xmin))/double(L-1);
  return 0.5f*float(delta);
}

// Case in which we expand only one endpoint
static inline float quantize_one_side_expansion_delta(float xmin,
						      float xmax,
						      std::size_t QBITS) {
  int L = (1<<QBITS); // Number of intervals
  double delta = L<=1 ? 0.0 : (double(xmax)-double(xmin))/double(L-0.5);
  return 0.5f*float(delta);
}


std::size_t extra_bits_per_gamma_range() {
  std::size_t bits_per_gamma_range = 0;
  if (arg_quantization_method == "q-mtv") {
    std::size_t used_block_header_bits = DEFAULT_ROCKDVR_bits_per_avg;
    std::size_t available_bits = DEFAULT_ROCKDVR_bits_per_block_header - used_block_header_bits;
    std::size_t non_zeros_encodable_count = arg_max_non_zeros+1;
    std::size_t bits_per_nonzero_count = (is_fixed_sparsity() ?
					  std::size_t(0) :
					  coded_index_bits(available_bits, non_zeros_encodable_count));
    
    used_block_header_bits += bits_per_nonzero_count;
    
    if (used_block_header_bits > DEFAULT_ROCKDVR_bits_per_block_header) {
      std::cerr << "ERROR: bits required per block header " << used_block_header_bits
		<< " > 32 available bits" << std::endl;
      exit(1);
    }
    bits_per_gamma_range = (DEFAULT_ROCKDVR_bits_per_block_header - used_block_header_bits)/2;
  } else {
    bits_per_gamma_range = DEFAULT_ROCKDVR_bits_per_gamma_range;
  } 
  return bits_per_gamma_range;
}
// ===========================================================================
void quantize_dequantize_in(std::vector<float>&             dequantized_page_y_avg,
			    std::vector<float>&             dequantized_gamma_val,
			    std::vector<std::size_t>&       dequantized_gamma_idx,
			    std::vector<std::size_t>&       dequantized_gamma_permutation,			    
			    const std::vector<float>&       page_y_avg,
			    const std::vector<std::size_t>& gamma_offset,
			    const std::vector<std::size_t>& gamma_sz,
			    const std::vector<float>&       gamma_val,
			    const std::vector<std::size_t>& gamma_idx) {
  const std::size_t N = page_y_avg.size();
  
  dequantized_page_y_avg = page_y_avg;
  dequantized_gamma_val  = gamma_val;
  dequantized_gamma_idx  = gamma_idx;
  
  // Permutation of representations by decreasing gamma value
  dequantized_gamma_permutation.resize(gamma_idx.size());
  for (std::size_t b=0; b<N; ++b) {
    for (std::size_t gb=0; gb<gamma_sz[b]; ++gb) {
      dequantized_gamma_permutation[gamma_offset[b]+gb] = gb;
    }
    // Sort by val
    for (std::size_t i=1; i<gamma_sz[b]; ++i) {
      for (std::size_t j=i; (j>0 && dequantized_gamma_val[gamma_offset[b]+j-1]<dequantized_gamma_val[gamma_offset[b]+j]); --j) {
	std::swap(dequantized_gamma_val[gamma_offset[b]+j-1],dequantized_gamma_val[gamma_offset[b]+j]);
	std::swap(dequantized_gamma_idx[gamma_offset[b]+j-1],dequantized_gamma_idx[gamma_offset[b]+j]);
	std::swap(dequantized_gamma_permutation[gamma_offset[b]+j-1],dequantized_gamma_permutation[gamma_offset[b]+j]);
      }
    }
  }
  
  // ROCKDVR Quantization + dequantization
  if (arg_is_quantization_enabled) {
    // Quantization bits
    //const std::size_t index_bits=coded_index_bits(arg_quantization_index_value_bits, arg_dictionary_element_count);
    const std::size_t gamma_bits=coded_gamma_bits(arg_quantization_index_value_bits, arg_dictionary_element_count);
    const std::size_t average_bits = DEFAULT_ROCKDVR_bits_per_avg;
    //    std::cerr << "index_bits " << index_bits << ", gamma_bits " << gamma_bits << std::endl;
    const std::size_t BRICK_SIZE = cube(arg_brick_multiblock_count1d*default_coding_budget_optimizer_group_size1d);
    // Block size
    for (std::size_t i_begin=0; i_begin<N; i_begin+=BRICK_SIZE) {
      std::size_t i_end=std::min(i_begin+BRICK_SIZE, N);
      
      float avg_min, avg_max;
      minmax_in(&avg_min, &avg_max, page_y_avg, i_begin, i_end);
      for (std::size_t i=i_begin; i<i_end; ++i) {
	dequantized_page_y_avg[i] = quantize_dequantize(page_y_avg[i],
							avg_min, avg_max, average_bits);
      }

      std::size_t brick_gamma_begin=gamma_offset[i_begin];
      std::size_t brick_gamma_count=0;
      for (std::size_t i=i_begin; i<i_end; ++i) {
	brick_gamma_count+= gamma_sz[i];
      }
      std::size_t brick_gamma_end=brick_gamma_begin+brick_gamma_count;
      
      float gamma_min, gamma_max;
      minmax_in(&gamma_min, &gamma_max, gamma_val, brick_gamma_begin, brick_gamma_end);

      if (arg_quantization_method == "covra") {
	// Quantization of gamma per brick
	for (std::size_t i=brick_gamma_begin; i<brick_gamma_end; ++i) {
	  dequantized_gamma_val[i] = quantize_dequantize(gamma_val[i],
							 gamma_min, gamma_max, gamma_bits);
	}
      } else {
	// Double quantization (per brick + per block)
	std::size_t bits_per_gamma_range = extra_bits_per_gamma_range();

	bool is_rockdvr_cag = false;
	if (arg_quantization_method=="q-esc") {
	  // nothing to do
	} else if (arg_quantization_method=="q-mtv") {
	  bits_per_gamma_range += gamma_bits;
	  is_rockdvr_cag=true;
	} 
	for (std::size_t i=i_begin; i<i_end; ++i) {
	  if (gamma_sz[i]) {
	    float block_gamma_min, block_gamma_max;
	    minmax_in(&block_gamma_min, &block_gamma_max,
		      gamma_val, gamma_offset[i], gamma_offset[i]+gamma_sz[i]);
	    // Ensure we dequantize extremes at max precision
	    float delta = quantize_two_side_expansion_delta(block_gamma_min, block_gamma_max, bits_per_gamma_range);
	    block_gamma_min -= delta;
	    block_gamma_max += delta;
	      
	    float dequantized_block_gamma_min = quantize_dequantize(block_gamma_min,
								    gamma_min, gamma_max, bits_per_gamma_range);
	    float dequantized_block_gamma_max = quantize_dequantize(block_gamma_max,
								    gamma_min, gamma_max, bits_per_gamma_range);
	    for (std::size_t gi=0; gi<gamma_sz[i]; ++gi) {
	      const std::size_t gi_orig = dequantized_gamma_permutation[gamma_offset[i]+gi];
	      if (is_rockdvr_cag) {
		if (gi==0) { // largest
		  dequantized_gamma_val[gamma_offset[i]+gi] = dequantized_block_gamma_max;
		} else if (gi+1==gamma_sz[i]) { // smallest
		  dequantized_gamma_val[gamma_offset[i]+gi] = dequantized_block_gamma_min;
		} else { // others
		  // Try to guess a narrower quantization interval
		  float gmin = dequantized_block_gamma_min;
		  float gmax = dequantized_gamma_val[gamma_offset[i]+gi-1];
		  float delta = quantize_one_side_expansion_delta(gmin, gmax, gamma_bits);
		  gmax = std::min(gmax+delta,dequantized_block_gamma_max);
		  dequantized_gamma_val[gamma_offset[i]+gi] = quantize_dequantize(gamma_val[gamma_offset[i]+gi_orig],
										  gmin,
										  gmax,
										  gamma_bits);
		}
	      } else {
		// Eurovis version
		dequantized_gamma_val[gamma_offset[i]+gi] = quantize_dequantize(gamma_val[gamma_offset[i]+gi_orig],
										dequantized_block_gamma_min,
										dequantized_block_gamma_max,
										gamma_bits);
	      }
	    }
	  }
	}
      }
    } // for each brick
  } // if quantize
}

// ===========================================================================

void approximate_in(std::vector<Eigen::VectorXf>& page_y_prime,
		    const std::vector<Eigen::VectorXf>& page_y,
		    std::size_t* page_gamma_count,
		    std::size_t* page_gamma_min_nonzeros,
		    std::size_t* page_gamma_max_nonzeros,
		    vic::tdm_sparse_coding::dictionary_coding_budget_optimizer& cbo) {
  // Remove average if needed
  const std::size_t N = page_y.size();

  std::vector<float> page_y_avg(N, 0.0f);
  std::vector<Eigen::VectorXf> page_y_hat(N);

  // Remove average if needed
  for (std::size_t k=0; k<N; ++k) {
    Eigen::VectorXf y_hat = page_y[k];
    if (default_is_average_encoded_separately) {
      page_y_avg[k] = y_hat.sum()/float(y_hat.size());
      y_hat.array() -= page_y_avg[k];
    }
    page_y_hat[k] = y_hat;
  }
	  
  // Encode --  cbo.approximate_in(page_y_prime,  page_y_hat, page_gamma_count);

  std::vector<std::size_t> gamma_offset;
  std::vector<std::size_t> gamma_sz;
  std::vector<float>       gamma_val;
  std::vector<std::size_t> gamma_idx;
  
  cbo.optimize_in(gamma_offset, gamma_sz, gamma_val, gamma_idx, page_y_hat);
  if (page_gamma_count) *page_gamma_count = gamma_idx.size();

  // Quantization
  std::vector<float>       dequantized_page_y_avg;
  std::vector<float>       dequantized_gamma_val;
  std::vector<std::size_t> dequantized_gamma_idx;
  std::vector<std::size_t> dequantized_gamma_permutation;
  quantize_dequantize_in(dequantized_page_y_avg,
			 dequantized_gamma_val,
			 dequantized_gamma_idx,
			 dequantized_gamma_permutation,
			 page_y_avg,
			 gamma_offset, gamma_sz, gamma_val, gamma_idx);
  
  // Decoding
  *page_gamma_max_nonzeros = 0;
  *page_gamma_min_nonzeros = std::size_t(-1);
  
  page_y_prime.resize(N, page_y_hat[0]);
  for (std::size_t r=0; r<N; ++r) {
    std::size_t i_r=gamma_offset[r];
    std::size_t S_r = gamma_sz[r];
    cbo.decode_in(page_y_prime[r],
		  &(dequantized_gamma_idx[i_r]),
		  &(dequantized_gamma_val[i_r]),
		  S_r);

    (*page_gamma_min_nonzeros) = std::min((*page_gamma_min_nonzeros), S_r);
    (*page_gamma_max_nonzeros) = std::max((*page_gamma_max_nonzeros), S_r);
  }

  // Add back average if needed
  if (default_is_average_encoded_separately) {
    for (std::size_t k=0; k<N; ++k) {
      page_y_prime[k] = page_y_prime[k].array()+dequantized_page_y_avg[k];
    }
  }

#if 1
  // DEBUG: PRINT HIGH ERRORS
  for (std::size_t r=0; r<N; ++r) {
    Eigen::VectorXf dy = page_y[r]-page_y_prime[r];
    double max_e = std::sqrt(double(dy.array().square().maxCoeff()));
    if (max_e > 1.2) {
      std::cerr << std::endl;
      std::cerr << "====== ERR: " << max_e << std::endl;
      std::cerr << "avg: " << page_y_avg[r] << " qavg: " << dequantized_page_y_avg[r] << std::endl;
      std::size_t i_r=  gamma_offset[r];
      std::size_t S_r = gamma_sz[r];
      for (std::size_t gb=0; gb<S_r; ++gb) {
	std::size_t gb_orig = dequantized_gamma_permutation[i_r+gb];
	std::cerr <<
	  " idx:" << gamma_idx[i_r+gb_orig] << " val:" << gamma_val[i_r+gb_orig] <<
	  " qidx:" << dequantized_gamma_idx[i_r+gb] << " qval:" << dequantized_gamma_val[i_r+gb] <<
	  std::endl;
      }
      std::cerr << std::endl;
    }
  }
  
#endif
}

// ===========================================================================
// APRON AND ERRORS
// ===========================================================================
static Eigen::VectorXf without_apron(const Eigen::VectorXf& yy) {
  int block_width = default_apron_size + arg_block_size + default_apron_size;

  Eigen::VectorXf result(cube(arg_block_size));
  for(int z = 0; z < arg_block_size; ++z) {
    int iz = default_apron_size + z;
    for(int y = 0; y < arg_block_size; ++y) {
      int iy = default_apron_size + y;
      for(int x = 0; x < arg_block_size; ++x) {
	int ix = default_apron_size + x;
	int offset_with_apron = ix + block_width * (iy + block_width * iz);
	int offset_without_apron = x + arg_block_size * (y + arg_block_size * z);
	result[offset_without_apron] = yy[offset_with_apron];
      }
    }
  }
  return result;
}

void average_vector_errors_in(float& inner_voxel_error,
			      float& face_voxel_error,
			      float& edge_voxel_error,
			      float& corner_voxel_error,
			      std::size_t block_size,
			      std::size_t apron_size,
			      const Eigen::VectorXd& last_avg_e2_v) {
  // average together errors of voxels of same position type
  // Position type
  // 0 boundary: inner
  // 1 boundary: face
  // 2 boundary: edge
  // 3 boundary: corner
  int block_width = apron_size + block_size + apron_size;
  int voxel_counter[4] = {0,0,0,0};
  float voxel_error[4] = {0,0,0,0};
  for(int z = 0; z < int(block_size); ++z) {
    int iz = apron_size + z;
    int is_z_boundary = (z == 0 || z == int(block_size)-1) ? 1 : 0;
    for(int y = 0; y < int(block_size); ++y) {
      int iy = apron_size + y;
      int is_y_boundary = (y == 0 || y == int(block_size)-1) ? 1 : 0;
      for(int x = 0; x < int(block_size); ++x) {
	int ix = apron_size + x;
	int offset = ix + block_width * (iy + block_width * iz);
	int is_x_boundary = (x == 0 || x == int(block_size)-1) ? 1 : 0;
	int voxel_type = is_x_boundary + is_y_boundary + is_z_boundary;
	if (offset >= last_avg_e2_v.size()) {
	  SL_TRACE_OUT(-1) << "Accessing wrong index " << offset << " >= " << last_avg_e2_v.size() << std::endl;
	} else {
	  voxel_error[voxel_type] += last_avg_e2_v[offset];
	  ++voxel_counter[voxel_type];
	}
      }
    }
  }
 
  inner_voxel_error  = voxel_counter[0] > 0 ? voxel_error[0] / float(voxel_counter[0]) : 0;
  face_voxel_error   = voxel_counter[1] > 0 ? voxel_error[1] / float(voxel_counter[1]) : 0;
  edge_voxel_error   = voxel_counter[2] > 0 ? voxel_error[2] / float(voxel_counter[2]) : 0;
  corner_voxel_error = voxel_counter[3] > 0 ? voxel_error[3] / float(voxel_counter[3]) : 0;
}
// ==========================================================================

static float estimated_tolerance_for_target_sparsity_with_variable_rate(vic::tdm_sparse_coding::dictionary_coder& coder,
									const std::vector<float>& W,
									const std::vector<Eigen::VectorXf>& Y,
									float target_sparsity,
									std::size_t max_non_zeros_per_signal,
									float zero_fraction) {
  std::cerr << "----------------------------------------------------------" << std::endl;
  std::cerr << "TOLERANCE COMPUTATION" << std::endl;
  std::cerr << "Target sparsity   : " << target_sparsity << std::endl;

  sl::real_time_clock ck; ck.restart();

  // Build multiple choice knapsack problem
  std::size_t group_size = cube(default_coding_budget_optimizer_group_size1d);
  std::size_t Signal_Count = Y.size();
  std::size_t MCK_Slot_Count = Signal_Count / group_size;
  if (MCK_Slot_Count * group_size < Signal_Count) ++MCK_Slot_Count; // Last slot partially empty!

  // Create incremental orthogonal matching pursuit optimizers
  std::vector<vic::tdm_sparse_coding::incremental_sparse_coder*> iomp(Signal_Count);
  std::vector<std::size_t> solution_nzc(MCK_Slot_Count);
  std::vector<double>      solution_err2(MCK_Slot_Count);
  
  typedef std::pair<float, std::size_t> error_index_t;
  std::priority_queue<error_index_t> err2_index_priority_queue;

  for (std::size_t g=0; g<MCK_Slot_Count; ++g) {
    std::size_t i_bgn = g*group_size;
    std::size_t i_end = std::min(Signal_Count, i_bgn+group_size);
    solution_nzc[g] = 0;
    solution_err2[g] = 0.0;
    for (std::size_t i=i_bgn; i<i_end; ++i) {
      iomp[i] = coder.new_incremental_sparse_coder(Y[i]);
      solution_err2[g] += iomp[i]->err2(solution_nzc[g]);
    }
    if (solution_nzc[g]<max_non_zeros_per_signal) {
      err2_index_priority_queue.push(std::make_pair(solution_err2[g], g));
    }
  }

  double total_coreset_weight_sum = 0;
  double current_weighted_sum = 0;
  for(std::size_t i = 0; i < Signal_Count; ++i) {
    total_coreset_weight_sum += W[i];
    // current_weighted_sum += bps_zero * W[i]; // add if bps(0) 
  }
  double target_sum = total_coreset_weight_sum * target_sparsity / (1.0f - zero_fraction);
  std::size_t saturated_count = 0;
  while ((current_weighted_sum < target_sum) && !err2_index_priority_queue.empty()) {
    error_index_t x = err2_index_priority_queue.top();
    err2_index_priority_queue.pop();
    std::size_t best_g = x.second;

    // Increase nonzero count of best group
    current_weighted_sum += W[best_g];
    ++solution_nzc[best_g];
    std::size_t i_bgn = best_g*group_size;
    std::size_t i_end = std::min(Signal_Count, i_bgn+group_size);
    solution_err2[best_g] = 0;
    for (std::size_t i=i_bgn; i<i_end; ++i) {
      solution_err2[best_g] += iomp[i]->err2(solution_nzc[best_g]);
    }
    if (solution_nzc[best_g]<max_non_zeros_per_signal) {
      err2_index_priority_queue.push(std::make_pair(solution_err2[best_g], best_g));
    } else {
      ++saturated_count;
    }
  }
  
  // Cleanup
  for (std::size_t i=0; i<Signal_Count; ++i) {
    delete iomp[i]; iomp[i] = 0;
  }
  iomp.clear();

  double eps = 0.0;
  if (!err2_index_priority_queue.empty()) {
    // The queue is not empty -- take the tolerance as the current max error
    eps = std::sqrt(err2_index_priority_queue.top().first);
  } else {
    // The queue is empty -- this means that all the signals are overflowing!
    std::cerr << "CAPACITY OVERFLOW" << std::endl;
    for(std::size_t i = 0; i < Signal_Count; ++i) {
      eps = std::max(eps, solution_err2[i]);
    }
    eps = std::sqrt(eps);
  }
      
  std::cerr << "Tolerance         : " << eps << std::endl;
  std::cerr << "Estimated sparsity: " << current_weighted_sum / total_coreset_weight_sum << std::endl;
  std::cerr << "Saturated count   : " << saturated_count << "/" << Signal_Count << " ( "<< sl::human_readable_percent(100.0*saturated_count/double(Signal_Count)) << ")" << std::endl;
  std::cerr << "Time elapsed      : " << sl::human_readable_duration(ck.elapsed()) << std::endl;
  std::cerr << "----------------------------------------------------------" << std::endl;

  return eps;
}

// ===========================================================================
// Estimate Coreset
// Learn Dictionary
// Encode/Decode data
// Save decoded data and/or encoding errors
void train_eval_save(vic::vol::volume_tile_stream* vts,
		     vic::tdm_sparse_coding::streaming_coreset_builder& scb,
		     vic::tdm_sparse_coding::dictionary_coder& coder,
		     vic::tdm_sparse_coding::dictionary_coding_budget_optimizer* cbo,
		     vic::vol::volume_tile_stream* vts_w,
		     vic::vol::volume_tile_stream* vts_dw) {
  std::vector<float>           W;
  std::vector<Eigen::VectorXf> Y;
  std::vector<float>           W_y_tol;
  std::vector<Eigen::VectorXf> Y_y_tol;
  float                        zero_fraction = 0.0f;
      
  sl::real_time_clock global_ck;   global_ck.restart();
  sl::real_time_clock ck;
  
  sl::uint64_t volume_size = cube(vts->block_size_without_apron())*vts->count();
  std::cerr <<
    "==========================================================" << std::endl <<
    "Volume" << std::endl <<
    "----------------------------------------------------------" << std::endl <<
    "Size           : " <<  sl::human_readable_quantity(volume_size) << "voxels" << std::endl <<
    "==========================================================" << std::endl;

  std::cerr << "==========================================================" << std::endl;
  std::cerr << "CORESET EXTRACTION" << std::endl;
  std::cerr <<  "Coreset Size    : " <<  sl::human_readable_quantity(arg_max_coreset_scalar_count) << "voxels" << std::endl;
  std::cerr << "==========================================================" << std::endl;
  std::cerr << "CORESET FOR TRAINING" << std::endl;
  ck.restart();
  scb.build(*vts);
  scb.extract_weights_and_signals_in(W,Y);
  double stats_coreset_time_s = ck.elapsed().as_milliseconds()/1000.0;

  double stats_second_coreset_computation = 0;
  if (arg_target_sparsity_with_variable_rate != 0.0f) {
    std::cerr << "CORESET FOR SIZE ESTIMATION" << std::endl;
    const std::size_t block_scalar_size = sl::ipow(arg_block_size,3); // coreset size independent from apron
    const std::size_t coreset_blocks = arg_max_coreset_scalar_count / block_scalar_size; // FIXME
    std::cerr << "CORESET EXTRACTION" << std::endl;
    std::cerr <<  "Coreset Size    : " <<  sl::human_readable_quantity(coreset_blocks * block_scalar_size) << "voxels" << std::endl;
    std::cerr << "----------------------------------------------------------" << std::endl;

    sl::real_time_clock ck_inner;
    ck_inner.restart();

    vic::tdm_sparse_coding::streaming_coreset_builder_uniform scb_for_sparsity;
    scb_for_sparsity.set_desired_coreset_size(coreset_blocks+1);
    scb_for_sparsity.set_is_average_removed(default_is_average_encoded_separately);
    scb_for_sparsity.set_are_partially_outside_blocks_skipped(false);
    scb_for_sparsity.build(*vts);
    scb_for_sparsity.extract_weights_and_signals_in(W_y_tol,Y_y_tol);
    
    if (scb_for_sparsity.in_count() == 0) {
      std::cerr << "Unable to find coreset. Exiting" << std::endl;
      exit(1);
    }
    stats_second_coreset_computation = ck_inner.elapsed().as_milliseconds() / 1000.0;
    zero_fraction = float(scb_for_sparsity.zero_y_count()) / float(scb_for_sparsity.in_count());
  }

#if 1
  std::cerr << "==========================================================" << std::endl;
  std::cerr << "RESCALING WEIGHTS" << std::endl;
  double wsum = 0.0;
  double wcount = 0.0;
  for (std::size_t i=0; i<W.size(); ++i) {
    for (std::size_t j=0; int(j)<Y[0].size(); ++j) {
      wsum += std::sqrt(double(W[i]))*std::abs(double(Y[i][j]));
      wcount += 1.0;
    }
  }
  double wavg = wsum/wcount;
  double wscale = 1.0/(wavg*wavg);
  for (std::size_t i=0; i<W.size(); ++i) {
    W[i] *= wscale;
  }
  std::cerr << "SCALE=" << wscale << std::endl;
#endif
  
  std::cerr << "==========================================================" << std::endl;
  std::cerr << "TRAINING ON CORESET" << std::endl;
  if (default_is_average_encoded_separately) {
    std::cerr << "(Performed on zero mean vectors -- ERROR STATS NOT MEANINGFUL DURING TRAINING)" << std::endl;
  }
  std::cerr << "==========================================================" << std::endl;

  ck.restart();
  coder.train_on_coreset(W,Y);
  
  double stats_training_time_s = ck.elapsed().as_milliseconds()/1000.0;
  std::cerr << "==========================================================" << std::endl;
    
  ck.restart();
  if (arg_target_sparsity_with_variable_rate != 0.0f) {
    std::cerr << "==========================================================" << std::endl;
    std::cerr << "COMPUTING TOLERANCE FOR TARGET SPARSITY " << arg_target_sparsity_with_variable_rate << std::endl;
    std::cerr << "----------------------------------------------------------" << std::endl;
    ck.restart(); // We restart here to remove useless coreset extraction time 
    
    float eps =estimated_tolerance_for_target_sparsity_with_variable_rate(coder, W_y_tol, Y_y_tol,
                                                                          arg_target_sparsity_with_variable_rate,
                                                                          arg_max_non_zeros,
                                                                          zero_fraction);

    std::cerr << "EPS = " << eps << std::endl;

    if (cbo) cbo->set_non_zeros_per_signal(arg_max_non_zeros); // Var rate: this is the max non zeros
    coder.set_tolerance(eps);
    std::cerr << "==========================================================" << std::endl;
  }
  double stats_tolerance_computation_time_s = ck.elapsed().as_milliseconds()/1000.0;

  
  if (cbo) cbo->set_coder(&coder);

  std::cerr << "==========================================================" << std::endl;
  std::cerr << "ERROR EVALUATION ON FULL DATA" << std::endl;
  if (vts_w) {
    std::cerr << "*** SAVING DECODING OF ENCODED VOLUME" << std::endl;
  }
  if (vts_dw) {
    std::cerr << "*** SAVING DELTA ORIGINAL-DECODING OF ENCODED VOLUME" << std::endl;
  }

  std::cerr << "CODING BUDGET OPTIMIZATION METHOD: " << arg_coding_budget_optimizer_method_string << std::endl;
  std::cerr << "==========================================================" << std::endl;
  std::size_t page_count = 0;
  if (arg_coding_budget_optimizer_method_string != "none") {
    std::size_t gamma_count = 0;
    std::size_t gamma_min_nonzeros=std::size_t(-1);
    std::size_t gamma_max_nonzeros=0;
    std::size_t vector_count = 0;
    std::size_t scalar_count = 0;
    
    ck.restart();
    vic::tdm_sparse_coding::error_evaluator ee;
    ee.begin(); {
      const std::size_t N_threads = omp_get_num_procs();
      
      const std::size_t BRICK_SIZE = cube(arg_brick_multiblock_count1d*default_coding_budget_optimizer_group_size1d);
      const std::size_t PAGE_SIZE = cube(arg_page_brick_count1d)*BRICK_SIZE;
      const std::size_t PARALLEL_PAGE_COUNT = 2*N_threads;
      std::size_t progress_counter = 0;
      
      if (vts_w) vts_w->restart();
      if (vts_dw) vts_dw->restart();
      vts->restart();
      while (!vts->off()) {
	// Read max PARALLEL_PAGE_COUNT pages of size PAGE_SIZE
	std::vector< std::vector<Eigen::VectorXf> > page_y;
	page_y.reserve(PARALLEL_PAGE_COUNT);
	
	while ((!vts->off()) && (page_y.size() < PARALLEL_PAGE_COUNT)) {
	  // Read page
	  page_y.push_back(std::vector<Eigen::VectorXf>());
	  while ((!vts->off()) && page_y.back().size() < PAGE_SIZE) {
	    Eigen::VectorXf y(vts->dimension());
	    vts->current_in(y);
	    vts->forth();
	    page_y.back().push_back(y);
	  }
	  if (page_y.back().size() != PAGE_SIZE) {
	    std::cerr << "WARNING: page_size = " << page_y.back().size() << " != " << PAGE_SIZE << std::endl;
	  }
	}
	page_count += page_y.size();

	// Prepare parallel encoding of pages
	const std::size_t N_PAGES = page_y.size();
	std::vector< std::vector<Eigen::VectorXf> > page_y_prime(N_PAGES);

	std::vector<vic::tdm_sparse_coding::error_evaluator> page_stats(N_PAGES);
	
	for (std::size_t p=0; p<N_PAGES; ++p) {
	  const std::size_t P = page_y[p].size();
	  Eigen::VectorXf y(vts->dimension());
	  page_y_prime[p] = std::vector<Eigen::VectorXf>(P,y);
	  
	  page_stats[p].set_is_verbose(false);

	  vector_count += P;
	  scalar_count += P * sl::ipow(arg_block_size,3); // input scalars
	}

	// Encode all pages in parallel
	std::size_t page_gamma_count[N_PAGES];
	std::size_t page_gamma_min_nonzeros[N_PAGES];
	std::size_t page_gamma_max_nonzeros[N_PAGES];

	// Note: as pages are large and might take a different
	// time each to encode, we use a dynamic schedule
#pragma omp parallel for schedule(dynamic)
	for (std::size_t p=0; p<N_PAGES; ++p) {
	  
	  approximate_in(page_y_prime[p],
			 page_y[p],
			 &page_gamma_count[p],
			 &page_gamma_min_nonzeros[p],
			 &page_gamma_max_nonzeros[p],
			 *cbo);
	  
	  // Eval errors
	  const std::size_t N = page_y[p].size();
	  for (std::size_t k=0; k<N; ++k) {
	    page_stats[p].put(without_apron(page_y[p][k]),
			      without_apron(page_y_prime[p][k]));
	  }
	}

	// Accumulate page stats and print error
	for (std::size_t p=0; p<N_PAGES; ++p) {
	  gamma_count+=page_gamma_count[p];
	  gamma_min_nonzeros = std::min(gamma_min_nonzeros, page_gamma_min_nonzeros[p]);
	  gamma_max_nonzeros = std::max(gamma_max_nonzeros, page_gamma_max_nonzeros[p]);
	  ee.accumulate(page_stats[p]);
	  progress_counter += page_stats[p].last_vector_count();
	}
	if (progress_counter>50000) {
	  ee.print_progress_at_put();
	  progress_counter = 0;
	}

	// Write out if needed
	if (vts_w || vts_dw) {
	  for (std::size_t p=0; p<N_PAGES; ++p) {
	    const std::size_t N = page_y[p].size();
	    for (std::size_t k=0; k<N; ++k) {
	      Eigen::VectorXf y_k = page_y[p][k];
	      Eigen::VectorXf y_tilde_k = page_y_prime[p][k];
	      Eigen::VectorXf delta_y_k = y_k-y_tilde_k;
	      if (vts_w) {
		vts_w->write_current(y_tilde_k);
		vts_w->forth();
	      }
	      if (vts_dw) {
		vts_dw->write_current(delta_y_k);
		vts_dw->forth();
	      }
	    }
	  }
	}
      } // while not off
    } // ee
    ee.end();
    double stats_coding_time_s = ck.elapsed().as_milliseconds()/1000.0;
    double stats_total_time_s = global_ck.elapsed().as_milliseconds()/1000.0;
    if (stats_second_coreset_computation) {
      std::cerr << "Removed from total time " << stats_second_coreset_computation << "s used for computing second coreset" << std::endl;
      stats_total_time_s -= stats_second_coreset_computation; // Remove computation of second coreset (which should be computed within the first corset computation
    }
    std::cerr << "==========================================================" << std::endl;
    if (vts_w) {
      std::cerr << "Encoded + decoded volume saved." << std::endl;
      std::cerr << "==========================================================" << std::endl;
    }
    if (vts_dw) {
      std::cerr << "Delta of encoded + decoded volume saved." << std::endl;
      std::cerr << "==========================================================" << std::endl;
    }

    // SUMMARY
    std::cout << "METHOD: " <<
      " Algorithm: " << "KSVD" <<
      " | Dictionary " << arg_dictionary_element_count <<
      std::endl;
    std::cout << "SUMMARY:" <<
      " Data_size: " << sl::human_readable_quantity(volume_size) << 
      " | Coreset_size: " << sl::human_readable_quantity(arg_max_coreset_scalar_count) << 
      " | Coreset_time: " << stats_coreset_time_s << 
      " | Training_time: " << stats_training_time_s  <<
      " | Tolerance_computation_time_s: " <<  stats_tolerance_computation_time_s << 
      " | Encoding_time: " << stats_coding_time_s <<
      " | Total_time: " << stats_total_time_s <<
      " | PSNR: " << ee.last_PSNR_dB() << std::endl;

    std::size_t ROCKDVR_index_bits=coded_index_bits(arg_quantization_index_value_bits, arg_dictionary_element_count);
    std::size_t ROCKDVR_gamma_bits=coded_gamma_bits(arg_quantization_index_value_bits, arg_dictionary_element_count);
    std::size_t byte_in_count = (scalar_count * vts->volume()->bps())/CHAR_BIT;
    std::size_t ROCKDVR_bits_per_dictionary_element = DEFAULT_ROCKDVR_bits_per_dictionary_element;
    std::size_t ROCKDVR_bits_per_index_gamma_pair = arg_quantization_index_value_bits; // ROCKDVR_gamma_bits + ROCKDVR_index_bits; // 
    std::size_t block_count = cube(arg_page_brick_count1d * arg_brick_multiblock_count1d * default_coding_budget_optimizer_group_size1d);
    std::size_t ROCKDVR_bits_per_nonzero_count = ((is_fixed_sparsity()) ? 0 : coded_index_bits(32, arg_max_non_zeros));
    std::size_t ROCKDVR_bits_per_avg = (default_is_average_encoded_separately ? DEFAULT_ROCKDVR_bits_per_avg : 0); // assume idx + val fit in 32 bit
    
    std::size_t ROCKDVR_bits_per_gamma_range = 2*extra_bits_per_gamma_range();
    std::size_t page_brick_count = cube(arg_page_brick_count1d);
    std::size_t brick_count = page_count * page_brick_count;
    std::size_t byte_out_count_volume = (vector_count * DEFAULT_ROCKDVR_bits_per_block_header +
					 gamma_count  * ROCKDVR_bits_per_index_gamma_pair +
					 brick_count * DEFAULT_ROCKDVR_bits_per_brick_header)/CHAR_BIT;
    std::size_t bits_per_fixed_page = (DEFAULT_ROCKDVR_bits_per_brick_header * page_brick_count +
				       DEFAULT_ROCKDVR_bits_per_block_header * block_count +
				       ROCKDVR_bits_per_index_gamma_pair * arg_max_non_zeros * block_count); // arg_quantization_index_value_bits 

    std::size_t byte_out_count_dictionary = (arg_dictionary_element_count * cube(arg_block_size) * ROCKDVR_bits_per_dictionary_element)/CHAR_BIT;
    std::size_t byte_out_count = byte_out_count_dictionary + byte_out_count_volume;

    std::size_t total_bits_fixed_page = bits_per_fixed_page * page_count + byte_out_count_dictionary * CHAR_BIT;
    float inner_voxel_error, face_voxel_error, edge_voxel_error, corner_voxel_error;
    average_vector_errors_in(inner_voxel_error, face_voxel_error, edge_voxel_error, corner_voxel_error,
			     arg_block_size, 0, ee.last_avg_e2_v()); // NOTE: error eval without apron
    
    std::cout << "INPUT  :" <<
      " nx= "<< vts->volume()->sample_counts()[0] <<
      " | ny= "<< vts->volume()->sample_counts()[1] <<
      " | nz= "<< vts->volume()->sample_counts()[2] <<
      " | range= " << ee.last_ymin() << ".." << ee.last_ymax() <<
      std::endl;
    std::cout << "LAYOUT :" <<
      " block= "      << "(" << arg_block_size << "+2*" << default_apron_size << ")^3 scalars" <<
      " | multiblock= " << default_coding_budget_optimizer_group_size1d << "^3 blocks" <<
      " | brick= "      << arg_brick_multiblock_count1d << "^3 multiblocks" << 
      " | page= "       << arg_page_brick_count1d << "^3 bricks" <<
      " | page_count= " << page_count <<
      std::endl;
    std::cout << "CODER :" <<
      " Tolerance= " << coder.tolerance() <<
      " | max_non_zeros= " << coder.max_non_zeros() <<
      " | dictionary = " << coder.signal_size() << "x" << coder.dictionary_element_count() <<
      std::endl;
    std::cout << "ENCODE :" <<
      " Quantization= " << arg_quantization_method <<
      " | bits_per_average= " << ROCKDVR_bits_per_avg <<
      " | bits_per_gamma_range= " << ROCKDVR_bits_per_gamma_range <<
      " | bits_per_nonzero_count= " << ROCKDVR_bits_per_nonzero_count <<
      " | bits_per_index_gamma= " << ROCKDVR_bits_per_index_gamma_pair <<
      " | bits_per_index= " << ROCKDVR_index_bits <<
      " | bits_per_gamma= " << ROCKDVR_gamma_bits <<
      " | bits_per_block_header= " << DEFAULT_ROCKDVR_bits_per_block_header <<
      " | bits_per_brick_header= " << DEFAULT_ROCKDVR_bits_per_brick_header <<
      " | bits_per_fixed_page= " << bits_per_fixed_page <<
      std::endl;
    
    // std::cerr << "DATA LAYOUT: ";
    // if (arg_quantization_method == "none") {
    //   std::cout << "NO QUANTIZATION" << std::endl;
    // } else {
    //   std::cout << " BRICK HEADER: float gamma-min, gamma-max, avg-min, avg-max ";
    //   if (arg_quantization_method == "q-mtv") {
    // 	std::cout << "| BLOCK HEADER uint32: [non-zero-count | avg(12) | bits-per-gamma-min | bits-per-gamma-max]" << std::endl;
    //   } else if (arg_quantization_method == "q-esc") {
    // 	std::cout << "| BLOCK HEADER uint32: [non-zero-count(8) | avg(12) | gamma-min(6) | gamma-max(6)]" << std::endl;
    //   }
    // }

    std::cout << "RATE   :" <<
      " in_sz= " << byte_in_count <<
      " | out_sz= " << byte_out_count <<
      " | k_avg= " << gamma_count / double(vector_count) <<
      " | k_min= " << gamma_min_nonzeros <<
      " | k_max= " << gamma_max_nonzeros <<
      " | ratio= " << (double)byte_in_count / byte_out_count <<
      " | in_bps= " << CHAR_BIT * (double)byte_in_count / scalar_count <<
      " | out_bps_nodict= " << CHAR_BIT * (double)byte_out_count_volume / scalar_count <<
      " | out_bps= " << CHAR_BIT * (double)byte_out_count / scalar_count <<
      " | out_bps_fixed_page=" << (double)total_bits_fixed_page / scalar_count <<
      " | ratio=" << (double)CHAR_BIT * (double)byte_out_count/total_bits_fixed_page <<
      std::endl;
    std::cout << "POSITION AVG ERR: " <<
      " Inner= " << inner_voxel_error <<
      " | Face= " << face_voxel_error <<
      " | Edge= " << edge_voxel_error <<
      " | Corner= " << corner_voxel_error << std::endl;

    std::cout << "QUALITY:" <<
      " rmse= " << ee.last_rmse() <<
      " | nrmse= " << ee.last_nrmse() <<
      " | maxe= " << ee.last_emax() <<
      " | psnr_zfp= " << ee.last_PSNR_dB_zfp() <<
      " | psnr_covra= " << ee.last_PSNR_dB() <<
      std::endl;
  } else {
    double stats_total_time_s = global_ck.elapsed().as_milliseconds()/1000.0;
    std::cerr << "NO ENCODING REQUIRED" << std::endl;
    std::cout << "METHOD: " <<
      " Algorithm: " << "KSVD" <<
      " | Dictionary " << arg_dictionary_element_count <<
      std::endl;
    std::cout << "SUMMARY:" <<
      " Data_size: " << sl::human_readable_quantity(volume_size) << 
      " | Coreset_size: " << sl::human_readable_quantity(arg_max_coreset_scalar_count) << 
      " | Coreset_time: " << stats_coreset_time_s <<
      " | Training_time: " << stats_training_time_s  <<
      " | Tolerance_computation_time_s: " <<  stats_tolerance_computation_time_s << 
      " | Total_time: " << stats_total_time_s << std::endl;

    std::cout << "INPUT  :" <<
      " nx= "<< vts->volume()->sample_counts()[0] <<
      " | ny= "<< vts->volume()->sample_counts()[1] <<
      " | nz= "<< vts->volume()->sample_counts()[2] << std::endl;
    std::cout << "LAYOUT :" <<
      " block= "      << "(" << arg_block_size << "+2*" << default_apron_size << ")^3 scalars" <<
      " | multiblock= " << default_coding_budget_optimizer_group_size1d << "^3 blocks" <<
      " | brick= "      << arg_brick_multiblock_count1d << "^3 multiblocks" << 
      " | page= "       << arg_page_brick_count1d << "^3 bricks" <<
      " | page_count= " << page_count <<
      std::endl;
    std::cout << "CODER :" <<
      " Tolerance= " << coder.tolerance() <<
      " | max_non_zeros= " << coder.max_non_zeros() <<
      " | dictionary = " << coder.signal_size() << "x" << coder.dictionary_element_count() <<
      std::endl;
    std::cout << "NO ENCODING REQUIRED" << std::endl;
  }
}

// ===========================================================================
// Budget optimizer from args
// ===========================================================================

vic::tdm_sparse_coding::dictionary_coding_budget_optimizer* make_cbo() {
  vic::tdm_sparse_coding::dictionary_coding_budget_optimizer* result = 0;
  if (arg_coding_budget_optimizer_method_string == "fixed") {
    result = new vic::tdm_sparse_coding::dictionary_coding_budget_optimizer;
  } else if (arg_coding_budget_optimizer_method_string == "greedygrow") {
    vic::tdm_sparse_coding::dictionary_coding_budget_optimizer_greedy_grow* gg = new vic::tdm_sparse_coding::dictionary_coding_budget_optimizer_greedy_grow;
    gg->set_is_maxe2_reduction_enabled(false);
    result = gg;
  } else if (arg_coding_budget_optimizer_method_string == "none") {
    std::cerr << "No encoding required withe method == none" << std::endl;
  } else {
    std::cerr << "Error: Unknown coding budget optimizer method '" << arg_coding_budget_optimizer_method_string << "'" << std::endl;
  }
  if (result) {
    result->set_non_zeros_per_signal(arg_max_non_zeros);
    result->set_group_size(cube(default_coding_budget_optimizer_group_size1d));
    result->set_max_non_zeros_per_signal(std::min(32*arg_max_non_zeros,128));// FIXME 
    // FIXME do it? result->set_is_average_removed(default_is_average_encoded_separately);
  }

  return result;
}

// ===========================================================================
// Coreset builder from args
// ===========================================================================

vic::tdm_sparse_coding::streaming_coreset_builder* make_scb() {
  vic::tdm_sparse_coding::streaming_coreset_builder* result =
    new vic::tdm_sparse_coding::streaming_coreset_builder_reservoir;
  const std::size_t block_scalar_size = sl::ipow(arg_block_size,3); // coreset size independent from apron
  const std::size_t coreset_blocks = arg_max_coreset_scalar_count / block_scalar_size;
  result->set_desired_coreset_size(coreset_blocks);
  result->set_is_average_removed(default_is_average_encoded_separately);

  return result;
}

// ===========================================================================
// Coder from args
// ===========================================================================
bool save_dictionary(const Eigen::MatrixXf& D, const std::string& filename) {
  FILE*fp = fopen(filename.c_str(), "wb");
  bool result = false;
  if (fp == 0) {
    std::cerr << "Unable to open file for saving dictionary " << filename << std::endl;
  } else {
    for(int y = 0; y < D.rows(); ++y) {
      for(int x = 0; x < D.cols(); ++x) {
	float v = D(y, x);
	if (fwrite(&v, sizeof(float), 1, fp) != 1) {
	  std::cerr << "Unable to write element to dictionary file" << std::endl;
	  fclose(fp);
	  return result;
	}
      }
    }
    result = true;
    std::cerr << "Saved dictionary to " << filename << std::endl;
  }
  fclose(fp);
  return result;
}

bool load_dictionary_in(Eigen::MatrixXf& D, const std::string& filename) {
  std::cerr << "Loading dictionary from " << filename << std::endl;
  FILE*fp = fopen(filename.c_str(), "rb");
  bool result = false;
  if (fp == 0) {
    std::cerr << "Unable to open file for reading dictionary " << filename << std::endl;
  } else {
    uint word_size = cube(arg_block_size+2*default_apron_size);
    D.resize(word_size, arg_dictionary_element_count);
    for(int y = 0; y < D.rows(); ++y) {
      for(int x = 0; x < D.cols(); ++x) {
	float v;
	int res = fread(&v, sizeof(float), 1, fp);
	if (res == 1) {
	  D(y, x) = v;
	} else {
	  std::cerr << "Unable to read element from dictionary file read item count " << res << std::endl;
	  fclose(fp);
	  exit(1);
	  return result;
	}	  
      }
    }
    result = true;
  }
  fclose(fp);
  return result;
}

vic::tdm_sparse_coding::dictionary_coder* make_coder() {
  vic::tdm_sparse_coding::dictionary_coder_ksvd* dc = new vic::tdm_sparse_coding::dictionary_coder_ksvd;
  float max_non_zeros = arg_max_non_zeros;
  float tolerance = arg_tolerance;
  if (arg_target_sparsity_with_variable_rate != 0) {
    max_non_zeros = int(arg_target_sparsity_with_variable_rate+0.5f);
    tolerance = 0.0f;
  }

  const float decorrelation_threshold = 1.0f;
  const bool  is_l1_pca_enabled = false;
  const int   dictionary_update_cycle_count = 1;
  
  dc->init(3,arg_block_size+2*default_apron_size,arg_dictionary_element_count,max_non_zeros,tolerance,
           decorrelation_threshold,
           dictionary_update_cycle_count,
           is_l1_pca_enabled);

  if (arg_input_dictionary_file != "") {
    Eigen::MatrixXf D_init;
    if (load_dictionary_in(D_init, arg_input_dictionary_file)) {
      dc->set_init_dictionary(D_init);
    }
  }
  
  dc->set_training_epoch_count(arg_epoch_count);

  return dc;
}

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char *argv[]) {
  int result = parse_options(argc,argv);
  if (result) return result;

  // Open input file
    
  vic::vol::raw_volume vol_r;
  vol_r.open_read(arg_input_file);
  if (!vol_r.is_open()) {
    std::cerr << "Unable to open: " << arg_input_file << " for reading" << std::endl;
    return 1;
  }

  vic::vol::volume_tile_stream vts_r(&vol_r,
                                                   arg_page_brick_count1d,
                                                   arg_brick_multiblock_count1d,
                                                   default_coding_budget_optimizer_group_size1d,
                                                   arg_block_size,
                                                   default_apron_size); // apron size

  vic::tdm_sparse_coding::streaming_coreset_builder* scb = make_scb();
  if (!scb) {
    std::cerr << "Cannot create coreset builder" << std::endl;
    return 1;
  }

  vic::tdm_sparse_coding::dictionary_coder* coder = make_coder();
  if (!coder) {
    std::cerr << "Cannot create dictionary coder" << std::endl;
    return 1;
  }

  vic::tdm_sparse_coding::dictionary_coding_budget_optimizer* cbo = make_cbo();
  if (!cbo && arg_coding_budget_optimizer_method_string != "none") {
    std::cerr << "Cannot create dictionary coding budget optimizer" << std::endl;
    return 1;
  }
    
  // Open output file
  vic::vol::raw_volume vol_w;
  vic::vol::volume_tile_stream* vts_w = 0; 

  if (arg_output_file != "") {
    // setup the volume
    vol_w.open_create(arg_output_file,
		      vol_r.sample_counts()[0],  vol_r.sample_counts()[1], vol_r.sample_counts()[2],
		      vol_r.sample_spacing()[0], vol_r.sample_spacing()[1], vol_r.sample_spacing()[2],
		      vol_r.bps());
    if (!vol_w.is_open()) {
      std::cerr << "Unable to open: " << arg_output_file << " for writing" << std::endl;
      return 1;
    }

    vts_w = new vic::vol::volume_tile_stream(&vol_w,
                                                           arg_page_brick_count1d,
                                                           arg_brick_multiblock_count1d,
                                                           default_coding_budget_optimizer_group_size1d,
                                                           arg_block_size,
                                                           default_apron_size); // apron size
  }

  vic::vol::raw_volume vol_dw;
  vic::vol::volume_tile_stream* vts_dw = 0; 
  if (arg_output_delta_file != "") {
    // setup the volume
    vol_dw.open_create(arg_output_delta_file,
		       vol_r.sample_counts()[0],  vol_r.sample_counts()[1], vol_r.sample_counts()[2],
		       vol_r.sample_spacing()[0], vol_r.sample_spacing()[1], vol_r.sample_spacing()[2],
		       vic::vol::float_xarray::FLOAT_T); // delta is always FP
    if (!vol_dw.is_open()) {
      std::cerr << "Unable to open: " << arg_output_delta_file << " for writing" << std::endl;
      return 1;
    }

    vts_dw = new vic::vol::volume_tile_stream(&vol_dw,
                                              arg_page_brick_count1d,
                                              arg_brick_multiblock_count1d,
                                              default_coding_budget_optimizer_group_size1d,
                                              arg_block_size,
                                              default_apron_size); // apron size of each block
  }
  
  // Do coding
  train_eval_save(&vts_r, *scb, *coder, cbo, vts_w, vts_dw);

  if (arg_output_dictionary_file != "") {
    save_dictionary(coder->dictionary(), arg_output_dictionary_file);
  }
  
  return 0;
}
