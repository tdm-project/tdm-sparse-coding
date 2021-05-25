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
#ifndef VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_HPP
#define VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_HPP

#include <Eigen/Core>
#include <sl/cstdint.hpp>
#include <sl/clock.hpp>
#include <vector>
#include <vic/tdm_sparse_coding/signal_stream.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    /**
     *  Single pass streaming coreset builder
     */
    class streaming_coreset_builder {
    protected:
      std::string                  method_name_;
      std::vector<float>           out_w_;
      std::vector<Eigen::VectorXf> out_Y_;
      std::size_t                  desired_coreset_size_;
      bool                         is_verbose_;
      bool                         is_average_removed_;
      bool                         are_partially_outside_blocks_skipped_;
      std::size_t                  pass_count_;
      bool                         is_building_;
      std::size_t                  current_pass_;
      std::size_t                  in_data_dimension_;
      sl::uint64_t                 in_count_;
      sl::uint64_t                 current_pass_in_count_;
      std::size_t                  zero_y_count_;
      sl::real_time_clock          progress_clock_;
      
    public: // Creation & destruction
      
      inline streaming_coreset_builder():
	method_name_("none"),
	desired_coreset_size_(1024),
	is_verbose_(true),
	is_average_removed_(false),
	are_partially_outside_blocks_skipped_(true),
	pass_count_(1),
	is_building_(false),
	current_pass_(0),
	in_data_dimension_(0),
	in_count_(0),
	current_pass_in_count_(0),
        zero_y_count_(0) {
      }

      virtual ~streaming_coreset_builder() {}

      virtual void clear() {
	assert(!is_building());
	in_count_ = 0;
	in_data_dimension_ = 0;
	current_pass_in_count_ = 0;
	current_pass_ = 0;
	out_w_.clear();
	out_Y_.clear();
      }

    public: // Params
      
      inline std::size_t desired_coreset_size() const { return desired_coreset_size_; }
      inline void set_desired_coreset_size(std::size_t R) { desired_coreset_size_ = R; }

      inline bool is_verbose() const { return is_verbose_; }
      inline void set_is_verbose(bool x) { is_verbose_ = x; }

      inline bool is_average_removed() const { return is_average_removed_; }
      inline void set_is_average_removed(bool x) { is_average_removed_ = x; }

      inline bool are_partially_outside_blocks_skipped() const { return are_partially_outside_blocks_skipped_; }
      inline void set_are_partially_outside_blocks_skipped(bool x) { are_partially_outside_blocks_skipped_ = x; }

    public: // Output

      inline sl::uint64_t in_count() const { return in_count_; }
      
      inline std::size_t zero_y_count() const {	return zero_y_count_; }
      inline std::size_t  in_data_dimension() const { return in_data_dimension_; }
      
      inline const std::vector<float>&           weights() const { return out_w_; }
      inline const std::vector<Eigen::VectorXf>& signals() const { return out_Y_; }

      inline void extract_weights_and_signals_in(std::vector<float>& w,
						 std::vector<Eigen::VectorXf>& Y) {
	std::swap(out_w_, w); out_w_.clear();
	std::swap(out_Y_, Y); out_Y_.clear();
      }
					 
    public: // Input: Streaming interface

      // Interface;
      //   for (pass=0; pass<pass_count; ++ pass) {
      //      begin();
      //        for each element e {
      //           put(e);
      //        }
      //      end();
      //   }
            
      inline std::size_t pass_count() const {
	return pass_count_;
      }
      
      virtual void begin() {
	if (current_pass_ == 0) clear();
	current_pass_in_count_ = 0;
        zero_y_count_ = 0;
	print_progress_at_begin();
	is_building_ = true;
      } 

      inline bool is_building() const {
	return is_building_;
      } 

      inline void put(const Eigen::VectorXf& y) {
	assert(in_data_dimension_ == 0 || in_data_dimension_ == std::size_t(y.size()));
	if (in_data_dimension_ == 0) in_data_dimension_ = std::size_t(y.size());
	if (is_average_removed()) {
	  // Remove average to make vector zero-mean
	  inner_put(y.array() - y.sum()/float(y.size()));
	} else {
	  // Use original vector
	  inner_put(y);
	}
	++current_pass_in_count_;
	if (current_pass_in_count_%50000 == 0) print_progress_at_put();
      }
      
      virtual void end() {
	assert(is_building());
	if (current_pass_ == 0) {
	  in_count_ = current_pass_in_count_;
	}
	print_progress_at_put();
	++current_pass_;
	print_progress_at_end();
	if (current_pass_>=pass_count_) {
	  is_building_ = false;
	  current_pass_ = 0; // Completed
	}
      }

    public:

      virtual void build(signal_stream& Y);
      
    protected:

      virtual void inner_put(const Eigen::VectorXf& y) = 0;
      
      void print_progress_at_begin();
      void print_progress_at_end();
      void print_progress_at_put();

    }; // class streaming_coreset_builder
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
