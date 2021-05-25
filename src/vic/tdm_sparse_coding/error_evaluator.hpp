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
#ifndef VIC_TDM_SPARSE_CODING_ERROR_EVALUATOR_HPP
#define VIC_TDM_SPARSE_CODING_ERROR_EVALUATOR_HPP

#include <Eigen/Core>
#include <sl/cstdint.hpp>
#include <sl/clock.hpp>
#include <vector>
#include <vic/tdm_sparse_coding/signal_stream.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    /**
     *  Streaming error computation
     */
    class error_evaluator {
    protected:
      bool is_verbose_;
      bool is_evaluating_;
      std::size_t in_data_dimension_;
      sl::uint64_t last_ycount_;
      double sum_w_;
      double sum_e2_;
      double sum_y2_;
      double max_e2_;
      
      Eigen::VectorXd sum_e2_v_;
      Eigen::VectorXd sum_y2_v_;
      Eigen::VectorXd max_e2_v_;
      
      double last_ymax_;
      double last_ymin_;
      double last_emax_;
      double last_rmse_;
      double last_nrmse_;
      double last_PSNR_dB_;
      double last_PSNR_dB_zfp_;

      Eigen::VectorXd last_avg_e2_v_;
      
      sl::real_time_clock          progress_clock_;
      
    public: // Creation & destruction
      
      inline error_evaluator():
	is_verbose_(true),
	is_evaluating_(false) {
	clear();
      }

      ~error_evaluator() {}

      void clear() {
	in_data_dimension_ = 0;

	sum_w_  = 0.0;
	sum_e2_ = 0.0;
	sum_y2_ = 0.0;
	
	max_e2_ = 0.0;
	
	last_ycount_ = 0;
	last_ymax_ = 0.0;
	last_ymin_ = 0.0;
	last_rmse_ = 0.0;
	last_nrmse_ = 0.0;
	last_PSNR_dB_ = 0.0;
	last_PSNR_dB_zfp_ = 0.0;
	sum_e2_v_ = Eigen::VectorXd();
	sum_y2_v_ = Eigen::VectorXd();
        max_e2_v_ = Eigen::VectorXd();
	last_avg_e2_v_ = Eigen::VectorXd();
      }

      void accumulate(const error_evaluator& other);

    public: // Results

      inline sl::uint64_t last_vector_count() const { return last_ycount_; }
      inline sl::uint64_t last_scalar_count() const { return last_ycount_ * in_data_dimension_; }
      inline double last_ymax() const { return last_ymax_; }
      inline double last_ymin() const { return last_ymin_; }
      inline double last_emax() const { return last_emax_; }
      inline double last_rmse() const { return last_rmse_; }
      inline double last_nrmse() const { return last_nrmse_; }
      inline double last_PSNR_dB() const { return last_PSNR_dB_; }
      inline double last_PSNR_dB_zfp() const { return last_PSNR_dB_zfp_; }

      inline const Eigen::VectorXd& last_avg_e2_v() const { return last_avg_e2_v_; } 
      
      inline bool is_verbose() const { return is_verbose_; }
      inline void set_is_verbose(bool x) { is_verbose_ = x; }
					 
    public: // Input: Streaming interface

      // Interface;
      //      begin();
      //        for each element 0 y{
      //           y_tilde = approximation(y)
      //           put(y,y_tilde);
      //        }
      //      end();
      //   }
      
      inline void begin() {
	assert(!is_evaluating());
	clear();
	is_evaluating_ = true;
	print_progress_at_begin();
      } 

      inline bool is_evaluating() const {
	return is_evaluating_;
      } 

      inline void put(const Eigen::VectorXf& y, const Eigen::VectorXf& y_tilde, float w=1.0f) {
	assert(y.size() == y_tilde.size());
	assert((in_data_dimension_ == 0) || in_data_dimension_ == std::size_t(y.size()));
	
	if (in_data_dimension_ == 0) {
	  in_data_dimension_ = y.size();
	  sum_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	  sum_y2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	  max_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	  last_avg_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	}
	inner_put(y,y_tilde,w);
	if (last_ycount_%50000 == 0) print_progress_at_put();
      }
      
      inline void end() {
	print_progress_at_put();
	print_progress_at_end();
	is_evaluating_ = false;
      }

         
      void print_progress_at_begin();
      void print_progress_at_end();
      void print_progress_at_put();

    protected:

      void inner_put(const Eigen::VectorXf& y,
		     const Eigen::VectorXf& y_tilde,
		     float w);
   
      void update_errors();
      
    }; // class error_evaluator
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
