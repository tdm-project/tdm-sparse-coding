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
#include <vic/tdm_sparse_coding/error_evaluator.hpp>
#include <sl/utility.hpp>

namespace vic {
  namespace tdm_sparse_coding {

    void error_evaluator::print_progress_at_begin() {
      if (!is_verbose()) return;
      std::cerr << "** Error evaluation started" << std::endl;
    }

    void error_evaluator::print_progress_at_end() {
      if (!is_verbose()) return;
      std::cerr << std::endl; // Complete current pass
      std::cerr << "** Error evaluation completed" << std::endl;
    }

    void error_evaluator::print_progress_at_put() {
      if (!is_verbose()) return;
      
      sl::time_duration elapsed = progress_clock_.elapsed();

      std::cerr << "In: " << sl::human_readable_quantity(last_ycount_) << ", elapsed " << sl::human_readable_duration(elapsed);
      std::cerr << ": [ STATS ]: " <<
	" RMSE = " << last_rmse_ <<
	" NRMSE = " << last_nrmse_ <<
	" PSNR = " << last_PSNR_dB_ << " dB" <<
	" (PSNR_zfp= " << last_PSNR_dB_zfp_ << " dB)" <<
	" MAXE = " << last_emax_ << 
	" (y: " << last_ymin_ << " ... " << last_ymax_ << ")" << 
	"        \r";
    }

    void error_evaluator::inner_put(const Eigen::VectorXf& y, const Eigen::VectorXf& y_tilde, float w) {
      double y2 = y.squaredNorm();
      Eigen::VectorXf dy = y_tilde - y;
      double e2 = dy.squaredNorm();
      sum_w_  += w;
      sum_y2_ += w*y2;
      sum_e2_ += w*e2;
      max_e2_ = std::max(double(dy.array().square().maxCoeff()), max_e2_);

      for (int i=0; i<sum_y2_v_.size(); ++i) {
	sum_y2_v_[i] += w*y[i]*y[i];
	double e2_i = dy[i]*dy[i];
	max_e2_v_[i] = std::max(max_e2_v_[i], e2_i);
	sum_e2_v_[i] += e2_i;
      }
	
      if (last_ycount_ == 0) {
	last_ymax_ = double(y.array().maxCoeff());
	last_ymin_ = double(y.array().minCoeff());
      }else {
	last_ymax_ = std::max(double(y.array().maxCoeff()), last_ymax_);
	last_ymin_ = std::min(double(y.array().minCoeff()), last_ymin_);
      }
      last_ycount_+= 1;

      update_errors();
    }

    void error_evaluator::accumulate(const error_evaluator& other) {
      assert((in_data_dimension_ == 0) || in_data_dimension_ == other.in_data_dimension_);
      if (in_data_dimension_ == 0) {
	in_data_dimension_ = other.in_data_dimension_;
	sum_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	sum_y2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	max_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
	last_avg_e2_v_ = Eigen::VectorXd::Zero(in_data_dimension_);
      }
      
      if (last_ycount_ == 0) {
	sum_w_ = other.sum_w_;
	sum_y2_ = other.sum_y2_;
	sum_e2_ = other.sum_e2_;
	max_e2_ = other.max_e2_;
      	last_ymax_ = other.last_ymax_;
	last_ymin_ = other.last_ymin_;
	last_ycount_ = other.last_ycount_;
	sum_y2_v_ = other.sum_y2_v_;
	sum_e2_v_ = other.sum_e2_v_;
	max_e2_v_ = other.max_e2_v_;
      } else if (other.last_ycount_ == 0) {
	// Do nothing
      } else {
	// Both have data
	sum_w_ += other.sum_w_;
	sum_y2_ += other.sum_y2_;
	sum_e2_ += other.sum_e2_;
	max_e2_ = std::max(max_e2_, other.max_e2_);
      	last_ymax_ = std::max(last_ymax_, other.last_ymax_);
	last_ymin_ = std::min(last_ymin_, other.last_ymin_);
	last_ycount_ += other.last_ycount_;

       	sum_y2_v_ += other.sum_y2_v_;
	sum_e2_v_ += other.sum_e2_v_;
	for (int i=0; i<max_e2_v_.size(); ++i) {
	  max_e2_v_[i] = std::max(max_e2_v_[i], other.max_e2_v_[i]);
	}
      }

      update_errors();
    }

    void error_evaluator::update_errors() {
      const double yrange = last_ymax_-last_ymin_;
      //const double yamax  = std::max(std::abs(last_ymax_),std::abs(last_ymin_));
      last_emax_    = std::sqrt(max_e2_);
      last_rmse_    = sum_w_ ? std::sqrt(sum_e2_/double(sum_w_*in_data_dimension_)) : 0.0;
      last_nrmse_   = yrange ? (last_rmse_/yrange) : (last_rmse_ ? 1e30 : 0.0);
      last_PSNR_dB_ = last_rmse_ ? (yrange ? 20.0*std::log10(yrange/last_rmse_) : 0.0) : 1e30;  // COVRA + Westermann
      last_PSNR_dB_zfp_ = last_rmse_ ? (yrange ? 20.0 * log10(0.5*yrange/last_rmse_) : 0.0) : 1e30; // this is from Lindstrom's paper on zfp

      for (int i=0; i<sum_e2_v_.size(); ++i) {
	last_avg_e2_v_[i] = sum_w_ ? (sum_e2_v_[i] / sum_w_) : 0.0;
      }

    }
    
  } // namespace tdm_sparse_coding
} // namespace vic
