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
#ifndef VIC_TDM_SPARSE_CODING_SIGNAL_STREAM_HPP
#define VIC_TDM_SPARSE_CODING_SIGNAL_STREAM_HPP

#include <Eigen/Core>
#include <sl/cstdint.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    /**
     *  Abstract interface to signal stream - subclasses define actual implementation
     */
    class signal_stream {
    protected:
      sl::uint64_t count_;
      sl::uint64_t current_;
      sl::uint32_t dimension_;
    public:
      signal_stream() : count_(sl::uint64_t(0)), current_(0), dimension_(0) {
      }

      virtual ~signal_stream() {
      }

      sl::uint32_t dimension() const {
	return dimension_;
      }
	
      sl::uint64_t count() const {
	return count_;
      }

      bool is_unbounded() const {
	return count_ == sl::uint64_t(-1);
      }
      
      virtual void restart() {
	current_ = 0;
      }

      virtual void current_in(Eigen::VectorXf& y, std::size_t* inside_count = 0) const = 0;
      
      virtual void forth() {
	assert(!off());
	++current_;
      }

      virtual bool off() const {
	return current_ >= count_;
      }

    };

  } // namespace tdm_sparse_coding
} // namespace vic

#endif
