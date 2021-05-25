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
#ifndef VIC_VOL_FLOAT_XARRAY_HPP
#define VIC_VOL_FLOAT_XARRAY_HPP

#include <sl/external_array.hpp>
#include <sl/utility.hpp>
#include <sl/math.hpp>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

namespace vic {
  namespace vol {

    template <class mantissa_t>
    struct float_xarray_item_conv {
      static inline mantissa_t from_float(float fx) {
	if (fx >= 1.0f) {
	  return sl::scalar_math<mantissa_t>::finite_upper_bound();
	} else if (fx <= 0.0f) {
	  return mantissa_t(0);
	} else {
	  return mantissa_t(fx * sl::scalar_math<mantissa_t>::finite_upper_bound() + 0.5f);
	}
      }
      
      static inline float from_fixed(mantissa_t ix) {
	sl::scalar_math<mantissa_t>::finite_upper_bound();
	return (1.0f / sl::scalar_math<mantissa_t>::finite_upper_bound()) * ix;
      }
    };

    template <>
    struct float_xarray_item_conv<float> {
      static inline float from_float(float fx) {
	return fx;
      }
      
      static inline float from_fixed(float ix) {
	return ix;
      }
    };
    
    
    // Wrapper for floats stored in external arrays in raw format
    // or fixed quantization as 8 or 16 bit integers. If fixed
    // the floats are assumed to be in the interval [0..1]
    class float_xarray {
    public:
      typedef enum { UNKNOWN_T=0, FLOAT_T=32, UINT16_T=16, UINT8_T=8 } xdata_tag_t;

      typedef sl::external_array1<uint8_t>	xarray_uint8_t;
      typedef sl::external_array1<uint16_t>	xarray_uint16_t;
      typedef sl::external_array1<float>	xarray_float_t;
    protected:

      xdata_tag_t xdata_tag_;
      union {
	void*            any_;
	xarray_uint8_t*  uint8_;
	xarray_uint16_t* uint16_;
	xarray_float_t*  float_;
      } xdata_array_;

      std::string last_open_fname_;
      sl::uint64_t last_open_cache_size_;
      
    public: // Create

      inline float_xarray(): xdata_tag_(xdata_tag_t(0)) { xdata_array_.any_ = 0; }
      inline ~float_xarray() {
	cleanup();
      }

    public: // Open and close
      
      inline void open(const std::string& fname, const char* mode, xdata_tag_t tag,
		       sl::uint64_t cache_size=1024*1024*64*sizeof(float)) {
	cleanup();
	switch (tag) {
	case FLOAT_T: {
	  xdata_array_.float_ = new xarray_float_t(fname, mode, cache_size);
	  if (!xdata_array_.float_->is_open()) {
	    delete xdata_array_.float_; xdata_array_.float_ = 0;
	  }
	} break;
	case UINT16_T: {
	  xdata_array_.uint16_ = new xarray_uint16_t(fname, mode, cache_size);
	  if (!xdata_array_.uint16_->is_open()) {
	    delete xdata_array_.uint16_; xdata_array_.uint16_ = 0;
	  }
	} break;
	case UINT8_T: {
	  xdata_array_.uint8_ = new xarray_uint8_t(fname, mode, cache_size);
	  if (!xdata_array_.uint8_->is_open()) {
	    delete xdata_array_.uint8_; xdata_array_.uint8_ = 0;
	  }
	} break;
	default: break;
	}
	if (xdata_array_.any_) {
	  xdata_tag_ = tag;
	}
      }
	
      inline bool is_open() const {
	switch (xdata_tag_) {
	case FLOAT_T : return xdata_array_.float_  && xdata_array_.float_->is_open(); 
	case UINT16_T: return xdata_array_.uint16_ && xdata_array_.uint16_->is_open(); 
	case UINT8_T : return xdata_array_.uint8_  && xdata_array_.uint8_->is_open(); 
	default: return false;
	}
      }

      inline void close() {
	if (xdata_array_.any_) {
	  // DEBUG: Print stats
	  std::cerr << std::endl <<
	    "*** Close vol file stats: " << std::endl <<
	    "*** SIZE: " << sl::human_readable_size(size()*item_size()) << std::endl <<
	    "*** MAP IN: " << sl::human_readable_quantity(stat_page_in_count()) << " times " << sl::human_readable_quantity(stat_page_in_byte_count()) << std::endl <<
	    "*** MAP OUT: " << sl::human_readable_quantity(stat_page_out_count()) << " times " << sl::human_readable_quantity(stat_page_out_byte_count()) << std::endl;
	    
	  // Do close
	  switch (xdata_tag_) {
	  case FLOAT_T : xdata_array_.float_->close(); break;
	  case UINT16_T: xdata_array_.uint16_->close(); break;
	  case UINT8_T : xdata_array_.uint8_->close(); break;
	  default: break;
	  }
	}
      }

      inline void reopen() {
	if (!is_open() && xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : xdata_array_.float_->reopen(); break;
	  case UINT16_T: xdata_array_.uint16_->reopen(); break;
	  case UINT8_T : xdata_array_.uint8_->reopen(); break;
	  default: break;
	  }
	}
      }
     
      inline void cleanup() {
	close();
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : delete xdata_array_.float_; xdata_array_.float_ = 0; break;
	  case UINT16_T: delete xdata_array_.uint16_; xdata_array_.float_ = 0; break;
	  case UINT8_T : delete xdata_array_.uint8_; xdata_array_.float_ = 0; break;
	  default: break;
	  }
	}
	xdata_tag_ = xdata_tag_t(0);
      }

      /// Move everything out-of-core
      inline void minimize_footprint() const {
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : xdata_array_.float_->minimize_footprint(); break;
	  case UINT16_T: xdata_array_.uint16_->minimize_footprint(); break;
	  case UINT8_T : xdata_array_.uint8_->minimize_footprint(); break;
	  default: break;
	  }
	}
      }

      /// Move everything in-core (single mmap for entire range)
      inline void maximize_footprint() const {
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : xdata_array_.float_->maximize_footprint(); break;
	  case UINT16_T: xdata_array_.uint16_->maximize_footprint(); break;
	  case UINT8_T : xdata_array_.uint8_->maximize_footprint(); break;
	  default: break;
	  }
	}
      }
       
    public: // Size

      inline void resize(sl::uint64_t sz) const {
	switch (xdata_tag_) {
	case FLOAT_T : xdata_array_.float_->resize(sz);  break;
	case UINT16_T: xdata_array_.uint16_->resize(sz); break;
	case UINT8_T : xdata_array_.uint8_->resize(sz);  break;
	default      : break;
	}
      }

      inline sl::uint64_t size() const {
	switch (xdata_tag_) {
	case FLOAT_T : return xdata_array_.float_->size(); 
	case UINT16_T: return xdata_array_.uint16_->size(); 
	case UINT8_T: return xdata_array_.uint8_->size(); 
	default      : return 0;
	}
      }

      inline std::size_t item_size() const {
	switch (xdata_tag_) {
	case FLOAT_T : return sizeof(float);
	case UINT16_T: return sizeof(sl::uint16_t);
	case UINT8_T: return sizeof(sl::uint8_t);
	default      : return 0;
	}
      }

    public: // access

      inline xdata_tag_t bps() const { return xdata_tag_; }
      
      inline float item(sl::uint64_t idx) const {
	assert(idx<size());
	switch (xdata_tag_) {
	case FLOAT_T : return float_xarray_item_conv<float>::from_fixed(xdata_array_.float_->item(idx)); 
	case UINT16_T: return float_xarray_item_conv<sl::uint16_t>::from_fixed(xdata_array_.uint16_->item(idx)); 
	case UINT8_T: return float_xarray_item_conv<sl::uint8_t>::from_fixed(xdata_array_.uint8_->item(idx)); 
	default      : return 0.0f;
	}
      }

      inline void put(float fx, sl::uint64_t idx) {
	assert(idx<size());
	switch (xdata_tag_) {
	case FLOAT_T : xdata_array_.float_->put(float_xarray_item_conv<float>::from_float(fx), idx); break;
	case UINT16_T: xdata_array_.uint16_->put(float_xarray_item_conv<sl::uint16_t>::from_float(fx), idx); break;
	case UINT8_T : xdata_array_.uint8_->put(float_xarray_item_conv<sl::uint8_t>::from_float(fx), idx); break;
	default      : break;
	}
      }

    public: // raw data  access
      
      inline xarray_uint8_t* to_xarray_uint8_t_pointer() {
	xarray_uint8_t* result = 0;
	switch (xdata_tag_) {
	case UINT8_T : result = xdata_array_.uint8_;
	default      : break;
	}
	return result;
      }
			   
      inline xarray_uint16_t* to_xarray_uint16_t_pointer() {
	xarray_uint16_t* result = 0;
	switch (xdata_tag_) {
	case UINT16_T : result = xdata_array_.uint16_;
	default      : break;
	}
	return result;
      }
      
      inline xarray_float_t* to_xarray_float_t_pointer() {
	xarray_float_t* result = 0;
	switch (xdata_tag_) {
	case FLOAT_T : result = xdata_array_.float_;
	default      : break;
	}
	return result;
      }
      
      inline const xarray_uint8_t* to_xarray_uint8_t_pointer() const {
	const xarray_uint8_t* result = 0;
	switch (xdata_tag_) {
	case UINT8_T : result = xdata_array_.uint8_;
	default      : break;
	}
	return result;
      }
			   
      inline const xarray_uint16_t* to_xarray_uint16_t_pointer() const {
	const xarray_uint16_t* result = 0;
	switch (xdata_tag_) {
	case UINT16_T : result = xdata_array_.uint16_;
	default      : break;
	}
	return result;
      }
      
      inline const xarray_float_t* to_xarray_float_t_pointer() const {
	const xarray_float_t* result = 0;
	switch (xdata_tag_) {
	case FLOAT_T : result = xdata_array_.float_;
	default      : break;
	}
	return result;
      }

    public: // stats
      
    // Clear paging statistics - usually done at file open
      inline void stat_clear() const {
	switch (xdata_tag_) {
	case FLOAT_T : xdata_array_.float_->stat_clear(); break;
	case UINT16_T: xdata_array_.uint16_->stat_clear(); break;
	case UINT8_T : xdata_array_.uint8_->stat_clear(); break;
	default      : break;
	}
      }
    
      // Number of mmaps since open
      inline std::size_t stat_page_in_count() const {
	std::size_t result=0;
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : result = xdata_array_.float_->stat_page_in_count(); break;
	  case UINT16_T: result = xdata_array_.uint16_->stat_page_in_count(); break;
	  case UINT8_T : result = xdata_array_.uint8_->stat_page_in_count(); break;
	  default      : break;
	  }
	}
	return result;
      }
      
      // Number of mmapped bytes since open
      inline sl::uint64_t stat_page_in_byte_count() const {
	sl::uint64_t result=0;
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : result = xdata_array_.float_->stat_page_in_byte_count(); break;
	  case UINT16_T: result = xdata_array_.uint16_->stat_page_in_byte_count(); break;
	  case UINT8_T : result = xdata_array_.uint8_->stat_page_in_byte_count(); break;
	  default      : break;
	  }
	}
	return result;
      }
      
      // Number of munmaps since open
      inline std::size_t stat_page_out_count() const {
	std::size_t result=0;
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : result = xdata_array_.float_->stat_page_out_count(); break;
	  case UINT16_T: result = xdata_array_.uint16_->stat_page_out_count(); break;
	  case UINT8_T : result = xdata_array_.uint8_->stat_page_out_count(); break;
	  default      : break;
	  }
	}
	return result;
      }
      
      // Number of munmapped bytes since open
      inline sl::uint64_t stat_page_out_byte_count() const {
	sl::uint64_t result=0;
	if (xdata_array_.any_) {
	  switch (xdata_tag_) {
	  case FLOAT_T : result = xdata_array_.float_->stat_page_out_byte_count(); break;
	  case UINT16_T: result = xdata_array_.uint16_->stat_page_out_byte_count(); break;
	  case UINT8_T : result = xdata_array_.uint8_->stat_page_out_byte_count(); break;
	  default      : break;
	  }
	}
	return result;
      }
      
    };
        
  }
}

#endif 

