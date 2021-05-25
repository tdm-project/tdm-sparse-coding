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
#ifndef VIC_VOL_RAW_VOLUME_HPP
#define VIC_VOL_RAW_VOLUME_HPP

#include "float_xarray.hpp"
#include <sl/external_array.hpp>
#include <sl/fixed_size_vector.hpp>
#include <sl/index.hpp>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>


namespace vic {
  namespace vol {

    /**
     * Raw volume made of file.raw and file.hdr
     * file.raw is a sequence of uint8_t or uint16_t or float
     * file.hdr contains these informations:
     *   raw_volume_header
     *   sample_counts x y z
     *   sample_spacing x y z
     *   bps [1..32]
     * Floating point numbers in fixed formats are assumed to
     * be in the interval [0..1]
     */
    class raw_volume {
    public:
      typedef sl::index3_t		index3_t;
      typedef float_xarray::xdata_tag_t xdata_tag_t;
    protected:
      mutable float_xarray*	xarray_;
      index3_t		        sample_counts_;
      sl::vector3f		sample_spacing_;
      
    private: // DISABLE COPY

      raw_volume(const raw_volume&);
      raw_volume& operator=(const raw_volume&);
      
    public:
      raw_volume();

      ~raw_volume();

    public:

      static std::string raw_filename(const std::string& fname);
      static std::string hdr_filename(const std::string& fname);

    public:
      
      static void delete_files(const std::string& fname_raw,
			       const std::string& fname_hdr="");
      
      void open_create(const std::string& fname_raw,
		       sl::uint64_t nx, sl::uint64_t ny, sl::uint64_t nz,
		       float sx=1.0f, float sy=1.0f, float sz=1.0f,
		       xdata_tag_t bps=xdata_tag_t::FLOAT_T,
		       const std::string& fname_hdr="");

      void open_create_tmp(const std::string& fname_raw,
			   sl::uint64_t nx, sl::uint64_t ny, sl::uint64_t nz,
			   float sx=1.0f, float sy=1.0f, float sz=1.0f,
			   xdata_tag_t bps=xdata_tag_t::FLOAT_T);

      void open_read(const std::string& fname_raw,
		     const std::string& fname_hdr="");

      raw_volume* copy_to(const std::string& fname_raw, xdata_tag_t bps) const;
        
      bool is_open() const;

      void close();
      
      void reopen();

      void cleanup();

      
    public: // Characteristics

      inline const sl::index3_t& sample_counts() const {return sample_counts_;}
      inline const sl::vector3f& sample_spacing() const { return sample_spacing_; }
      inline const sl::vector3f spatial_extent() const { return sl::vector3f(sample_spacing_[0]*sample_counts_[0],
									     sample_spacing_[1]*sample_counts_[1],
									     sample_spacing_[2]*sample_counts_[2]); }

      inline xdata_tag_t bps() const { return xarray_ ? xarray_->bps() : xdata_tag_t(0); }

      inline sl::uint64_t size() const { return sample_counts_[0]* sample_counts_[1]*sample_counts_[2]; }
      
    public: // Indexing

      inline sl::uint64_t offset(const sl::index3_t& idx) const {
	return idx[0] +  sample_counts_[0] * (idx[1] + sample_counts_[1] * idx[2]);
      }

      inline sl::index3_t clamped_inside(const sl::index3_t& idx) const {
	return sl::index3_t(std::min(idx[0],std::size_t((sample_counts_[0])-1)),
			    std::min(idx[1],std::size_t((sample_counts_[1])-1)),
			    std::min(idx[2],std::size_t((sample_counts_[2])-1)));      
      }

      inline sl::index3_t clamped_inside(int x, int y, int z) const {
	return sl::index3_t(sl::median(0,x,int(sample_counts_[0])-1),
			    sl::median(0,y,int(sample_counts_[1])-1),
			    sl::median(0,z,int(sample_counts_[2])-1));
      }

      inline bool is_index_inside(const sl::index3_t& idx) const {
	return
	  (idx[0]<sample_counts_[0]) &&
	  (idx[1]<sample_counts_[1]) &&
	  (idx[2]<sample_counts_[2]);
      }

      inline bool is_index_inside(int x, int y, int z) const {
	return
	  (x>=0) && (x<int(sample_counts_[0])) &&
	  (y>=0) && (y<int(sample_counts_[1])) &&
	  (z>=0) && (z<int(sample_counts_[2]));
      }
      
    public: // Raw access

      inline void CHECK_ACCESS(const std::string& function, int x, int y, int z) const {
	if (!is_index_inside(x,y,z)) {
	  std::cerr << "raw_volume::ACCESS OUT OF BOUNDARIES" << std::endl;
	  std::cerr << "raw_volume::" <<  function << std::endl;
	  std::cerr << "xyz " << x << " " << y << " " << z << std::endl;
	  std::cerr << "sample_counts_ " << sample_counts_[0] << " " << sample_counts_[1] << " " << sample_counts_[2] << std::endl;
	  std::cerr << "bye bye!" << std::endl;
	  abort();
	}
      }

      inline float item_noidxclamp(const sl::index3_t& idx) const {
	assert(is_open());
	assert(is_index_inside(idx));
	CHECK_ACCESS("item_noidxclamp", idx[0],idx[1],idx[2]);
	return xarray_->item(offset(idx));
      }

      inline float item_noidxclamp(int x, int y, int z) const {
	assert(is_open());
	assert(is_index_inside(x,y,z));
	CHECK_ACCESS("item_noidxclamp", x,y,z);
	return xarray_->item(offset(sl::index3_t(x,y,z)));
      }

      inline void put_noidxclamp(float fv, const sl::index3_t& idx) {
	assert(is_open());
	assert(is_index_inside(idx));
	CHECK_ACCESS("put_noidxclamp", idx[0],idx[1],idx[2]);
	return xarray_->put(fv, offset(idx));
      }

      inline void put_noidxclamp(float fv, int x, int y, int z) {
	assert(is_open());
	assert(is_index_inside(x,y,z));
	CHECK_ACCESS("put_noidxclamp", x,y,z);
	return xarray_->put(fv, offset(sl::index3_t(x,y,z)));
      }

    public: // Access

      inline float item(const sl::index3_t& idx) const {
	assert(is_open());
	return item_noidxclamp(clamped_inside(idx));
      }

      inline float item(int x, int y, int z) const {
	assert(is_open());
	return item_noidxclamp(clamped_inside(x,y,z));
      }

      inline void put(float fv, const sl::index3_t& idx) {
	assert(is_open());
	put_noidxclamp(fv,clamped_inside(idx));
      }

      inline void put(float fv, int x, int y, int z) {
	assert(is_open());
	put_noidxclamp(fv,clamped_inside(x,y,z));
      }

    public: // Brick read/write

      /*
      ** Read brick of size nx,ny,nz from starting position x0, y0, z0
      ** Only the portion falling within volume boundaries is read. The
      ** remaining data is replicated by fetching data after clamping
      ** coordinates.
      */
      void brick_item_in(float* brick,
			 std::size_t* inside_count,
			 std::size_t nx, std::size_t ny, std::size_t nz,
			 int x0, int y0, int z0) const;

      /*
      ** Put brick of size nx,ny,nz starting from position x0, y0, z0
      ** Only the portion falling within volume boundaries is written. 
      */
      void put_brick(const float* brick,
		     std::size_t nx, std::size_t ny, std::size_t nz,
		     int x0, int y0, int z0);
      
    public: // Stats
      
      std::pair<float, float> computed_data_value_range() const;

    };

  } 
} 

#endif
