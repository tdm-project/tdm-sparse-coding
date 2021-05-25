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
#include "raw_volume.hpp"
#include <sl/os_file.hpp>
#include <sl/utility.hpp>
#include <cstring>

#define ENABLE_BRICK_OPTIMIZATION 1

namespace vic {
  
  namespace vol {

    //============== Creation & Destruction
    raw_volume::raw_volume() {
      xarray_ = 0;
    }
    
    raw_volume::~raw_volume() {
      cleanup();
    }

    //============== Open & Close
    
    void raw_volume::close() {
      if (xarray_) {
	xarray_->close();
      }
    }

    void raw_volume::cleanup() {
      sample_counts_ = sl::index3_t(0,0,0);
      sample_spacing_ = sl::vector3f(0,0,0);
      if (xarray_) {
	delete xarray_; xarray_ = 0;
      }
    }

    void raw_volume::reopen() {
      if (xarray_) {
	xarray_->reopen();
	xarray_->maximize_footprint(); /*FIXME*/
      }
    }

    bool raw_volume::is_open() const {
      bool result = false;
      if (xarray_) {
	result = xarray_->is_open();
      }
      return result;
    }

    std::string raw_volume::raw_filename(const std::string& fname) {
      return (fname=="") ? "" : sl::pathname_without_extension(fname) + ".raw";
    }

    std::string raw_volume::hdr_filename(const std::string& fname) {
      return (fname=="") ? "" : sl::pathname_without_extension(fname) + ".hdr";
    }
    
    void raw_volume::delete_files(const std::string& fname_raw,
				  const std::string& fname_hdr) {
      if (fname_raw!="") sl::os_file::file_delete(raw_filename(fname_raw).c_str());
      if (fname_hdr!="") sl::os_file::file_delete(hdr_filename(fname_hdr).c_str());
    }
    
    void raw_volume::open_create(const std::string& fname_raw,
				 sl::uint64_t nx, sl::uint64_t ny, sl::uint64_t nz,
				 float sx, float sy, float sz,
				 xdata_tag_t bps,
				 const std::string& fname_hdr) {
      cleanup();

      std::size_t CACHE_SZ=std::min(std::size_t(nx*ny*64*sizeof(float)),std::size_t(512*1024*1024));
       
      sample_counts_ = sl::index3_t(int(nx),int(ny),int(nz));
      sample_spacing_ = sl::vector3f(sx,sy,sz);
      xarray_ = new float_xarray();
      xarray_->open(raw_filename(fname_raw), "w", bps, CACHE_SZ);
      if (!xarray_->is_open()) {
	std::cerr << "EEE raw_volume::open_create() unable to create data file " << raw_filename(fname_raw) <<std::endl;
	cleanup();
	return;
      }
      // Allocate data
      sl::uint64_t size = nx*ny*nz;
      xarray_->resize(size);
      if (xarray_->size()!=size) {
	std::cerr << "EEE raw_volume::open_create() unable to resize data file " << raw_filename(fname_raw) <<std::endl;
	cleanup();
	delete_files(fname_raw);
	return;
      }
      xarray_->maximize_footprint(); /*FIXME*/

      // Write header
      std::string hdr=hdr_filename(fname_hdr);
      if (hdr=="") hdr=hdr_filename(fname_raw);
      std::ofstream os(hdr.c_str());
      if (!os.is_open()) {
	std::cerr << "EEE raw_volume::open_create() unable to create header " << hdr <<std::endl;
	cleanup();
	delete_files(fname_raw,hdr);
	return;
      }
      os << "raw_volume_header" << std::endl;
      os << "sample_counts "  << sample_counts_[0] << " " << sample_counts_[1] << " " << sample_counts_[2] << std::endl;
      os << "sample_spacing " << sample_spacing_[0] << " " << sample_spacing_[1] << " " << sample_spacing_[2] << std::endl;
      os << "bps " << int(bps) << std::endl;
      os.close();
    }
    
    void raw_volume::open_create_tmp(const std::string& fname_raw,
				     sl::uint64_t nx, sl::uint64_t ny, sl::uint64_t nz,
				     float sx, float sy, float sz,
				     xdata_tag_t bps) {
      cleanup();

      std::size_t CACHE_SZ=std::min(std::size_t(nx*ny*64*sizeof(float)),std::size_t(512*1024*1024));
      
      sample_counts_ = sl::index3_t(int(nx),int(ny),int(nz));
      sample_spacing_ = sl::vector3f(sx,sy,sz);
      xarray_ = new float_xarray();
      xarray_->open(raw_filename(fname_raw), "t", bps, CACHE_SZ);
      if (!xarray_->is_open()) {
	cleanup();
	return;
      }
      // Allocate data
      sl::uint64_t size = nx*ny*nz;
      xarray_->resize(size);
      if (xarray_->size()!=size) {
	std::cerr << "EEE raw_volume::open_create_tmp() unable to resize data file " << raw_filename(fname_raw) <<std::endl;
	cleanup();
	delete_files(fname_raw);
	return;
      }

      xarray_->maximize_footprint(); /*FIXME*/
    }
  
    void raw_volume::open_read(const std::string& fname_raw,
			       const std::string& fname_hdr) {
      cleanup();
      
       // Set defaults
      sample_counts_ = sl::index3_t(0,0,0);
      sample_spacing_ = sl::vector3f(1.0f,1.0f,1.0f);
      int bps = 0;
      
      // Parse header
      std::string hdr=hdr_filename(fname_hdr);
      if (hdr=="") hdr=hdr_filename(fname_raw);
      std::ifstream is(hdr.c_str());
      if (is.is_open()) {
	std::string line; bool ok=true;
	while(std::getline(is, line) && ok) {
	  std::cerr << "READING: " << line << std::endl;
	  std::stringstream   lis(line);
	  std::string tag;
	  lis >> tag;
	  if (tag == "raw_volume_header") {
	    // OK
	  } else if (tag == "sample_counts") {
	    lis >> sample_counts_[0] >> sample_counts_[1] >> sample_counts_[2];
	  } else if (tag == "sample_spacing") {
	    lis >> sample_spacing_[0] >> sample_spacing_[1] >> sample_spacing_[2];
	  } else if (tag == "bps") {
	    lis >> bps;
	  }
	}
      }
    
      bool bps_valid = ((bps==8) || (bps==16) || (bps==32));
      bool sc_valid = ((sample_counts_[0]>0) && (sample_counts_[1]>0) && (sample_counts_[2]>0));
      
      std::cerr << bps_valid << " " << sc_valid << std::endl;
      if (!(bps_valid && sc_valid)) {
	std::cerr << "EEE raw_volume::open_read() unable to read header " << hdr <<std::endl;
	cleanup();
	return;
      }
	  
      std::size_t CACHE_SZ=std::min(std::size_t(sample_counts_[0]*sample_counts_[1]*64*sizeof(float)),std::size_t(512*1024*1024));
      xarray_ = new float_xarray();
      xarray_->open(raw_filename(fname_raw), "r", xdata_tag_t(bps), CACHE_SZ);
      if (!xarray_->is_open()) {
	std::cerr << "EEE raw_volume::open_read() unable to read data " << raw_filename(fname_raw) <<std::endl;
	cleanup();
	return;
      }

      std::size_t total_sample_count = size();
      if (total_sample_count != xarray_->size()) {
        sc_valid = false;
        std::cerr << "EEE raw_volume::open_read() :" << std::endl
                  << "    header sample count " << sample_counts_[0] << " * "
                  << sample_counts_[1] << " * " << sample_counts_[2] << " = " << total_sample_count
                  << " != array sample count " << xarray_->size() << std::endl;
   	cleanup();
	return;
      }

      xarray_->maximize_footprint(); /*FIXME*/
    }
    
    raw_volume* raw_volume::copy_to(const std::string& fname_raw, xdata_tag_t bps) const {
      raw_volume* result = new raw_volume();

      result->open_create(raw_filename(fname_raw),
			  sample_counts_[0], sample_counts_[1], sample_counts_[2],
			  sample_spacing_[0], sample_spacing_[1], sample_spacing_[2],
			  bps);
      if (!result->is_open()) {
	delete result;
	result = 0;
      } else {        
        for (uint z=0; z < sample_counts_[2]; ++z) {
          for (uint y=0; y < sample_counts_[1]; ++y) {
            for (uint x=0; x < sample_counts_[0]; ++x) {
              result->put(item(x,y,z), x, y, z);
            }
          }
        }
      }
      
      return result;
    }
   
    std::pair<float, float> raw_volume::computed_data_value_range() const {
      float min_value = 1e30;
      float max_value = -1e30;
      for (uint z=0; z < sample_counts_[2]; ++z) {
	for (uint y=0; y < sample_counts_[1]; ++y) {
	  for (uint x=0; x < sample_counts_[0]; ++x) {
	    float v = item(x,y,z);
	    min_value = std::min(v, min_value);
	    max_value = std::max(v, max_value);
	  }
	}
      }
      
      return std::make_pair(min_value, max_value);    
    }

    // ============================== Brick Input

    template <class xarray_t>
    static void xarray_brick_item_in(float* brick,
				     std::size_t* inside_count,
				     std::size_t nx, std::size_t ny, std::size_t nz,
				     int x0, int y0, int z0,
				     const xarray_t* xarray_ptr,
				     std::size_t sizex, std::size_t sizey, std::size_t sizez) {
      typedef typename xarray_t::value_type xarray_value_t;
      
      std::size_t brick_sz = nz*nx*ny;
      
      int in_x_bgn = std::max(x0, 0);
      int in_y_bgn = std::max(y0, 0);
      int in_z_bgn = std::max(z0, 0);

      int in_x_end = std::min(x0+int(nx), int(sizex));
      int in_y_end = std::min(y0+int(ny), int(sizey));
      int in_z_end = std::min(z0+int(nz), int(sizez));
	
      std::size_t in_x_sz = std::max(0, in_x_end - in_x_bgn);
      std::size_t in_y_sz = std::max(0, in_y_end - in_y_bgn);
      std::size_t in_z_sz = std::max(0, in_z_end - in_z_bgn);
      std::size_t in_sz = in_x_sz * in_y_sz * in_z_sz;
      
      if (inside_count) *inside_count = in_sz;

      if (in_sz != brick_sz) {
	//SL_TRACE_OUT(-1) << "BRICK NOT FULLY INSIDE (" << x0 << " " << y0 << " " << z0 << ") - (" << x0+nx << " " << y0+nx << " " << z0+nx << ")" << std::endl;
	// Partially inside
	for (std::size_t dz=0; dz<nz; ++dz) {
	  const sl::uint64_t zz = sl::median(sl::int64_t(z0) + sl::int64_t(dz), sl::int64_t(0), sl::int64_t(sizez-1));
	  const std::size_t brick_idx_zz = nx * ny * dz;

	  for (std::size_t dy=0; dy<ny; ++dy) {
	    const sl::uint64_t yy = sl::median(sl::int64_t(y0) + sl::int64_t(dy), sl::int64_t(0), sl::int64_t(sizey-1));
	    const std::size_t brick_idx_zz_yy = brick_idx_zz + nx * dy;

	    for (std::size_t dx=0; dx<nx; ++dx) {
	      const sl::uint64_t xx = sl::median(sl::int64_t(x0) + sl::int64_t(dx), sl::int64_t(0), sl::int64_t(sizex-1));
	      const std::size_t brick_idx = brick_idx_zz_yy + dx;
	      
	      sl::uint64_t vol_idx = xx + sizex * (yy + sizey * zz);
	      
	      brick[brick_idx] = float_xarray_item_conv<xarray_value_t>::from_fixed(xarray_ptr->item(vol_idx));
	    } // for xx
	  } // for yy
	}  // for zz
      } else {
	//SL_TRACE_OUT(-1) << "BRICK FULLY INSIDE (" << x0 << " " << y0 << " " << z0 << ") - (" << x0+nx << " " << y0+nx << " " << z0+nx << ")" << std::endl;
	// Fully inside
	for (std::size_t dz=0; dz<nz; ++dz) {
	  const sl::uint64_t zz = sl::uint64_t(z0) + dz;
	  const std::size_t brick_idx_zz = nx * ny * dz;

	  for (std::size_t dy=0; dy<ny; ++dy) {
	    const sl::uint64_t yy = sl::uint64_t(y0) + dy;
	    const std::size_t brick_idx_zz_yy = brick_idx_zz + nx * dy;

	    const xarray_value_t* line=xarray_ptr->range_page_in(sizex * (yy + sizey * zz) + x0,
								 sizex * (yy + sizey * zz) + x0 + nx);
  
	    for (std::size_t dx=0; dx<nx; ++dx) {
	      const std::size_t brick_idx = brick_idx_zz_yy + dx;
	      
	      brick[brick_idx] = float_xarray_item_conv<xarray_value_t>::from_fixed(line[dx]);
	    } // for xx
	  } // for yy
	}  // for zz
      }	
    }
        
    void raw_volume::brick_item_in(float* brick,
				   std::size_t* inside_count,
				   std::size_t nx, std::size_t ny, std::size_t nz,
				   int x0, int y0, int z0) const {
#if ENABLE_BRICK_OPTIMIZATION
      {
	float_xarray::xarray_float_t* ptr = xarray_->to_xarray_float_t_pointer();
	if (ptr) {
	  xarray_brick_item_in(brick, inside_count,
			       nx, ny, nz, x0, y0, z0,
			       ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
      {
	float_xarray::xarray_uint16_t* ptr = xarray_->to_xarray_uint16_t_pointer();
	if (ptr) {
	  xarray_brick_item_in(brick, inside_count,
			       nx, ny, nz, x0, y0, z0,
			       ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
      {
	float_xarray::xarray_uint8_t* ptr = xarray_->to_xarray_uint8_t_pointer();
	if (ptr) {
	  xarray_brick_item_in(brick,
			       inside_count,
			       nx, ny, nz, x0, y0, z0,
			       ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
#endif
      // Fall back to SLOW implementation
      if (inside_count) *inside_count = 0;
      for (std::size_t dz=0; dz<nz; ++dz) {
	const int zz = z0 + dz;
	const std::size_t brick_idx_zz = nx * ny * dz;
	for (std::size_t dy=0; dy<ny; ++dy) {
	  const int yy = y0+dy;
	  const std::size_t brick_idx_zz_yy = brick_idx_zz + nx * dy;
	  for (std::size_t dx=0; dx<nx; ++dx) {
	    const int xx = x0 + dx;
	    const std::size_t brick_idx = brick_idx_zz_yy + dx;
	    float y_idx = item(xx,yy,zz);
	    brick[brick_idx] = y_idx;
	    if (inside_count && is_index_inside(xx,yy,zz)) ++(*inside_count);
	  } // for xx
	} // for yy
      }  // for zz
    }

    // ============================== Brick Output
    template <class xarray_t>
    static void xarray_put_brick(const float* brick,
				 std::size_t nx, std::size_t ny, std::size_t nz,
				 int x0, int y0, int z0,
				 xarray_t* xarray_ptr,
				 std::size_t sizex, std::size_t sizey, std::size_t sizez) {
      typedef typename xarray_t::value_type xarray_value_t;

      //std::size_t brick_sz = nz*nx*ny;

      int out_x_bgn = std::max(x0, 0);
      int out_y_bgn = std::max(y0, 0);
      int out_z_bgn = std::max(z0, 0);

      int out_x_end = std::min(x0+int(nx), int(sizex));
      int out_y_end = std::min(y0+int(ny), int(sizey));
      int out_z_end = std::min(z0+int(nz), int(sizez));
		
      std::size_t out_x_sz = std::max(0, out_x_end - out_x_bgn);
      std::size_t out_y_sz = std::max(0, out_y_end - out_y_bgn);
      std::size_t out_z_sz = std::max(0, out_z_end - out_z_bgn);
      std::size_t out_sz = out_x_sz * out_y_sz * out_z_sz;

      if (out_sz == 0) {
	// ALL OUTSIDE!
	// SL_TRACE_OUT(-1) << "BRICK FULLY OUTSIDE (" << x0 << " " << y0 << " " << z0 << ") - (" << x0+nx << " " << y0+nx << " " << z0+nx << ")" << std::endl;
      } else {
	//SL_TRACE_OUT(-1) << "BRICK PARTIALLY INSIDE (" << x0 << " " << y0 << " " << z0 << ") - (" << x0+nx << " " << y0+nx << " " << z0+nx << ")" << std::endl;
	// At least partially inside
	for (std::size_t dz=0; dz<std::size_t(out_z_sz); ++dz) {
	  
	  const sl::uint64_t zz = sl::uint64_t(out_z_bgn) + dz;
	  const std::size_t brick_idx_zz = nx * ny * (zz-z0);

	  for (std::size_t dy=0; dy<std::size_t(out_y_sz); ++dy) {
	    const sl::uint64_t yy = sl::uint64_t(out_y_bgn) + dy;
	    const std::size_t brick_idx_zz_yy = brick_idx_zz + nx * (yy-y0);

	    xarray_value_t* line=xarray_ptr->range_page_in(sizex * (yy + sizey * zz) + out_x_bgn,
							   sizex * (yy + sizey * zz) + out_x_bgn + out_x_sz);
  
	    for (std::size_t dx=0; dx<std::size_t(out_x_sz); ++dx) {
	      const sl::uint64_t xx = sl::uint64_t(out_x_bgn) + dx;
	      const std::size_t brick_idx = brick_idx_zz_yy + (xx-x0);

	      line[dx] = float_xarray_item_conv<xarray_value_t>::from_float(brick[brick_idx]);
	    } // for xx
	  } // for yy
	}  // for zz
      }	
    }
 
    void raw_volume::put_brick(const float* brick,
			       std::size_t nx, std::size_t ny, std::size_t nz,
			       int x0, int y0, int z0) {
#if ENABLE_BRICK_OPTIMIZATION
      {
	float_xarray::xarray_float_t* ptr = xarray_->to_xarray_float_t_pointer();
	if (ptr) {
	  xarray_put_brick(brick, nx, ny, nz, x0, y0, z0,
			   ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
      {
	float_xarray::xarray_uint16_t* ptr = xarray_->to_xarray_uint16_t_pointer();
	if (ptr) {
	  xarray_put_brick(brick, nx, ny, nz, x0, y0, z0,
			   ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
      {
	float_xarray::xarray_uint8_t* ptr = xarray_->to_xarray_uint8_t_pointer();
	if (ptr) {
	  xarray_put_brick(brick, nx, ny, nz, x0, y0, z0,
			   ptr, sample_counts_[0], sample_counts_[1], sample_counts_[2]);
	  return;
	}
      }
#endif

      // Fall back to SLOW implementation
      for (std::size_t dz=0; dz<nz; ++dz) {
	const int zz = z0 + dz;
	const std::size_t brick_idx_zz = nx * ny * dz;
	for (std::size_t dy=0; dy<ny; ++dy) {
	  const int yy = y0 + dy;
	  const std::size_t brick_idx_zz_yy = brick_idx_zz + nx * dy;
	  for (std::size_t dx=0; dx<nx; ++dx) {
	    const int xx = x0 + dx;
	    if (is_index_inside(xx,yy,zz)) {
	      const std::size_t brick_idx = brick_idx_zz_yy + dx;
	      float y_idx = brick[brick_idx];
	      put(y_idx, xx,yy,zz);
	    }
	  } // for xx
	} // for yy
      }  // for zz
    }
  } 
} 
