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
#include "volume_tile_stream.hpp"

namespace vic {
  
  namespace vol {

    volume_tile_stream::volume_tile_stream(raw_volume* vol,
					   sl::uint32_t h0, sl::uint32_t h1, sl::uint32_t h2, sl::uint32_t h3,
					   sl::uint32_t h3_apron_sz) {
      vol_ = vol;
      ht4_volume_layout_ = vol::ht4_volume_layout(vol->sample_counts()[0], vol->sample_counts()[1], vol->sample_counts()[2],
						  h0, h1, h2, h3);
      block_apron_size_ = h3_apron_sz;
      dimension_ = block_size_with_apron()*block_size_with_apron()*block_size_with_apron(); // includes apron
      count_ = ht4_volume_layout_.volume_tile_count(3);

      end_idx_at_level_[0] = index3_t(ht4_volume_layout_.volume_tile_count1d(0,0),
				      ht4_volume_layout_.volume_tile_count1d(0,1),
				      ht4_volume_layout_.volume_tile_count1d(0,2));
      std::cerr << "END[0]" << end_idx_at_level_[0] << std::endl;
      for (std::size_t l=1; l<4; ++l) {
	end_idx_at_level_[l] = index3_t(ht4_volume_layout_.htile_size1d_at_level(l-1),
					ht4_volume_layout_.htile_size1d_at_level(l-1),
					ht4_volume_layout_.htile_size1d_at_level(l-1));
	std::cerr << "END[" << l << "]" << end_idx_at_level_[l] << std::endl;
      }
      restart();
      
    }
    
    volume_tile_stream::~volume_tile_stream() {
      vol_ = 0;
    }
    
    void volume_tile_stream::restart() {
      current_ = 0;
      for (std::size_t l=1; l<4; ++l) {
	current_idx_at_level_[l] = index3_t(0,0,0);
      }
      std::cerr << "RESTART" << std::endl;
    }

    void volume_tile_stream::forth()  {
      assert(!off());
      
      ++current_;
       
      std::size_t l=3;
      while (true) {
	current_idx_at_level_[l][0] += 1; 
	if (current_idx_at_level_[l][0]>=end_idx_at_level_[l][0]) {
	  //std::cerr << l << ":ENDX" << std::endl;
	  current_idx_at_level_[l][0]=0;
	  current_idx_at_level_[l][1]+=1; 
	  if (current_idx_at_level_[l][1]>=end_idx_at_level_[l][1]) {
	    //std::cerr << l << ":ENDY" << std::endl;
	    current_idx_at_level_[l][1]=0;
	    current_idx_at_level_[l][2]+=1; 
	    if (current_idx_at_level_[l][2]>=end_idx_at_level_[l][2]) {
	      current_idx_at_level_[l][2]=0;
	      //std::cerr << l << ":ENDZ" << std::endl;
	      if (l==0) break;
	      --l;
	    } else {
	      break;
	    }
	  } else {
	    break;
	  }
	} else {
	  break;
	}
      }
    }

    
    void volume_tile_stream::current_in(Eigen::VectorXf& y, std::size_t* inside_count) const {
      const std::size_t D = dimension_;
      const std::size_t B_with_apron = block_size_with_apron();
      const std::size_t A = block_apron_size();
      
      index3_t block_origin = ht4_volume_layout_.tile_origin(current_idx_at_level_[0],
							     current_idx_at_level_[1],
							     current_idx_at_level_[2],
							     current_idx_at_level_[3]);

      block_origin[0] -= A;
      block_origin[1] -= A;
      block_origin[2] -= A;
      
      //std::cerr << "READ (" << current_ << ")" << block_origin[0] << " " << block_origin[1] << " " << block_origin[2] << std::endl;

      y.resize(D);
	  
      vol_->brick_item_in(y.data(),
			  inside_count,
			  B_with_apron, B_with_apron, B_with_apron,
			  block_origin[0],block_origin[1],block_origin[2]);
    }

    void volume_tile_stream::write_current(const Eigen::VectorXf& y) const {
      //const std::size_t D = dimension_;
      const std::size_t B_with_apron     = block_size_with_apron();
      const std::size_t A                = block_apron_size();
      const std::size_t B_without_apron  = block_size_without_apron();
      const std::size_t B3_without_apron = B_without_apron*B_without_apron*B_without_apron;
      
      index3_t block_origin = ht4_volume_layout_.tile_origin(current_idx_at_level_[0],
							     current_idx_at_level_[1],
							     current_idx_at_level_[2],
							     current_idx_at_level_[3]);
      if (std::size_t(y.size())==B3_without_apron) {
	// We are writing a block without apron voxels - write it directy
	vol_->put_brick(y.data(),
			B_without_apron, B_without_apron, B_without_apron,
			block_origin[0],block_origin[1],block_origin[2]);
      } else {
	// Block has apron. Remove it before writing.
	Eigen::VectorXf y_without_apron(B3_without_apron);
        for (uint dz=0; dz < B_without_apron; ++dz) {
          for (uint dy=0; dy < B_without_apron; ++dy) {
            for (uint dx=0; dx < B_without_apron; ++dx) {
	      y_without_apron[dx + B_without_apron * (dy + B_without_apron * dz)] =
		y[(dx+A) + B_with_apron * ((dy+A) + B_with_apron * (dz+A))];
	    }
	  }
	}
	vol_->put_brick(y_without_apron.data(),
			B_without_apron, B_without_apron, B_without_apron,
			block_origin[0],block_origin[1],block_origin[2]);
      }
    }

  } // namespace tdm_sparse_coding
} // namespace vic
