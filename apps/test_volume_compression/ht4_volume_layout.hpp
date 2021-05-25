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
#ifndef VIC_VOL_HT4_VOLUME_LAYOUT_HPP
#define VIC_VOL_HT4_VOLUME_LAYOUT_HPP

#include "raw_volume.hpp"

namespace vic {
  
  namespace vol {

    /**
     *  Hierarchical indexing of a tiled volume - 4 levels
     *  For rockdvr
     *  0 = page
     *  1 = brick
     *  2 = multiblock
     *  3 = block
     */
    class ht4_volume_layout {
    public:
      typedef sl::fixed_size_array<3,sl::int32_t> index3_t;
    protected: // hierarchy definition
      sl::uint32_t volume_scalar_extent_[3];      // Size of indexed volume for each dimension 
      sl::uint32_t htile_count1d_[4]; // Size of groups
    protected: // Derived info
      sl::uint32_t summed_htile_count1d_[4];

    protected:
      
      void rebuild() {
	summed_htile_count1d_[3] = htile_count1d_[3];
	summed_htile_count1d_[2] = htile_count1d_[2]*summed_htile_count1d_[3];
	summed_htile_count1d_[1] = htile_count1d_[1]*summed_htile_count1d_[2];
	summed_htile_count1d_[0] = htile_count1d_[0]*summed_htile_count1d_[1];

	std::cerr << "REBUILD:" << std::endl <<
	  htile_count1d_[0] << " " <<
	  htile_count1d_[1] << " " <<
	  htile_count1d_[2] << " " <<
	  htile_count1d_[3] << " " << std::endl <<
	  summed_htile_count1d_[0] << " " <<
	  summed_htile_count1d_[1] << " " <<
	  summed_htile_count1d_[2] << " " <<
	  summed_htile_count1d_[3] << " " << std::endl;
      }

    public:

      ht4_volume_layout() {
	volume_scalar_extent_[0] = 0;
	volume_scalar_extent_[1] = 0;
	volume_scalar_extent_[2] = 0;
	htile_count1d_[0] = 0;
	htile_count1d_[1] = 0;
	htile_count1d_[2] = 0;
	htile_count1d_[3] = 0;
	rebuild();
      }

      ht4_volume_layout(sl::uint32_t nx, sl::uint32_t ny, sl::uint32_t nz,
			sl::uint32_t h0, sl::uint32_t h1, sl::uint32_t h2, sl::uint32_t h3) {
	volume_scalar_extent_[0] = nx;
	volume_scalar_extent_[1] = ny;
	volume_scalar_extent_[2] = nz;
	htile_count1d_[0] = h0;
	htile_count1d_[1] = h1;
	htile_count1d_[2] = h2;
	htile_count1d_[3] = h3;
	rebuild();
      }

    public:

      // Number of scalars per dimension -- only real ones
      inline sl::uint32_t volume_scalar_extent(sl::uint32_t dim) const {
	return volume_scalar_extent_[dim];
      }

    public: // Hierarchically indexed volume

      // For a single dimension, how many tiles at level l+1 are contained in a tile at level l
      inline sl::uint32_t htile_size1d_at_level(int level) const {
	return htile_count1d_[level];
      }

      // For a single dimension, how many voxels are contained in a tile at level l
      inline sl::uint32_t htile_subtree_size1d_at_level(int level) const {
	return summed_htile_count1d_[level];
      }

      
      inline sl::uint64_t volume_tile_count1d(int level, int dim) const {
	sl::uint64_t htile_subtree_size1d_l0 = htile_subtree_size1d_at_level(0);
	sl::uint64_t subtree_count = (volume_scalar_extent(dim)+htile_subtree_size1d_l0-1)/htile_subtree_size1d_l0;
	sl::uint64_t result = subtree_count * htile_subtree_size1d_l0 / htile_subtree_size1d_at_level(level);
	return result;
      }
      
      inline sl::uint64_t volume_tile_count(int level) const {
	return
	  volume_tile_count1d(level,0)*
	  volume_tile_count1d(level,1)*
	  volume_tile_count1d(level,2);
      }
	  
      inline index3_t tile_origin(const index3_t& idx_l0) const {
	sl::uint64_t htile_subtree_size1d_l0 = htile_subtree_size1d_at_level(0);
	return index3_t(idx_l0[0]*htile_subtree_size1d_l0,
			idx_l0[1]*htile_subtree_size1d_l0,
			idx_l0[2]*htile_subtree_size1d_l0);
      }

      inline index3_t tile_origin(const index3_t& idx_l0,
				  const index3_t& idx_l1) const {
	index3_t result = tile_origin(idx_l0);
	sl::uint64_t htile_subtree_size1d_l1 =  htile_subtree_size1d_at_level(1);
	result[0] += idx_l1[0]*htile_subtree_size1d_l1;
	result[1] += idx_l1[1]*htile_subtree_size1d_l1;
	result[2] += idx_l1[2]*htile_subtree_size1d_l1;
	return result;
      }
      
      inline index3_t tile_origin(const index3_t& idx_l0,
				  const index3_t& idx_l1,
				  const index3_t& idx_l2) const {
	index3_t result = tile_origin(idx_l0, idx_l1);
	sl::uint64_t htile_subtree_size1d_l2 =  htile_subtree_size1d_at_level(2);
	result[0] += idx_l2[0]*htile_subtree_size1d_l2;
	result[1] += idx_l2[1]*htile_subtree_size1d_l2;
	result[2] += idx_l2[2]*htile_subtree_size1d_l2;
	return result;
      }
      
      inline index3_t tile_origin(const index3_t& idx_l0,
				  const index3_t& idx_l1,
				  const index3_t& idx_l2,
				  const index3_t& idx_l3) const {
	index3_t result = tile_origin(idx_l0, idx_l1, idx_l2);
	sl::uint64_t htile_subtree_size1d_l3 =  htile_subtree_size1d_at_level(3);
	result[0] += idx_l3[0]*htile_subtree_size1d_l3;
	result[1] += idx_l3[1]*htile_subtree_size1d_l3;
	result[2] += idx_l3[2]*htile_subtree_size1d_l3;
	return result;
      }

    };
  }
}

#endif
