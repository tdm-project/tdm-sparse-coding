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
#ifndef VIC_TDM_SPARSE_CODING_VOLUME_TILE_STREAM_HPP
#define VIC_TDM_SPARSE_CODING_VOLUME_TILE_STREAM_HPP

#include <vic/tdm_sparse_coding/signal_stream.hpp>
#include "raw_volume.hpp"
#include "ht4_volume_layout.hpp"

namespace vic {
  
  namespace vol {
     
    /**
     *  Signal stream for a rockdvr volume. Each signal is a block.
     */
    class volume_tile_stream: public tdm_sparse_coding::signal_stream {
    protected:
      typedef vol::ht4_volume_layout::index3_t index3_t;
      
      vic::vol::raw_volume* vol_;
      vol::ht4_volume_layout     ht4_volume_layout_;

      std::size_t block_apron_size_;
      index3_t current_idx_at_level_[4]; 
      index3_t end_idx_at_level_[4]; 
      
    public:

      volume_tile_stream(vic::vol::raw_volume* vol,
			 sl::uint32_t h0, sl::uint32_t h1, sl::uint32_t h2, sl::uint32_t h3,
			 sl::uint32_t h3_apron_sz=0);

      inline const vic::vol::raw_volume* volume() const { return vol_; }

      inline std::size_t page_size() const       { return ht4_volume_layout_.htile_size1d_at_level(0); }
      inline std::size_t brick_size() const      { return ht4_volume_layout_.htile_size1d_at_level(1); }
      inline std::size_t multiblock_size() const { return ht4_volume_layout_.htile_size1d_at_level(2); }
      inline std::size_t block_apron_size() const { return block_apron_size_; }
      inline std::size_t block_size_without_apron() const { return ht4_volume_layout_.htile_size1d_at_level(3); }
      inline std::size_t block_size_with_apron() const { return block_size_without_apron()+2*block_apron_size(); }
      
      virtual ~volume_tile_stream();

      virtual void current_in(Eigen::VectorXf& y, std::size_t* inside_count=0) const;

      virtual void write_current(const Eigen::VectorXf& y) const;

      virtual void restart();

      virtual void forth();

    };

  } // namespace tdm_sparse_coding
} // namespace vic

#endif
