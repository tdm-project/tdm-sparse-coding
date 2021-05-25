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
#include <vic/tdm_sparse_coding/streaming_coreset_builder.hpp>
#include <sl/utility.hpp>

namespace vic {
  namespace tdm_sparse_coding {

    void streaming_coreset_builder::print_progress_at_begin() {
      if (!is_verbose()) return;
      if (current_pass_ == 0) {
	// Start
	std::cerr << "** CORESET BUILDING START. METHOD=" << method_name_ << ". TARGET= " << sl::human_readable_quantity(desired_coreset_size_) << " samples" << std::endl;
      }
      std::cerr << "   Streaming pass " << current_pass_+1 << "/" << pass_count_ << "\r";
    }

    void streaming_coreset_builder::print_progress_at_end() {
      if (!is_verbose()) return;
      std::cerr << std::endl; // Complete current pass
      if (current_pass_ >= pass_count_) {
	// End
	std::cerr << "** CORESET BUILDING COMPLETED." <<
	  "METHOD= " << method_name_ << "." <<
	  "TARGET= " << sl::human_readable_quantity(desired_coreset_size_) << " samples" <<
	  " CORESET SIZE = " << sl::human_readable_quantity(out_Y_.size()) << " samples (" << sl::human_readable_percent(100.0f*float(out_Y_.size())/float(in_count_))<< ")" <<
	  " ZERO COUNT = " << sl::human_readable_quantity(zero_y_count()) << " samples (" << sl::human_readable_percent(100.0f*float(zero_y_count())/float(in_count_))<< ")" << std::endl;
      }
    }

    void streaming_coreset_builder::print_progress_at_put() {
      std::cerr << "   Streaming pass " << current_pass_+1 << "/" << pass_count_ << ": ";
      sl::time_duration elapsed = progress_clock_.elapsed();

      std::cerr << "In: " << sl::human_readable_quantity(current_pass_in_count_) << ", elapsed " << sl::human_readable_duration(elapsed);
      if (in_count_ == 0) {
	// Don't know total count -- ETA
      } else {
	// We know total count -- ETA
	sl::uint64_t current_put_count = current_pass_in_count_ + (pass_count_-1) * in_count_;

	float speed = float(current_put_count) / float(elapsed.as_microseconds());
	sl::time_duration local_eta((in_count_ - current_pass_in_count_)/speed);
	sl::time_duration total_eta((pass_count_*in_count_ - current_put_count)/speed);
	std::cerr << " ETA " <<  sl::human_readable_duration(local_eta) 
		  << ", total ETA " << sl::human_readable_duration(total_eta);
      }
      std::cerr << "     \r";
    }

    void streaming_coreset_builder::build(signal_stream& Y) {
      Eigen::VectorXf y(Y.dimension());
      
      for (std::size_t pass=0; pass<pass_count(); ++ pass) {
	std::size_t total_blocks = 0;
	std::size_t inside_blocks = 0;
	std::size_t inside_scalars = 0;
	
	begin();
	{	  
	  Y.restart();
	  while (!Y.off()) {
	    ++total_blocks;
	    std::size_t in_count=0;
	    Y.current_in(y, &in_count);
	    inside_scalars += in_count;
	    if (in_count == Y.dimension()) {
	      // Insert in stream only blocks that are fully inside
	      ++inside_blocks;
	    }
	    if ((!are_partially_outside_blocks_skipped()) || (in_count == Y.dimension())) {
	      put(y);
	    }
	    Y.forth();
	  }
	}
	end();

	std::cerr << "INSIDE BLOCKS: " << inside_blocks << "/" << total_blocks << std::endl;
	std::cerr << "INSIDE SCALARS: " << inside_scalars << "/" << total_blocks * Y.dimension() << std::endl;
      }
    }    
  } // namespace tdm_sparse_coding
} // namespace vic
