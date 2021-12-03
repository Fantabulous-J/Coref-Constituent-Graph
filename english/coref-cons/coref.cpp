#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
torch::Tensor extract_spans(torch::Tensor& sorted_span_indices, torch::Tensor& candidate_starts,
    torch::Tensor& candidate_ends, int num_output_spans, int max_sentence_length,
    bool _sort_spans){
    auto num_input_spans = sorted_span_indices.size(0);

    auto output_span_indices = torch::zeros({num_output_spans}, sorted_span_indices.type().toScalarType(at::kLong));
    auto *output_span_indices_idx = output_span_indices.data_ptr<int64_t>();

//    std::vector<int> sorted_input_span_indices(num_input_spans);

//    auto *span_scores_idx = span_scores.data_ptr<float64_t>();
//    std::iota(sorted_input_span_indices.begin(), sorted_input_span_indices.end(), 0);
//    std::sort(sorted_input_span_indices.begin(), sorted_input_span_indices.end(),
//                [&span_scores_idx](int i1, int i2) {
//                  return span_scores_idx[i1] < span_scores_idx[i2];
//                });
    std::vector<int> top_span_indices;
    std::unordered_map<int, int> end_to_earliest_start;
    std::unordered_map<int, int> start_to_latest_end;

    int current_span_index = 0, num_selected_spans = 0;
    auto *sorted_span_indices_idx = sorted_span_indices.data_ptr<int64_t>();
    auto *candidate_starts_idx = candidate_starts.data_ptr<int64_t>();
    auto *candidate_ends_idx = candidate_ends.data_ptr<int64_t>();
    while (num_selected_spans < num_output_spans && current_span_index < num_input_spans) {
      int i = sorted_span_indices_idx[current_span_index];
      bool any_crossing = false;
      const int start = candidate_starts_idx[i];
      const int end = candidate_ends_idx[i];
      for (int j = start; j <= end; ++j) {
        auto latest_end_iter = start_to_latest_end.find(j);
        if (latest_end_iter != start_to_latest_end.end() && j > start && latest_end_iter->second > end) {
            // Given (), exists [], such that ( [ ) ]
          any_crossing = true;
          break;
        }
        auto earliest_start_iter = end_to_earliest_start.find(j);
        if (earliest_start_iter != end_to_earliest_start.end() && j < end && earliest_start_iter->second < start) {
          // Given (), exists [], such that [ ( ] )
          any_crossing = true;
          break;
        }
      }
      if (!any_crossing) {
        if (_sort_spans) {
          top_span_indices.push_back(i);
        } else {
          output_span_indices_idx[num_selected_spans] = i;
        }
        ++num_selected_spans;
          // Update data struct.
        auto latest_end_iter = start_to_latest_end.find(start);
        if (latest_end_iter == start_to_latest_end.end() || end > latest_end_iter->second) {
          start_to_latest_end[start] = end;
        }
        auto earliest_start_iter = end_to_earliest_start.find(end);
        if (earliest_start_iter == end_to_earliest_start.end() || start < earliest_start_iter->second) {
          end_to_earliest_start[end] = start;
        }
      }
      ++current_span_index;
    }
      // Sort and populate selected span indices.
    if (_sort_spans) {
      std::sort(top_span_indices.begin(), top_span_indices.end(),
                [candidate_starts_idx, candidate_ends_idx] (int i1, int i2) {
                  if (candidate_starts_idx[i1] < candidate_starts_idx[i2]) {
                    return true;
                  } else if (candidate_starts_idx[i1] > candidate_starts_idx[i2]) {
                    return false;
                  } else if (candidate_ends_idx[i1] < candidate_ends_idx[i2]) {
                    return true;
                  } else if (candidate_ends_idx[i1] > candidate_ends_idx[i2]) {
                    return false;
                  } else {
                    return i1 < i2;
                  }
                });
      for (int i = 0; i < num_output_spans; ++i) {
          output_span_indices_idx[i] = top_span_indices[i];
        }
    }
    return output_span_indices;
}

torch::Tensor coref_sigmoid(torch::Tensor z, int constant) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s * constant;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid", &coref_sigmoid, "COREF sigmoid");
  m.def("extract_spans", &extract_spans, "COREF extract spans");
}
