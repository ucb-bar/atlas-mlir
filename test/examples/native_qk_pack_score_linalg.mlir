module {
  func.func @qk_pack_score_tile(
      %tokens: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %wq: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %wk: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<32x32xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %q_empty = tensor.empty() : tensor<32x32xbf16>
    %q_init = linalg.fill ins(%zero : bf16) outs(%q_empty : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
    %q = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%tokens, %wq : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%q_init : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    %k_empty = tensor.empty() : tensor<32x32xbf16>
    %k_init = linalg.fill ins(%zero : bf16) outs(%k_empty : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
    %k = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%tokens, %wk : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%k_init : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    %q_fp8_empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %q_fp8 = linalg.generic {
      indexing_maps = [affine_map<(m,k)->(m,k)>, affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%q : tensor<32x32xbf16>)
      outs(%q_fp8_empty : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "fp8-e4m3-unit-e8m0-seven-codes-v1",
               merlin.input_domain = "bf16-seven-powers-and-zero-v1",
               merlin.scale_code = 127 : i32} {
      ^bb0(%value: bf16, %old: f8E4M3FN):
        %converted = arith.truncf %value : bf16 to f8E4M3FN
        linalg.yield %converted : f8E4M3FN
    } -> tensor<32x32xf8E4M3FN>
    %k_fp8_empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %k_fp8 = linalg.generic {
      indexing_maps = [affine_map<(m,k)->(m,k)>, affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%k : tensor<32x32xbf16>)
      outs(%k_fp8_empty : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "fp8-e4m3-unit-e8m0-seven-codes-v1",
               merlin.input_domain = "bf16-seven-powers-and-zero-v1",
               merlin.scale_code = 127 : i32} {
      ^bb0(%value: bf16, %old: f8E4M3FN):
        %converted = arith.truncf %value : bf16 to f8E4M3FN
        linalg.yield %converted : f8E4M3FN
    } -> tensor<32x32xf8E4M3FN>
    %score_empty = tensor.empty() : tensor<32x32xbf16>
    %score_init = linalg.fill ins(%zero : bf16) outs(%score_empty : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
    %scores = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%q_fp8, %k_fp8 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%score_init : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    func.return %scores : tensor<32x32xbf16>
  }
}
