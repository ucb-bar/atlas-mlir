module {
  func.func @mxu0_pack_mxu0_tile(
      %a: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %first_w: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %second_w: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<32x32xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %empty = tensor.empty() : tensor<32x32xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
    %first = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %first_w : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%init : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    %pack_empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %packed = linalg.generic {
      indexing_maps = [affine_map<(m,k)->(m,k)>, affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%first : tensor<32x32xbf16>)
      outs(%pack_empty : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "fp8-e4m3-unit-e8m0-seven-codes-v1",
               merlin.input_domain = "bf16-seven-powers-and-zero-v1",
               merlin.scale_code = 127 : i32} {
      ^bb0(%value: bf16, %old: f8E4M3FN):
        %converted = arith.truncf %value : bf16 to f8E4M3FN
        linalg.yield %converted : f8E4M3FN
    } -> tensor<32x32xf8E4M3FN>
    %zero2 = arith.constant 0.000000e+00 : bf16
    %empty2 = tensor.empty() : tensor<32x32xbf16>
    %init2 = linalg.fill ins(%zero2 : bf16) outs(%empty2 : tensor<32x32xbf16>)
      -> tensor<32x32xbf16>
    %second = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%packed, %second_w : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%init2 : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    func.return %second : tensor<32x32xbf16>
  }
}
