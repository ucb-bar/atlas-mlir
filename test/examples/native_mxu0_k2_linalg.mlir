module {
  func.func @mxu0_k2_tile(
      %a0: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %w0: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %a1: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %w1: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<32x32xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %empty = tensor.empty() : tensor<32x32xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %first = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a0, %w0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
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
    %result = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a1, %w1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
      outs(%first : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<32x32xbf16>
    func.return %result : tensor<32x32xbf16>
  }
}
