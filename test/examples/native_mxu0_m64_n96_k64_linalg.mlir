module {
  func.func @mxu0_m64_n96_k64(
      %a: tensor<64x64xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %w: tensor<96x64xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<64x96xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %empty = tensor.empty() : tensor<64x96xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                       affine_map<(m,n,k)->(n,k)>,
                       affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %w : tensor<64x64xf8E4M3FN>, tensor<96x64xf8E4M3FN>)
      outs(%init : tensor<64x96xbf16>)
      attrs = {merlin.numeric_semantics = "mxu0-e4m3-ordered-bf16-finite-v1",
               merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
               merlin.reduction_order = "k-ascending"} {
      ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
        %ab = arith.extf %av : f8E4M3FN to bf16
        %wb = arith.extf %wv : f8E4M3FN to bf16
        %next = math.fma %ab, %wb, %acc : bf16
        linalg.yield %next : bf16
    } -> tensor<64x96xbf16>
    func.return %result : tensor<64x96xbf16>
  }
}
