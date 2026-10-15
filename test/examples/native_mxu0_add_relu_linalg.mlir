module {
  func.func @mxu0_add_relu_tile(
    %a: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
    %w: tensor<32x32xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
    %bias: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"})
      -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %zero = arith.constant 0.000000e+00 : bf16
    %empty = tensor.empty() : tensor<32x32xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %matmul = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %w : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>)
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
    %sum_storage = tensor.empty() : tensor<32x32xbf16>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul, %bias : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%sum_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
               merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %fp32 = arith.addf %lf, %rf : f32
        %bits = arith.bitcast %fp32 : f32 to i32
        %shift = arith.constant 16 : i32
        %high = arith.shrui %bits, %shift : i32
        %narrow = arith.trunci %high : i32 to i16
        %bf16 = arith.bitcast %narrow : i16 to bf16
        linalg.yield %bf16 : bf16
    } -> tensor<32x32xbf16>
    %relu_storage = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum : tensor<32x32xbf16>)
      outs(%relu_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%value: bf16, %old: bf16):
        %zero_relu = arith.constant 0.000000e+00 : bf16
        %positive = arith.maximumf %value, %zero_relu : bf16
        linalg.yield %positive : bf16
    } -> tensor<32x32xbf16>
    func.return %sum, %relu : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
