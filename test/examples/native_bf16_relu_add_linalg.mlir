module {
  func.func @relu_then_add(
    %a: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"},
    %b: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"})
      -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %relu_storage = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a : tensor<32x32xbf16>)
      outs(%relu_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%value: bf16, %old: bf16):
        %zero = arith.constant 0.000000e+00 : bf16
        %positive = arith.maximumf %value, %zero : bf16
        linalg.yield %positive : bf16
    } -> tensor<32x32xbf16>
    %sum_storage = tensor.empty() : tensor<32x32xbf16>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu, %b : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%sum_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
               merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %wide_sum = arith.addf %lf, %rf : f32
        %bits = arith.bitcast %wide_sum : f32 to i32
        %shift = arith.constant 16 : i32
        %hi = arith.shrui %bits, %shift : i32
        %narrow = arith.trunci %hi : i32 to i16
        %chopped = arith.bitcast %narrow : i16 to bf16
        linalg.yield %chopped : bf16
    } -> tensor<32x32xbf16>
    func.return %relu, %sum : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
