module {
  func.func @add_mul(
    %a: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"},
    %b: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"},
    %c: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-bf16-product-normal-v1"})
      -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %sum_storage = tensor.empty() : tensor<32x32xbf16>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%sum_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
               merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %wide_sum = arith.addf %lf, %rf : f32
        %bits = arith.bitcast %wide_sum : f32 to i32
        %shift = arith.constant 16 : i32
        %high = arith.shrui %bits, %shift : i32
        %narrow = arith.trunci %high : i32 to i16
        %chopped = arith.bitcast %narrow : i16 to bf16
        linalg.yield %chopped : bf16
    } -> tensor<32x32xbf16>
    %product_storage = tensor.empty() : tensor<32x32xbf16>
    %product = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum, %c : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%product_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-product-rne-v1",
               merlin.input_domain = "finite-normal-bf16-product-normal-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %wide_product = arith.mulf %lf, %rf : f32
        %bits = arith.bitcast %wide_product : f32 to i32
        %shift = arith.constant 16 : i32
        %half = arith.constant 32767 : i32
        %one = arith.constant 1 : i32
        %high = arith.shrui %bits, %shift : i32
        %lsb = arith.andi %high, %one : i32
        %bias = arith.addi %half, %lsb : i32
        %biased = arith.addi %bits, %bias : i32
        %rounded = arith.shrui %biased, %shift : i32
        %narrow = arith.trunci %rounded : i32 to i16
        %bf16 = arith.bitcast %narrow : i16 to bf16
        linalg.yield %bf16 : bf16
    } -> tensor<32x32xbf16>
    func.return %sum, %product : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
