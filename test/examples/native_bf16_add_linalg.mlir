module {
  func.func @add_tile(
    %a: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"},
    %b: tensor<32x32xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"}) -> tensor<32x32xbf16> {
    %empty = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>)
      outs(%empty : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
               merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
      ^bb0(%left: bf16, %right: bf16, %old: bf16):
        %lf = arith.extf %left : bf16 to f32
        %rf = arith.extf %right : bf16 to f32
        %sum = arith.addf %lf, %rf : f32
        %bits = arith.bitcast %sum : f32 to i32
        %shift = arith.constant 16 : i32
        %hi = arith.shrui %bits, %shift : i32
        %narrow = arith.trunci %hi : i32 to i16
        %chopped = arith.bitcast %narrow : i16 to bf16
        linalg.yield %chopped : bf16
    } -> tensor<32x32xbf16>
    func.return %result : tensor<32x32xbf16>
  }
}
