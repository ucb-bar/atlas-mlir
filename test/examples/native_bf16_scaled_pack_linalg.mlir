module {
  func.func @pack_tile(%x: tensor<32x32xbf16> {merlin.input_domain = "bf16-positive-zero-or-exact-normal-under-scale-122-128-v1"}) -> tensor<32x32xf8E4M3FN> {
    %empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %packed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<32x32xbf16>)
      outs(%empty : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "fp8-e4m3-scaled-e8m0-exact-normal-122-128-v1",
               merlin.input_domain = "bf16-positive-zero-or-exact-normal-under-scale-122-128-v1",
               merlin.scale_code = 122 : i32} {
      ^bb0(%value: bf16, %old: f8E4M3FN):
        %inverse_scale = arith.constant 3.200000e+01 : bf16
        %scaled = arith.mulf %value, %inverse_scale : bf16
        %converted = arith.truncf %scaled : bf16 to f8E4M3FN
        linalg.yield %converted : f8E4M3FN
    } -> tensor<32x32xf8E4M3FN>
    func.return %packed : tensor<32x32xf8E4M3FN>
  }
}
