module {
  func.func @relu_pack_tile(%x: tensor<32x32xbf16> {merlin.input_domain = "bf16-seven-powers-and-zero-v1"}) -> tensor<32x32xf8E4M3FN> {
    %relu_empty = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<32x32xbf16>)
      outs(%relu_empty : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%value: bf16, %old: bf16):
        %zero = arith.constant 0.000000e+00 : bf16
        %positive = arith.maximumf %value, %zero : bf16
        linalg.yield %positive : bf16
      } -> tensor<32x32xbf16>
    %pack_empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %packed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu : tensor<32x32xbf16>)
      outs(%pack_empty : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "fp8-e4m3-unit-e8m0-seven-codes-v1",
               merlin.input_domain = "bf16-seven-powers-and-zero-v1",
               merlin.scale_code = 127 : i32} {
      ^bb0(%value: bf16, %old: f8E4M3FN):
        %converted = arith.truncf %value : bf16 to f8E4M3FN
        linalg.yield %converted : f8E4M3FN
      } -> tensor<32x32xf8E4M3FN>
    func.return %packed : tensor<32x32xf8E4M3FN>
  }
}
