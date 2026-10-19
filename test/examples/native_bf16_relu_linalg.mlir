module {
  func.func @relu_tile(%input: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %empty = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<32x32xbf16>)
      outs(%empty : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%value: bf16, %old: bf16):
        %zero = arith.constant 0.000000e+00 : bf16
        %positive = arith.maximumf %value, %zero : bf16
        linalg.yield %positive : bf16
    } -> tensor<32x32xbf16>
    func.return %result : tensor<32x32xbf16>
  }
}
