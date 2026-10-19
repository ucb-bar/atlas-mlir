module {
  func.func @raw_and_relu(%input: tensor<32x32xbf16>)
      -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %raw_storage = tensor.empty() : tensor<32x32xbf16>
    %raw = linalg.copy ins(%input : tensor<32x32xbf16>)
      outs(%raw_storage : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %relu_storage = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<32x32xbf16>)
      outs(%relu_storage : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%value: bf16, %old: bf16):
        %zero = arith.constant 0.000000e+00 : bf16
        %positive = arith.maximumf %value, %zero : bf16
        linalg.yield %positive : bf16
    } -> tensor<32x32xbf16>
    func.return %raw, %relu : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
