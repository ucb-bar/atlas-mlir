builtin.module attributes {prov.weights_file = "weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%lhs: tensor<3x2xf32>, %rhs: tensor<2x35xf32>) -> tensor<3x35xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<3x35xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<3x35xf32>) -> tensor<3x35xf32>
    %result = linalg.matmul {prov.region_id = "large"}
      ins(%lhs, %rhs : tensor<3x2xf32>, tensor<2x35xf32>)
      outs(%init : tensor<3x35xf32>) -> tensor<3x35xf32>
    func.return %result : tensor<3x35xf32>
  }
}
