module {
  func.func @forward(%lhs: tensor<1x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x2xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %result = linalg.matmul {prov.region_id = "small"}
      ins(%lhs, %rhs : tensor<1x2xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x2xf32>) -> tensor<1x2xf32>
    func.return %result : tensor<1x2xf32>
  }
}
