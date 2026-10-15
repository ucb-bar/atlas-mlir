module {
  func.func @transpose(%input: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN> {
    %empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %result = linalg.transpose ins(%input : tensor<32x32xf8E4M3FN>)
      outs(%empty : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    func.return %result : tensor<32x32xf8E4M3FN>
  }
}
