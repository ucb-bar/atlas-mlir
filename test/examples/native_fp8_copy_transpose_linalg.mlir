module {
  func.func @copy_transpose(%input: tensor<32x32xf8E4M3FN>) -> (tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) {
    %empty0 = tensor.empty() : tensor<32x32xf8E4M3FN>
    %copy = linalg.copy ins(%input : tensor<32x32xf8E4M3FN>)
      outs(%empty0 : tensor<32x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN>
    %empty1 = tensor.empty() : tensor<32x32xf8E4M3FN>
    %transpose = linalg.transpose ins(%input : tensor<32x32xf8E4M3FN>)
      outs(%empty1 : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    func.return %copy, %transpose : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>
  }
}
