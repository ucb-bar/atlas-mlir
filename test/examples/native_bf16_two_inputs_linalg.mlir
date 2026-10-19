module {
  func.func @two_inputs(%left: tensor<32x32xbf16>, %right: tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %empty0 = tensor.empty() : tensor<32x32xbf16>
    %first = linalg.copy ins(%left : tensor<32x32xbf16>)
      outs(%empty0 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %empty1 = tensor.empty() : tensor<32x32xbf16>
    %second = linalg.copy ins(%right : tensor<32x32xbf16>)
      outs(%empty1 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    func.return %first, %second : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
