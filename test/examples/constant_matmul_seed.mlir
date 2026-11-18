module {
  func.func @forward(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %positive = arith.constant dense<0.25> : tensor<32x32xf64>
    %negative = arith.constant dense<-0.25> : tensor<32x32xf64>
    %empty64 = tensor.empty() : tensor<32x32xf64>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%positive, %negative : tensor<32x32xf64>, tensor<32x32xf64>) outs(%empty64 : tensor<32x32xf64>) {
    ^bb0(%x: f64, %y: f64, %unused: f64):
      %value = arith.addf %x, %y : f64
      linalg.yield %value : f64
    } -> tensor<32x32xf64>
    %empty32 = tensor.empty() : tensor<32x32xf32>
    %seed = linalg.generic {
      indexing_maps = [affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum : tensor<32x32xf64>) outs(%empty32 : tensor<32x32xf32>) {
    ^bb0(%x: f64, %unused: f32):
      %value = arith.truncf %x : f64 to f32
      linalg.yield %value : f32
    } -> tensor<32x32xf32>
    // A source-only temporary must disappear before region placement.
    %unused_product = linalg.generic {
      indexing_maps = [affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%empty32 : tensor<32x32xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %value = arith.mulf %x, %y : f32
      linalg.yield %value : f32
    } -> tensor<32x32xf32>
    %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
                            outs(%seed : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}
