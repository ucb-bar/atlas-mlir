#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module attributes {prov.weights_file = "captured_mlp32_weights.safetensors"} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x32xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) permutation = [1, 0] 
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = linalg.matmul ins(%arg4, %transposed : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg1 : tensor<32x32xf32>, tensor<32xf32>) outs(%0 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.addf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.cmpf ugt, %in, %cst : f32
      %8 = arith.select %7, %in, %cst : f32
      linalg.yield %8 : f32
    } -> tensor<32x32xf32>
    %transposed_0 = linalg.transpose ins(%arg2 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) permutation = [1, 0] 
    %5 = linalg.matmul ins(%4, %transposed_0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %arg3 : tensor<32x32xf32>, tensor<32xf32>) outs(%0 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.addf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<32x32xf32>
    return %6 : tensor<32x32xf32>
  }
}
